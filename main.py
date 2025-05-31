from maim_message import (
    MessageServer,
    Router,
    RouteConfig,
    TargetConfig,
    MessageBase,
    Seg,
)
from src.config import Config
from src.plugins.base_tts_model import BaseTTSModel
from src.utils.audio_encode import encode_audio, encode_audio_stream
import asyncio
from typing import List, Tuple, Dict
import random
import importlib
from pathlib import Path


class TTSPipeline:
    tts_list: List[BaseTTSModel] = []

    def __init__(self, config_path: str):  # sourcery skip: dict-comprehension
        self.config: Config = Config(config_path)
        self.server = MessageServer(
            host=self.config.server.host,
            port=self.config.server.port,
        )

        # 设置路由
        route_config = {}
        for platform, url in self.config.routes.items():
            route_config[platform] = TargetConfig(url=url, token=None)

        self.router = Router(RouteConfig(route_config))

        self.server.register_message_handler(self.server_handle)
        self.router.register_class_handler(self.client_handle)

        # 按群/用户分组的文本缓冲队列和处理任务
        self.text_buffer_dict: Dict[str, asyncio.Queue[Tuple[str, MessageBase]]] = {}
        self.buffer_task_dict: Dict[str, asyncio.Task] = {}
        self.buffer_timeout: int = 2  # 默认2秒

    def import_module(self):
        """动态导入TTS适配"""
        for tts in self.config.enabled_plugin.enabled:
            # 动态导入模块
            module_name = f"src.plugins.{tts}"
            try:
                module = importlib.import_module(module_name)
                tts_class: BaseTTSModel = module.TTSModel()
                self.tts_list.append(tts_class)
            except ImportError as e:
                print(f"Error importing {module_name}: {e}")
                raise
            except AttributeError as e:
                print(f"Error accessing TTSModel in {module_name}: {e}")
                raise
            except Exception as e:
                print(f"Unexpected error importing {module_name}: {e}")
                raise

    async def start(self):
        """启动服务器和路由，并导入设定的模块"""
        self.import_module()
        # 创建任务而不是直接返回 gather 结果
        self.server_task = asyncio.create_task(self.server.run())
        self.router_task = asyncio.create_task(self.router.run())
        # 返回任务以便外部可以等待或取消
        return self.server_task, self.router_task

    async def server_handle(self, message_data: dict):
        """处理服务器收到的消息"""
        message = MessageBase.from_dict(message_data)
        if message.message_info.format_info:
            if "voice" in message.message_info.format_info.accept_format:
                message.message_info.format_info.accept_format.append("tts_text")
        await self.router.send_message(message)

    def process_seg(self, seg: Seg) -> str:
        """处理消息段，提取文本内容"""
        message_text = ""
        if seg.type == "seglist":
            for s in seg.data:
                message_text += self.process_seg(s)
        if seg.type == "tts_text":
            message_text += seg.data
        return message_text

    async def client_handle(self, message_dict: dict) -> None:
        # sourcery skip: remove-redundant-if
        """处理客户端收到的消息并进行TTS转换（分群缓冲）"""
        message = MessageBase.from_dict(message_dict)
        stream_mode = self.config.tts_base_config.stream_mode
        if stream_mode:
            await self.send_voice_stream(message)
            return

        message_text = self.process_seg(message.message_segment)
        if message_text == "":
            # 非文本消息直接透传
            await self.server.send_message(message)
            return

        if not message_text:
            print("处理文本为空，跳过发送")
            return

        # 获取分组ID（优先群id，否则用户id）
        group_id = getattr(message.message_info.group_info, "group_id", None)
        if group_id is None:
            print("没有群消息id使用用户id代替")
            group_id = getattr(message.message_info.user_info, "user_id", None)
        if not group_id:
            print("无法定位目标发送位置，跳过TTS处理")
            await self.server.send_message(message)
            return
        group_id = str(group_id)

        # 保证队列存在
        if group_id not in self.text_buffer_dict:
            self.text_buffer_dict[group_id] = asyncio.Queue()
        # 创建处理任务
        if group_id not in self.buffer_task_dict:
            self.buffer_task_dict[group_id] = asyncio.create_task(self._buffer_queue_handler(group_id))
        # 将文本加入队列
        await self.text_buffer_dict[group_id].put((message_text, message))

    async def _buffer_queue_handler(self, group_id: str) -> None:
        """处理每个群/用户的缓冲队列，定时合成语音并发送"""
        latest_message_obj: MessageBase = None
        while True:
            try:
                message_text, message_obj = await asyncio.wait_for(
                    self.text_buffer_dict[group_id].get(), timeout=self.buffer_timeout
                )
                latest_message_obj = message_obj
            except asyncio.TimeoutError:  # 向下兼容3.10与3.11
                print("等待结束，进入处理")
                break
            except TimeoutError:  # 支持3.12及以上版本
                print("等待结束，进入处理")
                break
            except Exception as e:
                print(f"处理缓冲队列时发生错误: {str(e)}")
                raise
        if not message_text or not latest_message_obj:
            print("数据为空，跳过处理")
            await self.cleanup_task(group_id)
            return
        text: str = message_text.strip()
        print(f"[聊天: {group_id}]缓冲区合成文本:", text)
        message = latest_message_obj
        new_seg = await self.get_voice_no_stream(text, message.message_info.platform)
        if not new_seg:
            print("语音消息为空，跳过发送")
            await self.cleanup_task(group_id)
            return
        message.message_segment = new_seg
        message.message_info.format_info.content_format = ["voice"]
        if not message.message_info.additional_config:
            message.message_info.additional_config = {}
        message.message_info.additional_config["original_text"] = text
        await self.server.send_message(message)
        await self.cleanup_task(group_id)
        return

    async def cleanup_task(self, group_id: str):
        task = self.buffer_task_dict.pop(group_id)
        task.cancel()

    async def get_voice_no_stream(self, text: str, platform: str) -> Seg | None:
        """获取语音消息段"""
        if not self.tts_list:
            print("没有启用任何tts，跳过处理")
            return None
        # tts_class = random.choice(self.tts_list)
        tts_class = self.tts_list[0]
        try:
            # 使用非流式TTS
            audio_data = await tts_class.tts(text=text, platform=platform)
            # 对整个音频数据进行base64编码
            encoded_audio = encode_audio(audio_data)
            # 创建语音消息
            return Seg(type="voice", data=encoded_audio)
        except Exception as e:
            print(f"TTS处理过程中发生错误: {str(e)}")
            print(f"文本为: {text}")
            return None

    async def send_voice_stream(self, message: MessageBase) -> None:
        """流式发送语音消息"""
        platform = message.message_info.platform
        message_text = self.process_seg(message.message_segment)
        if not message_text:
            print("处理文本为空，跳过发送")
            return
        text = message_text
        if not self.tts_list:
            print("没有启用任何tts，跳过处理")
            return None
        # tts_class = random.choice(self.tts_list)
        tts_class = self.tts_list[0]
        try:
            audio_stream = await tts_class.tts_stream(text=text, platform=platform)
            # 从音频流中读取和处理数据
            for chunk in audio_stream:
                if chunk:  # 确保chunk不为空
                    try:
                        # 对音频数据进行base64编码
                        encoded_chunk = encode_audio_stream(chunk)
                        # 创建语音消息
                        new_seg = Seg(type="voice_stream", data=encoded_chunk)
                        message.message_segment = new_seg
                        message.message_info.format_info.content_format = ["voice_stream"]
                        if not message.message_info.additional_config:
                            message.message_info.additional_config = {}
                        message.message_info.additional_config["original_text"] = text

                        # 发送到下游
                        await self.server.send_message(message)
                    except Exception as e:
                        print(f"处理音频块时发生错误: {str(e)}")
                        continue
            print("流式语音消息发送完成")
        except Exception as e:
            print(f"TTS处理过程中发生错误: {str(e)}")
            print(f"文本为: {text}")
            return None

    async def stop(self):
        """停止服务器和路由"""
        print("正在停止TTS服务...")
        # 停止所有正在运行的缓冲任务
        for _, task in list(self.buffer_task_dict.items()):
            if not task.done() and not task.cancelled():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"取消缓冲任务时出错: {e}")

        # 安全地停止服务器和路由器
        try:
            await self.server.stop()
        except Exception as e:
            print(f"停止服务器时发生错误: {e}")

        try:
            await self.router.stop()
        except Exception as e:
            print(f"停止路由器时发生错误: {e}")

        # 如果有任务属性，取消这些任务
        if hasattr(self, "server_task") and not self.server_task.done():
            self.server_task.cancel()

        if hasattr(self, "router_task") and not self.router_task.done():
            self.router_task.cancel()

        print("TTS服务已停止")


if __name__ == "__main__":
    config_path = Path(__file__).parent / "configs" / "base.toml"
    pipeline = TTSPipeline(str(config_path))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 定义信号处理函数和主程序
    shutdown_requested = False

    async def shutdown():
        """安全地关闭服务"""
        try:
            print("正在关闭服务...")
            await pipeline.stop()
            print("服务已关闭")
            return True
        except Exception as e:
            print(f"关闭服务时发生错误: {str(e)}")
            return False

    async def main():
        """主程序"""
        try:
            # 启动服务
            server_task, router_task = await pipeline.start()

            # 等待直到程序终止（如Ctrl+C）
            while not shutdown_requested:
                await asyncio.sleep(0.5)

            # 安全关闭
            await shutdown()

        except asyncio.CancelledError:
            print("主任务被取消")
            await shutdown()
        except Exception as e:
            print(f"运行中发生错误: {str(e)}")
            await shutdown()

    try:
        # 设置信号处理
        import signal

        # 定义信号处理函数
        def handle_signal():
            global shutdown_requested
            print("接收到中断信号，准备关闭...")
            shutdown_requested = True

        # 注册信号处理
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

        # 运行主程序
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("接收到键盘中断...")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 确保服务被关闭
        if not loop.is_closed():
            try:
                # 带超时的关闭服务
                loop.run_until_complete(asyncio.wait_for(shutdown(), timeout=5.0))
            except:
                print("关闭服务超时或出错")

            # 取消所有未完成的任务
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()

                # 等待任务取消
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except:
                    pass

            # 关闭事件循环
            loop.close()
