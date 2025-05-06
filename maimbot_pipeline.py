from maim_message import (
    MessageServer,
    Router,
    RouteConfig,
    TargetConfig,
    MessageBase,
    Seg,
)
from utils.config import get_default_config, Config
from tts_model import TTSModel
# 导入大模型音频生成模块
from omni_tts import OmniTTS
import base64
import io
import wave
import asyncio
from typing import List, Tuple, Dict
import random
import os


class TTSPipeline:
    def __init__(self, config_path: str = None):
        # 加载配置
        if config_path:
            self.config: Config = Config(config_path)
        else:
            self.config: Config = get_default_config()
            
        # 检查是否启用大模型TTS
        self.use_omni_tts = hasattr(self.config, "omni_tts") and self.config.omni_tts.enabled
        
        # 初始化大模型TTS API
        self.omni_tts = None
        if self.use_omni_tts:
            api_key = self.config.omni_tts.api_key or os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                print("警告: 未找到大模型API密钥，请在配置中设置api_key或设置环境变量DASHSCOPE_API_KEY")
            else:
                # 获取后处理配置
                enable_post_processing = False
                volume_reduction_db = 0
                noise_level = 0
                
                if hasattr(self.config.omni_tts, "post_processing"):
                    enable_post_processing = getattr(self.config.omni_tts.post_processing, "enabled", False)
                    volume_reduction_db = getattr(self.config.omni_tts.post_processing, "volume_reduction", 0)
                    noise_level = getattr(self.config.omni_tts.post_processing, "noise_level", 0)
                
                self.omni_tts = OmniTTS(
                    api_key=api_key, 
                    model_name=self.config.omni_tts.model_name,
                    voice=self.config.omni_tts.voice,
                    format=self.config.omni_tts.format,
                    enable_post_processing=enable_post_processing,
                    volume_reduction_db=volume_reduction_db,
                    noise_level=noise_level
                )
                print(f"已初始化大模型TTS: {self.config.omni_tts.model_name}")
                if enable_post_processing:
                    print(f"已启用音频后处理: 音量-{volume_reduction_db}dB, 杂音强度{noise_level*100:.1f}%")

        # 初始化TTS模型 (仅在不使用大模型或需要同时支持两种方式时)
        self.tts_model = None
        if not self.use_omni_tts or self.config.pipeline.keep_original_tts:
            try:
                self.tts_model = TTSModel(
                    config=self.config, host=self.config.tts.host, port=self.config.tts.port
                )
                
                # 设置默认参考音频
                if self.config.tts.ref_audio_path and self.config.tts.prompt_text:
                    self.tts_model.set_refer_audio(
                        audio_path=self.config.tts.ref_audio_path,
                        prompt_text=self.config.tts.prompt_text,
                    )
                    
                # 加载默认预设
                self.tts_model.load_preset(self.config.pipeline.default_preset)
                print("已初始化原始TTS模型")
            except Exception as e:
                print(f"初始化原始TTS模型失败: {str(e)}")
                if not self.use_omni_tts:
                    print("警告: 原始TTS模型初始化失败且未启用大模型TTS，语音合成功能将不可用")
                self.tts_model = None

        # 初始化服务器
        self.server = MessageServer(
            host=self.config.server.host, port=self.config.server.port
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

    def start(self):
        """启动服务器和路由"""
        return asyncio.gather(
            self.server.run(),
            self.router.run(),
        )

    def get_platform_preset(self, platform: str) -> str:
        """获取平台对应的预设名称

        Args:
            platform: 平台名称

        Returns:
            预设名称，如果平台没有指定预设则返回默认预设
        """
        return self.config.pipeline.platform_presets.get(
            platform, self.config.pipeline.default_preset
        )

    def encode_audio(self, audio_data: bytes, media_type: str = "wav") -> str:
        """对音频数据进行base64编码

        Args:
            audio_data: 原始音频数据
            media_type: 音频格式，默认为wav

        Returns:
            base64编码后的音频数据
        """
        # 确保音频是wav格式
        if media_type != "wav":
            # 先将音频数据转换为wav格式
            with io.BytesIO(audio_data) as audio_io:
                with wave.open(audio_io, "rb") as wav_file:
                    # 获取音频参数
                    channels = wav_file.getnchannels()
                    width = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())

                    # 创建新的wav文件
                    with io.BytesIO() as wav_io:
                        with wave.open(wav_io, "wb") as new_wav:
                            new_wav.setnchannels(channels)
                            new_wav.setsampwidth(width)
                            new_wav.setframerate(framerate)
                            new_wav.writeframes(frames)
                        audio_data = wav_io.getvalue()

        # base64编码
        return base64.b64encode(audio_data).decode("utf-8")

    def encode_audio_stream(self, audio_chunk: bytes) -> str:
        """对音频数据块进行base64编码

        Args:
            audio_chunk: 音频数据块

        Returns:
            base64编码后的数据
        """
        return base64.b64encode(audio_chunk).decode("utf-8")

    def process_seg(self, seg: Seg) -> Tuple[List[str], bool, bool]:
        have_text = False
        have_other = False
        message_text = []
        if seg.type == "seglist":
            for s in seg.data:
                zip_content = self.process_seg(s)
                message_text += zip_content[0]
                have_text = have_text or zip_content[1]
                have_other = have_other or zip_content[2]
        if seg.type == "text":
            message_text.append(seg.data)
            have_text = True
        else:
            # 标记含有其他类型的消息
            have_other = True
        return message_text, have_text, have_other

    async def client_handle(self, message_dict: dict) -> None:
        """处理客户端收到的消息并进行TTS转换（分群缓冲）"""
        message = MessageBase.from_dict(message_dict)
        streaming_mode = (
            self.config.tts.streaming_mode
            if hasattr(self.config.tts, "streaming_mode")
            else False
        )
        
        # 检查是否有可用的TTS模型
        if not self.omni_tts and not self.tts_model:
            print("警告: 没有可用的TTS模型，跳过语音处理")
            await self.server.send_message(message)
            return
            
        if streaming_mode:
            await self.send_voice_stream(message)
            return

        message_text, have_text, have_other = self.process_seg(message.message_segment)
        if have_other and not have_text:
            # 非文本消息直接透传
            await self.server.send_message(message)
            return
        elif have_other and have_text:
            print("检测到混合类型消息，丢弃其他类型")

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
            self.buffer_task_dict[group_id] = asyncio.create_task(
                self._buffer_queue_handler(group_id)
            )
        # 将文本加入队列
        await self.text_buffer_dict[group_id].put((message_text, message))

    async def _buffer_queue_handler(self, group_id: str) -> None:
        """处理每个群/用户的缓冲队列，定时合成语音并发送"""
        buffer: List[str] = []
        latest_message_obj: MessageBase = None
        while True:
            try:
                message_text, message_obj = await asyncio.wait_for(
                    self.text_buffer_dict[group_id].get(), timeout=self.buffer_timeout
                )
                buffer.extend(message_text)
                latest_message_obj = message_obj
            except asyncio.TimeoutError:
                print("等待结束，进入处理")
                break
        if not buffer or not latest_message_obj:
            print("数据为空，跳过处理")
            await self.cleanup_task(group_id)
            return
        if random.random() > self.config.probability.voice_probability:
            # 使用临时的原样发送方式
            print("发送原文本")
            await self.temporary_send_method(latest_message_obj, buffer, group_id)
            return
        text: str = ",".join(buffer)
        print(f"[聊天: {group_id}]缓冲区合成文本:", text)
        message = latest_message_obj
        platform = message.message_info.platform
        preset_name = self.get_platform_preset(platform)
        
        # 尝试加载预设
        try:
            if self.tts_model and self.tts_model._current_preset != preset_name:
                self.tts_model.load_preset(preset_name)
        except Exception as e:
            print(f"加载预设失败: {e}")
            
        # 尝试生成语音
        try:
            print("开始生成语音...")
            new_seg = await self.get_voice_no_stream(text)
            if not new_seg:
                print("语音消息为空，跳过发送")
                await self.cleanup_task(group_id)
                return
            print("语音生成成功，准备发送")
            message.message_segment = new_seg
            message.message_info.format_info.content_format = ["voice"]
            if not message.message_info.additional_config:
                message.message_info.additional_config = {}
            message.message_info.additional_config["original_text"] = text
            print("正在发送语音消息...")
            await self.server.send_message(message)
            print("语音消息发送完成")
        except Exception as e:
            import traceback
            print(f"语音生成或发送过程中发生错误: {str(e)}")
            print(f"详细错误信息: {traceback.format_exc()}")
            # 如果语音生成失败，尝试发送原文本
            print("尝试发送原文本作为备选...")
            await self.temporary_send_method(latest_message_obj, buffer, group_id)
        
        await self.cleanup_task(group_id)
        return

    async def cleanup_task(self, group_id: str):
        task = self.buffer_task_dict.pop(group_id, "没有对应的键")
        task.cancel()

    async def get_voice_no_stream(self, text: str):
        try:
            # 使用大模型生成音频
            if self.use_omni_tts and self.omni_tts:
                audio_data = await self.omni_tts.generate_audio(text)
            elif self.tts_model:
                # 使用非流式TTS
                audio_data = self.tts_model.tts(text=text)
            else:
                print("错误: 没有可用的TTS模型")
                return None
                
            # 对整个音频数据进行base64编码
            encoded_audio = self.encode_audio(audio_data)
            # 创建语音消息
            new_seg = Seg(type="voice", data=encoded_audio)
            return new_seg
        except Exception as e:
            print(f"TTS处理过程中发生错误: {str(e)}")
            print(f"文本为: {text}")
            return None

    async def temporary_send_method(
        self, message: MessageBase, text_list: List[str], group_id: str
    ) -> None:
        """临时使用的原样发送函数"""
        for text in text_list:
            new_seg = Seg(type="text", data=text)
            message.message_segment = new_seg
            await self.server.send_message(message)
        await self.cleanup_task(group_id)

    async def send_voice_stream(self, message: MessageBase) -> None:
        """流式发送语音消息"""
        # 根据平台切换预设
        platform = message.message_info.platform
        preset_name = self.get_platform_preset(platform)
        if self.tts_model and self.tts_model._current_preset != preset_name:
            self.tts_model.load_preset(preset_name)

        message_text, have_text, have_other = self.process_seg(message.message_segment)
        if not message_text:
            print("处理文本为空，跳过发送")
            return
        text = ",".join(message_text)
        try:
            # 使用大模型流式生成
            if self.use_omni_tts and self.omni_tts:
                audio_stream = await self.omni_tts.generate_audio_stream(text)
            elif self.tts_model:
                # 使用原TTS模型
                audio_stream = self.tts_model.tts_stream(text=text)
            else:
                print("错误: 没有可用的TTS模型")
                return
                
            # 从音频流中读取和处理数据
            for chunk in audio_stream:
                if chunk:  # 确保chunk不为空
                    try:
                        # 对音频数据进行base64编码
                        encoded_chunk = self.encode_audio_stream(chunk)

                        # 创建语音消息
                        new_seg = Seg(type="voice", data=encoded_chunk)
                        message.message_segment = new_seg
                        message.message_info.format_info.content_format = ["voice"]
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

    async def server_handle(self, message_data: dict):
        """处理服务器收到的消息"""
        message = MessageBase.from_dict(message_data)
        # print(f"Received message from server: {message_data}")
        await self.router.send_message(message)


if __name__ == "__main__":
    pipeline = TTSPipeline()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(pipeline.start())
