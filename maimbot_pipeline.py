from maim_message import (
    MessageServer,
    Router,
    RouteConfig,
    TargetConfig,
    MessageBase,
    Seg,
)
from utils.config import get_default_config, load_config, Config
from tts_model import TTSModel
import base64
import io
import wave
import asyncio


class TTSPipeline:
    def __init__(self, config_path: str = None):
        # 加载配置
        if config_path:
            self.config: Config = Config(config_path)
        else:
            self.config: Config = get_default_config()

        # 初始化TTS模型
        self.tts_model = TTSModel(
            config=self.config, host=self.config.tts.host, port=self.config.tts.port
        )

        # 设置默认参考音频
        if self.config.tts.ref_audio_path and self.config.tts.prompt_text:
            self.tts_model.set_refer_audio(
                audio_path=self.config.tts.ref_audio_path,
                prompt_text=self.config.tts.prompt_text,
            )

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

        # 加载默认预设
        self.tts_model.load_preset(self.config.pipeline.default_preset)

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

    async def client_handle(self, message_dict: dict):
        """根据streaming_mode参数处理消息"""
        # print(f"Received message from client: {message_dict}")
        message = MessageBase.from_dict(message_dict)

        # 发送原始消息到服务器
        # await self.server.send_message(message)
        if (
            not message.message_info.additional_config
            or not message.message_info.additional_config.get("allow_tts", False)
        ):
            print("跳过TTS处理")
            return

        message_text = []

        # 根据平台切换预设
        platform = message.message_info.platform
        preset_name = self.get_platform_preset(platform)
        if self.tts_model._current_preset != preset_name:
            self.tts_model.load_preset(preset_name)

        def process_seg(seg: Seg):
            if seg.type == "seglist":
                for s in seg.data:
                    process_seg(s)
            if seg.type == "text":
                message_text.append(seg.data)

        process_seg(message.message_segment)
        text = ",".join(message_text)
        print("处理文本:", text)

        try:
            # 检查配置中的streaming_mode设置
            streaming_mode = (
                self.config.tts.streaming_mode
                if hasattr(self.config.tts, "streaming_mode")
                else False
            )

            if streaming_mode:
                # 使用流式TTS
                audio_stream = self.tts_model.tts_stream(text=text)

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
                            message.message_info.additional_config["original_text"] = (
                                text
                            )

                            # 发送到下游
                            await self.server.send_message(message)
                        except Exception as e:
                            print(f"处理音频块时发生错误: {str(e)}")
                            continue

                print("流式语音消息发送完成")

            else:
                # 使用非流式TTS
                audio_data = self.tts_model.tts(text=text)

                # 对整个音频数据进行base64编码
                encoded_audio = self.encode_audio(audio_data)

                # 创建语音消息
                new_seg = Seg(type="voice", data=encoded_audio)
                message.message_segment = new_seg
                message.message_info.format_info.content_format = ["voice"]
                message.message_info.additional_config["original_text"] = text

                # 发送到下游
                await self.server.send_message(message)
                print("非流式语音消息发送完成")

        except Exception as e:
            print(f"TTS处理过程中发生错误: {str(e)}")
            # 可以在这里添加错误处理逻辑，比如发送错误消息给客户端

    async def server_handle(self, message_data: dict):
        """处理服务器消息"""
        message = MessageBase.from_dict(message_data)
        # print(f"Received message from server: {message_data}")
        await self.router.send_message(message)


if __name__ == "__main__":
    pipeline = TTSPipeline()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(pipeline.start())
