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


class TTSPipeline:
    def __init__(self, config_path: str = None):
        # 加载配置
        if config_path:
            self.config: Config = Config(config_path)
        else:
            self.config: Config = get_default_config()

        # 初始化TTS模型
        self.tts_model = TTSModel(
            config=self.config,
            host=self.config.tts.host,
            port=self.config.tts.port
        )

        # 设置默认参考音频
        if self.config.tts.ref_audio_path and self.config.tts.prompt_text:
            self.tts_model.set_refer_audio(
                audio_path=self.config.tts.ref_audio_path,
                prompt_text=self.config.tts.prompt_text
            )

        # 初始化服务器
        self.server = MessageServer(
            host=self.config.server.host,
            port=self.config.server.port
        )

        # 设置路由
        route_config = {}
        for platform, url in self.config.routes.items():
            route_config[platform] = TargetConfig(url=url, token=None)

        self.router = Router(RouteConfig(route_config))

        # 加载默认预设
        self.tts_model.load_preset(self.config.pipeline.default_preset)

    def tts(self, text, ref_audio_path=None, **kwargs):
        """执行TTS转换"""
        return self.tts_model.tts(text, ref_audio_path, **kwargs)

    def set_ref_aud(self, audio_path, prompt_text):
        """设置参考音频"""
        self.tts_model.set_refer_audio(audio_path, prompt_text)

    def get_platform_preset(self, platform: str) -> str:
        """获取平台对应的预设名称
        
        Args:
            platform: 平台名称
            
        Returns:
            预设名称，如果平台没有指定预设则返回默认预设
        """
        return self.config.pipeline.platform_presets.get(
            platform, 
            self.config.pipeline.default_preset
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
                with wave.open(audio_io, 'rb') as wav_file:
                    # 获取音频参数
                    channels = wav_file.getnchannels()
                    width = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    
                    # 创建新的wav文件
                    with io.BytesIO() as wav_io:
                        with wave.open(wav_io, 'wb') as new_wav:
                            new_wav.setnchannels(channels)
                            new_wav.setsampwidth(width)
                            new_wav.setframerate(framerate)
                            new_wav.writeframes(frames)
                        audio_data = wav_io.getvalue()

        # base64编码
        return base64.b64encode(audio_data).decode('utf-8')

    def handle(self, message_dict: dict):
        """处理消息
        
        会根据消息来源平台自动切换对应的预设
        """
        message = MessageBase.from_dict(message_dict)
        message_text = []
        
        # 根据平台切换预设
        platform = message.message_info.platform
        preset_name = self.get_platform_preset(platform)
        self.tts_model.load_preset(preset_name)

        def process_seg(seg: Seg):
            if seg.type == "seglist":
                for s in seg.data:
                    process_seg(s)
            if seg.type == "text":
                message_text.append(seg.data)

        process_seg(message.message_segment)
        text = " ".join(message_text)

        # 使用配置中的TTS参数，强制使用wav格式
        audio_data = self.tts_model.tts(
            text=text,
            ref_audio_path=self.config.tts.ref_audio_path,
            prompt_text=self.config.tts.prompt_text,
            aux_ref_audio_paths=self.config.tts.aux_ref_audio_paths,
            text_lang=self.config.tts.text_language,
            prompt_lang=self.config.tts.prompt_language,
            media_type="wav",  # 强制使用wav格式
            streaming_mode=self.config.tts.streaming_mode,
            temperature=self.config.tts.temperature,
            top_k=self.config.tts.top_k,
            top_p=self.config.tts.top_p,
            batch_size=self.config.tts.batch_size,
            batch_threshold=self.config.tts.batch_threshold,
            speed_factor=self.config.tts.speed_factor,
            text_split_method=self.config.tts.text_split_method,
            repetition_penalty=self.config.tts.repetition_penalty,
            sample_steps=self.config.tts.sample_steps,
            super_sampling=self.config.tts.super_sampling
        )

        # 对音频数据进行base64编码
        encoded_audio = self.encode_audio(audio_data)

        new_seg = Seg(type='seglist', data=[
            message.message_segment,
            Seg(type='voice', data=encoded_audio)
        ])
        message.message_segment = new_seg
        message.message_info.format_info.content_format.append('voice')
                
        return encoded_audio


if __name__ == "__main__":
    pipeline = TTSPipeline()
    pipeline.start()
