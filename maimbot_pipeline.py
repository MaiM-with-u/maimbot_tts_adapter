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

        # 设置路由 - 从routes配置创建TargetConfig
        route_config = {}
        for platform, url in self.config.routes.items():
            route_config[platform] = TargetConfig(url=url, token=None)

        self.router = Router(RouteConfig(route_config))

    def tts(self, text, ref_audio_path=None, **kwargs):
        """执行TTS转换"""
        return self.tts_model.tts(text, ref_audio_path, **kwargs)

    def start(self):
        """启动服务"""
        self.server.start()

    def set_ref_aud(self, audio_path, prompt_text):
        """设置参考音频"""
        self.tts_model.set_refer_audio(audio_path, prompt_text)

    def handle(self, message_dict: dict):
        """处理消息"""
        message = MessageBase.from_dict(message_dict)
        message_text = []

        def process_seg(seg: Seg):
            if seg.type == "seglist":
                for s in seg.data:
                    process_seg(s)
            if seg.type == "text":
                message_text.append(seg.data)

        process_seg(message.message_segment)
        text = " ".join(message_text)

        # 使用配置中的TTS参数
        audio_data = self.tts_model.tts(
            text=text,
            ref_audio_path=self.config.tts.ref_audio_path,
            prompt_text=self.config.tts.prompt_text,
            aux_ref_audio_paths=self.config.tts.aux_ref_audio_paths,
            text_lang=self.config.tts.text_language,
            prompt_lang=self.config.tts.prompt_language,
            media_type=self.config.tts.media_type,
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

        new_seg = Seg(type='seglist', data=[
            message.message_segment,
            Seg(type='voice', data=audio_data)
        ])
        message.message_segment = new_seg
        message.message_info.format_info.content_format.append('voice')
                
        return audio_data


if __name__ == "__main__":
    pipeline = TTSPipeline()
    pipeline.start()
