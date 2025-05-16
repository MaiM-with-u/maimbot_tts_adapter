import requests
import aiohttp
from typing import Dict, Any, List
from pathlib import Path
from src.plugins.base_tts_model import BaseTTSModel
from .tts_config import TTSBaseConfig, TTSPreset


class TTSModel(BaseTTSModel):
    def __init__(self):
        """初始化TTS模型"""
        self.config = self.load_config()
        if not self.config:
            raise ValueError("配置文件不存在或加载失败")
        self.host = self.config.tts.host
        self.port = self.config.tts.port

        self.base_url = f"http://{self.host}:{self.port}"
        self._ref_audio_path: str = None  # 存储当前使用的参考音频路径
        self._prompt_text: str = ""  # 存储当前使用的提示文本
        self._current_preset: str = "default"  # 当前使用的角色预设名称
        self._initialized: bool = False  # 标记是否已完成初始化
        self.initialize()

    def load_config(self) -> "TTSBaseConfig":
        """加载配置文件"""
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "gpt-sovits.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return TTSBaseConfig(str(config_path))

    def initialize(self) -> None:
        """初始化模型和预设

        如果已经初始化过，则跳过
        """
        if self._initialized:
            return
        self._initialized = True

        if self.config:
            if self.config.tts.models.gpt_model:
                self.set_gpt_weights(self.config.tts.models.gpt_model)
            if self.config.tts.models.sovits_model:
                self.set_sovits_weights(self.config.tts.models.sovits_model)

        # 设置默认角色预设
        if self.config:
            self.load_preset("default")
            self._current_preset = "default"

    @property
    def ref_audio_path(self) -> str | None:
        """获取当前使用的参考音频路径"""
        return self._ref_audio_path

    @property
    def prompt_text(self) -> str | None:
        """获取当前使用的提示文本"""
        return self._prompt_text

    @property
    def current_preset(self) -> str | None:
        """获取当前使用的角色预设名称"""
        return self._current_preset

    def get_preset(self, preset_name: str) -> TTSPreset | None:
        """获取指定名称的角色预设配置

        Args:
            preset_name: 预设名称

        Returns:
            预设配置字典，如果不存在则返回None
        """
        if not self.config:
            return None

        presets = self.config.tts.models.presets
        return presets.get(preset_name)

    def load_preset(self, preset_name: str) -> None:
        """加载指定的角色预设

        Args:
            preset_name: 预设名称

        Raises:
            ValueError: 当预设不存在时抛出
        """
        if not self._initialized:
            self.initialize()
        preset = self.get_preset(preset_name)
        if not preset:
            raise ValueError(f"预设 {preset_name} 不存在")

        # 设置参考音频和提示文本
        self.set_refer_audio(preset.ref_audio, preset.prompt_text)

        # 如果预设指定了模型，则切换模型
        if preset.gpt_model:
            self.set_gpt_weights(preset.gpt_model)
        if preset.sovits_model:
            self.set_sovits_weights(preset.sovits_model)

        self._current_preset = preset_name

    def get_platform_preset(self, platform: str) -> str:
        """获取指定平台的角色预设配置

        Args:
            platform: 平台名称

        Returns:
            预设配置字典，如果不存在则返回None
        """
        preset = self.config.pipeline.platform_presets.get(platform)
        if not preset:
            print(f"平台 {platform} 没有指定预设，使用默认预设")
            return self.config.pipeline.default_preset
        return preset

    def set_refer_audio(self, audio_path: str, prompt_text: str) -> None:
        """设置参考音频和对应的提示文本

        Args:
            audio_path: 音频文件路径
            prompt_text: 对应的提示文本，必须提供

        Raises:
            ValueError: 当参数无效时抛出异常
        """
        if not audio_path:
            raise ValueError("audio_path不能为空")
        if not prompt_text:
            raise ValueError("prompt_text不能为空")

        # if not os.path.exists(audio_path):
        #     raise ValueError(f"音频文件不存在: {audio_path}")

        self._ref_audio_path = audio_path
        self._prompt_text = prompt_text

    def set_gpt_weights(self, weights_path) -> None:
        """
        设置GPT权重

        Args:
            weights_path: 权重文件路径
        Raises:
            Exception: 当设置gpt weights失败时抛出异常
        """
        # if not os.path.exists(weights_path):
        #     raise ValueError(f"GPT模型文件不存在: {weights_path}")

        response = requests.get(f"{self.base_url}/set_gpt_weights", params={"weights_path": weights_path})
        if response.status_code != 200:
            raise Exception(response.json()["message"])

    def set_sovits_weights(self, weights_path):
        """
        设置SoVITS权重

        Args:
            weights_path: 权重文件路径
        Raises:
            Exception: 当设置sovits weights失败时抛出异常
        """
        # if not os.path.exists(weights_path):
        #     raise ValueError(f"SoVITS模型文件不存在: {weights_path}")

        response = requests.get(f"{self.base_url}/set_sovits_weights", params={"weights_path": weights_path})
        if response.status_code != 200:
            raise Exception(response.json()["message"])

    def build_parameters(
        self,
        text: str,
        ref_audio_path: str = None,
        aux_ref_audio_paths: List[str] = None,
        text_lang: str = None,
        prompt_text: str = None,
        prompt_lang: str = None,
        top_k: int = None,
        top_p: float = None,
        temperature: float = None,
        text_split_method: str = None,
        batch_size: int = None,
        batch_threshold: float = None,
        speed_factor: float = None,
        streaming_mode: bool = None,
        media_type: str = None,
        repetition_penalty: float = None,
        sample_steps: int = None,
        super_sampling: bool = None,
    ) -> Dict[str, Any]:
        """构建请求参数"""
        if not self._initialized:
            self.initialize()

        # 优先使用传入的ref_audio_path和prompt_text,否则使用持久化的值
        ref_audio_path = ref_audio_path or self._ref_audio_path
        if not ref_audio_path:
            raise ValueError("未设置参考音频")

        prompt_text = prompt_text if prompt_text is not None else self._prompt_text

        # 使用配置文件默认值
        if self.config:
            cfg = self.config.tts
            text_lang = text_lang or cfg.text_language
            prompt_lang = prompt_lang or cfg.prompt_language
            media_type = media_type or cfg.media_type
            top_k = top_k or cfg.top_k
            top_p = top_p or cfg.top_p
            temperature = temperature or cfg.temperature
            text_split_method = text_split_method or cfg.text_split_method
            batch_size = batch_size or cfg.batch_size
            batch_threshold = batch_threshold or cfg.batch_threshold
            speed_factor = speed_factor or cfg.speed_factor
            repetition_penalty = repetition_penalty or cfg.repetition_penalty
            sample_steps = sample_steps or cfg.sample_steps
            super_sampling = super_sampling if super_sampling is not None else cfg.super_sampling
        else:
            # 使用默认值
            text_lang = text_lang or "zh"
            prompt_lang = prompt_lang or "zh"
            media_type = media_type or "wav"
            top_k = top_k or 5
            top_p = top_p or 1.0
            temperature = temperature or 1.0
            text_split_method = text_split_method or "cut5"
            batch_size = batch_size or 1
            batch_threshold = batch_threshold or 0.75
            speed_factor = speed_factor or 1.0
            repetition_penalty = repetition_penalty or 1.35
            sample_steps = sample_steps or 32
            super_sampling = super_sampling or False

        params = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "batch_threshold": batch_threshold,
            "speed_factor": speed_factor,
            "streaming_mode": str(streaming_mode) if streaming_mode is not None else None,
            "media_type": media_type,
            "repetition_penalty": repetition_penalty,
            "sample_steps": sample_steps,
            "super_sampling": str(super_sampling) if super_sampling is not None else None,
        }
        params = {k: v for k, v in params.items() if v is not None}
        return params

    async def tts(
        self,
        text: str,
        ref_audio_path: str = None,
        aux_ref_audio_paths: List[str] = None,
        text_lang: str = None,
        prompt_text: str = None,
        prompt_lang: str = None,
        top_k: int = None,
        top_p: float = None,
        temperature: float = None,
        text_split_method: str = None,
        batch_size: int = None,
        batch_threshold: float = None,
        speed_factor: float = None,
        media_type: str = None,
        repetition_penalty: float = None,
        sample_steps: int = None,
        super_sampling: bool = None,
        **kwargs,
    ):
        """文本转语音

        Args:
            text: 要合成的文本
            ref_audio_path: 参考音频路径，如果为None则使用上次设置的参考音频
            aux_ref_audio_paths: 辅助参考音频路径列表(用于多说话人音色融合)
            prompt_text: 提示文本，如果为None则使用上次设置的提示文本
            text_lang: 文本语言,默认使用配置文件中的设置
            prompt_lang: 提示文本语言,默认使用配置文件中的设置
            top_k: top k采样
            top_p: top p采样
            temperature: 温度系数
            text_split_method: 文本分割方法
            batch_size: 批处理大小
            batch_threshold: 批处理阈值
            speed_factor: 语速控制
            streaming_mode: 是否启用流式输出
            media_type: 音频格式(wav/raw/ogg/aac)
            repetition_penalty: 重复惩罚系数
            sample_steps: VITS采样步数
            super_sampling: 是否启用超采样
        Returns:
            content (bytes): 合成的wav音频数据
        """
        platform = kwargs.get("platform")
        if not platform:
            print("未指定平台,使用默认平台")
            platform = "default"
        preset_name = self.get_platform_preset(platform)
        if self._current_preset != preset_name:
            self.load_preset(preset_name)
        params = self.build_parameters(
            text=text,
            ref_audio_path=ref_audio_path,
            aux_ref_audio_paths=aux_ref_audio_paths,
            text_lang=text_lang,
            prompt_text=prompt_text,
            prompt_lang=prompt_lang,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            text_split_method=text_split_method,
            batch_size=batch_size,
            batch_threshold=batch_threshold,
            speed_factor=speed_factor,
            streaming_mode=False,  # 强制使用非流式模式
            media_type=media_type,
            repetition_penalty=repetition_penalty,
            sample_steps=sample_steps,
            super_sampling=super_sampling,
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/tts", params=params, timeout=60) as response:  # noqa
                response.raise_for_status()
                return await response.read()

    async def tts_stream(
        self,
        text,
        ref_audio_path=None,
        aux_ref_audio_paths=None,
        text_lang=None,
        prompt_text=None,
        prompt_lang=None,
        top_k=None,
        top_p=None,
        temperature=None,
        text_split_method=None,
        batch_size=None,
        batch_threshold=None,
        speed_factor=None,
        media_type=None,
        repetition_penalty=None,
        sample_steps=None,
        super_sampling=None,
        **kwargs,
    ):
        """流式文本转语音,返回音频数据流

        Args:
            与tts()方法相同,但streaming_mode强制为True
        Returns:
            response (byte): 流式的wav格式音频数据
        """
        platform = kwargs.get("platform")
        if not platform:
            print("未指定平台,使用默认平台")
            platform = "default"
        preset_name = self.get_platform_preset(platform)
        if self._current_preset != preset_name:
            self.load_preset(preset_name)
        params = self.build_parameters(
            text=text,
            ref_audio_path=ref_audio_path,
            aux_ref_audio_paths=aux_ref_audio_paths,
            text_lang=text_lang,
            prompt_text=prompt_text,
            prompt_lang=prompt_lang,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            text_split_method=text_split_method,
            batch_size=batch_size,
            batch_threshold=batch_threshold,
            speed_factor=speed_factor,
            streaming_mode=True,  # 强制使用流式模式
            media_type=media_type,
            repetition_penalty=repetition_penalty,
            sample_steps=sample_steps,
            super_sampling=super_sampling,
        )

        # async with aiohttp.ClientSession() as session:
        #     async with session.get(f"{self.base_url}/tts", params=params, timeout=aiohttp.ClientTimeout(connect=3.05, sock_read=None)) as response:
        #         if response.status != 200:
        #             raise Exception(await response.json().get("message", "未知错误"))

        #         # 使用更小的块大小来提高流式传输的响应性
        #         async for chunk in response.content.iter_any(4096):
        #             yield chunk
        # 使用自定义超时，并设置较小的块大小来保持流式传输的响应性
        response = requests.get(
            f"{self.base_url}/tts",
            params=params,
            stream=True,
            timeout=(3.05, None),  # (连接超时, 读取超时)
            headers={"Connection": "keep-alive"},
        )

        if response.status_code != 200:
            raise Exception(response.json()["message"])

        # 使用更小的块大小来提高流式传输的响应性
        return response.iter_content(chunk_size=4096)
