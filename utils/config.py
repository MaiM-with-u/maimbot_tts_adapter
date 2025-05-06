from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import toml
from pathlib import Path


@dataclass
class TTSPreset:
    name: str
    ref_audio: str
    prompt_text: str
    gpt_model: str = ""
    sovits_model: str = ""


@dataclass
class TTSModels:
    gpt_model: str
    sovits_model: str
    presets: Dict[str, TTSPreset]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TTSModels":
        presets = {
            name: TTSPreset(**preset_data)
            for name, preset_data in data.get("presets", {}).items()
        }
        return cls(
            gpt_model=data.get("gpt_model", ""),
            sovits_model=data.get("sovits_model", ""),
            presets=presets,
        )


@dataclass
class OmniTTSConfig:
    """大模型TTS配置"""
    enabled: bool
    api_key: str
    model_name: str
    voice: str
    format: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OmniTTSConfig":
        return cls(
            enabled=data.get("enabled", False),
            api_key=data.get("api_key", ""),
            model_name=data.get("model_name", "qwen-omni-turbo"),
            voice=data.get("voice", "Chelsie"),
            format=data.get("format", "wav"),
        )


@dataclass
class TTSConfig:
    # api_url: str
    host: str
    port: int
    ref_audio_path: str
    prompt_text: str
    aux_ref_audio_paths: List[str]
    text_language: str
    prompt_language: str
    media_type: str
    streaming_mode: bool
    top_k: int
    top_p: float
    temperature: float
    batch_size: int
    batch_threshold: float
    speed_factor: float
    text_split_method: str
    repetition_penalty: float
    sample_steps: int
    super_sampling: bool
    models: TTSModels

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TTSConfig":
        models_data = data.pop("models", {})
        return cls(
            **{k: v for k, v in data.items() if k != "models"},
            models=TTSModels.from_dict(models_data),
        )


@dataclass
class ServerConfig:
    host: str
    port: int


@dataclass
class PipelineConfig:
    default_preset: str
    platform_presets: Dict[str, str]
    keep_original_tts: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            default_preset=data.get("default_preset", "default"),
            platform_presets=data.get("platform_presets", {}),
            keep_original_tts=data.get("keep_original_tts", False),
        )


@dataclass
class ProbabilityConfig:
    voice_probability: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbabilityConfig":
        return cls(
            voice_probability=data.get("voice_probability"),
        )


@dataclass
class BaseConfig:
    tts: TTSConfig
    server: ServerConfig
    routes: Dict[str, str]
    pipeline: PipelineConfig
    probability: ProbabilityConfig
    omni_tts: Optional[OmniTTSConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        omni_tts_data = data.get("omni_tts")
        return cls(
            tts=TTSConfig.from_dict(data["tts"]),
            server=ServerConfig(**data["server"]),
            routes=data["routes"],
            pipeline=PipelineConfig.from_dict(data.get("pipeline", {})),
            probability=ProbabilityConfig.from_dict(data.get("probability", {})),
            omni_tts=OmniTTSConfig.from_dict(omni_tts_data) if omni_tts_data else None,
        )


class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = load_config(config_path)
        self.base_config = BaseConfig.from_dict(self.config_data)

    def __getitem__(self, key: str) -> Any:
        return self.config_data[key]

    def __setitem__(self, key: str, value: Any):
        self.config_data[key] = value

    def __repr__(self) -> str:
        return str(self.config_data)

    @property
    def tts(self) -> TTSConfig:
        return self.base_config.tts

    @property
    def server(self) -> ServerConfig:
        return self.base_config.server

    @property
    def routes(self) -> Dict[str, str]:
        return self.base_config.routes

    @property
    def pipeline(self) -> PipelineConfig:
        return self.base_config.pipeline
    
    @property
    def probability(self) -> ProbabilityConfig:
        return self.base_config.probability
        
    @property
    def omni_tts(self) -> Optional[OmniTTSConfig]:
        return self.base_config.omni_tts


def load_config(config_path: str) -> Dict[str, Any]:
    """加载TOML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
    return config


def get_default_config() -> Config:
    """获取默认配置"""
    config_path = Path(__file__).parent.parent / "configs" / "base.toml"
    return Config(str(config_path))
