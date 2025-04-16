import requests
import os
from utils.config import Config
from typing import Optional, Dict, Any


class TTSModel:
    def __init__(self, config: Config = None, host="127.0.0.1", port=9880):
        """初始化TTS模型
        
        Args:
            config: 配置对象,如果为None则使用默认配置
            host: API服务器地址
            port: API服务器端口
        """
        self.config = config
        if config:
            self.host = config.tts.host
            self.port = config.tts.port
        else:
            self.host = host
            self.port = port
            
        self.base_url = f"http://{self.host}:{self.port}"
        self._ref_audio_path = None  # 存储当前使用的参考音频路径
        self._prompt_text = ""       # 存储当前使用的提示文本
        self._current_preset = "default"  # 当前使用的角色预设名称
        self._initialized = False    # 标记是否已完成初始化

    def _ensure_api_available(self, timeout=5):
        """确保API服务可用
        
        Args:
            timeout: 连接超时时间（秒）
            
        Returns:
            bool: API是否可用
        """
        try:
            requests.get(f"{self.base_url}/ping", timeout=timeout)
            return True
        except requests.exceptions.RequestException:
            return False

    def initialize(self):
        """初始化模型和预设
        
        如果已经初始化过，则跳过
        """
        if self._initialized:
            return
        self._initialized = True
            
        if not self._ensure_api_available():
            raise ConnectionError("无法连接到GPT-SoVITS API服务")
        
        # 初始化默认模型
        if self.config:
            if self.config.tts.models.gpt_model:
                self.set_gpt_weights(self.config.tts.models.gpt_model)
            if self.config.tts.models.sovits_model:
                self.set_sovits_weights(self.config.tts.models.sovits_model)
        
        # 设置默认角色预设
        if self.config:
            self.load_preset("default")
            

    @property 
    def ref_audio_path(self):
        """获取当前使用的参考音频路径"""
        return self._ref_audio_path

    @property
    def prompt_text(self):
        """获取当前使用的提示文本"""
        return self._prompt_text
    
    @property
    def current_preset(self):
        """获取当前使用的角色预设名称"""
        return self._current_preset

    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
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

    def load_preset(self, preset_name: str):
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

    def set_refer_audio(self, audio_path: str, prompt_text: str):
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
        
        if not os.path.exists(audio_path):
            raise ValueError(f"音频文件不存在: {audio_path}")
            
        self._ref_audio_path = audio_path
        self._prompt_text = prompt_text

    def set_gpt_weights(self, weights_path):
        """设置GPT权重"""
        if not os.path.exists(weights_path):
            raise ValueError(f"GPT模型文件不存在: {weights_path}")
            
        response = requests.get(
            f"{self.base_url}/set_gpt_weights", params={"weights_path": weights_path}
        )
        if response.status_code != 200:
            raise Exception(response.json()["message"])

    def set_sovits_weights(self, weights_path):
        """设置SoVITS权重"""
        if not os.path.exists(weights_path):
            raise ValueError(f"SoVITS模型文件不存在: {weights_path}")
            
        response = requests.get(
            f"{self.base_url}/set_sovits_weights", params={"weights_path": weights_path}
        )
        if response.status_code != 200:
            raise Exception(response.json()["message"])

    def tts(
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
        streaming_mode=None,
        media_type=None,
        repetition_penalty=None,
        sample_steps=None,
        super_sampling=None,
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
        """
        if not self._initialized:
            self.initialize()
        
        # 优先使用传入的ref_audio_path和prompt_text,否则使用持久化的值
        ref_audio_path = ref_audio_path or self._ref_audio_path
        if not ref_audio_path:
            raise ValueError("未设置参考音频，请先调用set_refer_audio设置参考音频和提示文本")
            
        prompt_text = prompt_text if prompt_text is not None else self._prompt_text
        
        # 使用配置文件中的默认值
        if self.config:
            cfg = self.config.tts
            text_lang = text_lang or cfg.text_language
            prompt_lang = prompt_lang or cfg.prompt_language
            media_type = media_type or cfg.media_type 
            streaming_mode = streaming_mode if streaming_mode is not None else cfg.streaming_mode
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
            streaming_mode = streaming_mode or False
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
            "streaming_mode": streaming_mode,
            "media_type": media_type,
            "repetition_penalty": repetition_penalty,  
            "sample_steps": sample_steps,
            "super_sampling": super_sampling
        }

        response = requests.get(f"{self.base_url}/tts", params=params)
        if response.status_code != 200:
            raise Exception(response.json()["message"])
        return response.content


def test_tts_model():
    """测试TTS模型的基本功能和性能"""
    from utils.config import get_default_config
    import wave
    import os
    import time
    import requests.exceptions
    import statistics
    
    print("开始TTS模型测试...\n")
    start_time = time.time()
    
    # 1. 测试API连接
    print("1. 测试API连接...")
    config = get_default_config()
    tts = TTSModel(config)
    max_retries = 3
    retry_delay = 2  # 秒
    
    # 2. 测试预设管理
    print("\n2. 测试预设管理...")
    try:
        # 测试获取预设列表
        presets = tts.config.tts.models.presets
        print(f"发现 {len(presets)} 个预设:")
        for name, preset in presets.items():
            print(f"  - {name}: {preset.get('name', '未命名')}")
        
        # 测试加载默认预设
        tts.load_preset("default")
        print(f"✓ 成功加载默认预设")
        print(f"  - 参考音频: {tts.ref_audio_path}")
        print(f"  - 提示文本: {tts.prompt_text}")
        
        # 尝试加载其他预设（如果存在）
        other_presets = [p for p in presets.keys() if p != "default"]
        if other_presets:
            test_preset = other_presets[0]
            try:
                tts.load_preset(test_preset)
                print(f"✓ 成功加载预设 '{test_preset}'")
            except Exception as e:
                print(f"✗ 加载预设 '{test_preset}' 失败: {str(e)}")
    except Exception as e:
        print(f"✗ 预设管理测试失败: {str(e)}")
    
    # 3. 测试文本转语音
    print("\n3. 测试文本转语音和性能...")
    test_texts = [
        "这是一个简单的测试。",
        "让我们来测试一个稍长的句子，看看效果如何？",
        "这是一个包含数字和标点的测试：2025年4月16日，温度是25.6℃！",
        "这是一个很长的句子用来测试分段处理的效果，它包含了很多内容",
        "现在我们测试一下中英混合的情况： 阳光明媚，天气很好！",
    ]
    
    performance_stats = {
        'processing_times': [],
        'audio_durations': [],
        'chars_per_second': []
    }
    
    print("\n性能测试结果:")
    print("-" * 50)
    print("| 测试文本 | 处理时间 | 音频时长 | 字符处理速度 |")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试文本 {i}: {text}")
        try:
            # 计时开始
            text_start_time = time.time()
            audio_data = tts.tts(text)
            processing_time = time.time() - text_start_time
            
            if audio_data:
                # 保存测试音频
                output_path = f"test_output_{i}.wav"
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                
                # 验证生成的音频文件
                with wave.open(output_path, 'rb') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    chars_per_sec = len(text) / duration
                    
                    performance_stats['processing_times'].append(processing_time)
                    performance_stats['audio_durations'].append(duration)
                    performance_stats['chars_per_second'].append(chars_per_sec)
                    
                    print(f"✓ 成功生成音频文件:")
                    print(f"  - 采样率: {wav_file.getframerate()} Hz")
                    print(f"  - 时长: {duration:.2f} 秒")
                    print(f"  - 字符处理速度: {chars_per_sec:.2f} 字/秒")
                    print(f"  - 处理耗时: {processing_time:.2f} 秒")
                    print(f"  - 输出文件: {output_path}")
                
                print(f"| {text[:20]}... | {processing_time:.2f}秒 | {duration:.2f}秒 | {chars_per_sec:.2f}字/秒 |")
                
        except Exception as e:
            import traceback
            print(f"✗ 生成音频失败: {str(e)}")
            print(traceback.format_exc())
            continue
    
    # 4. 输出整体性能统计
    print("\n4. 性能统计总结")
    print("-" * 50)
    if performance_stats['processing_times']:
        avg_processing_time = statistics.mean(performance_stats['processing_times'])
        avg_duration = statistics.mean(performance_stats['audio_durations'])
        avg_chars_per_sec = statistics.mean(performance_stats['chars_per_second'])
        
        print(f"平均处理时间: {avg_processing_time:.2f} 秒")
        print(f"平均音频时长: {avg_duration:.2f} 秒")
        print(f"平均字符处理速度: {avg_chars_per_sec:.2f} 字/秒")
        
        if len(performance_stats['processing_times']) > 1:
            std_processing_time = statistics.stdev(performance_stats['processing_times'])
            print(f"处理时间标准差: {std_processing_time:.2f} 秒")
    
    total_time = time.time() - start_time
    print(f"\n总测试时间: {total_time:.2f} 秒")
    print("\n测试完成!")

if __name__ == "__main__":
    test_tts_model()
