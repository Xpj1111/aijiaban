# -*- coding: utf-8 -*-
"""
模型管理器 - 轻量级LLM推理引擎
支持llama-cpp-python进行GGUF量化模型推理和Transformers加载
"""

import os
import logging
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger("ZhiJiaBan.Inference")

# ==================== 依赖检查 ====================
def _check_dependencies(model_type: str = "transformers"):
    """检查必要的依赖库是否已安装"""
    if model_type == "gguf":
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python 未安装！\n"
                "请运行: pip install llama-cpp-python\n"
                "Windows用户建议使用预编译版本: pip install llama-cpp-python --only-binary :all:"
            )
    elif model_type == "transformers":
        try:
            import transformers
            import torch
        except ImportError:
            raise ImportError(
                "transformers 或 torch 未安装！\n"
                "请运行: pip install transformers torch accelerate"
            )


class ModelManager:
    """
    轻量级模型管理器
    
    职责：
    - 加载GGUF量化模型（INT4）
    - 提供统一的聊天接口
    - 管理模型生命周期
    
    使用的库：llama-cpp-python
    模型格式：GGUF (Qwen2.5-1.5B-Instruct-Q4_K_M)
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # 模型实例
        self._llm = None
        self._tokenizer = None
        self._model = None
        
        # 线程锁（保证线程安全）
        self._lock = threading.Lock()
        
        # 状态标志
        self.is_initialized = False
        self._model_loaded = False
        
        # 配置参数（带默认值）
        self.model_type = getattr(config, 'model_type', 'transformers')  # 'gguf' 或 'transformers'
        
        # 依赖检查
        _check_dependencies(self.model_type)
        
        if self.model_type == 'transformers':
            # Transformers 模式
            self.local_path = getattr(config, 'local_path', './models/qwen')
            inf_config = getattr(config, 'inference', None)
            if inf_config:
                self.max_length = getattr(inf_config, 'max_length', 2048)
                self.temperature = getattr(inf_config, 'temperature', 0.7)
                self.top_p = getattr(inf_config, 'top_p', 0.9)
                self.max_new_tokens = getattr(inf_config, 'max_new_tokens', 512)
                self.device_map = getattr(inf_config, 'device_map', 'auto')
            else:
                self.max_length = 2048
                self.temperature = 0.7
                self.top_p = 0.9
                self.max_new_tokens = 512
                self.device_map = 'auto'
        else:
            # GGUF 模式（原有逻辑）
            self.model_path = getattr(config, 'gguf_path', './models/qwen2.5-1.5b-instruct-q4_k_m.gguf')
            self.download_url = getattr(config, 'download_url', '')
            
            inf_config = getattr(config, 'inference', None)
            if inf_config:
                self.n_ctx = getattr(inf_config, 'n_ctx', 2048)
                self.n_threads = getattr(inf_config, 'n_threads', 4)
                self.n_gpu_layers = getattr(inf_config, 'n_gpu_layers', 0)
                self.temperature = getattr(inf_config, 'temperature', 0.7)
                self.top_p = getattr(inf_config, 'top_p', 0.9)
                self.max_tokens = getattr(inf_config, 'max_tokens', 512)
            else:
                self.n_ctx = 2048
                self.n_threads = 4
                self.n_gpu_layers = 0
                self.temperature = 0.7
                self.top_p = 0.9
                self.max_tokens = 512
    
    def initialize(self) -> bool:
        """
        初始化并加载模型
        
        Returns:
            是否加载成功
        """
        try:
            if self.model_type == 'transformers':
                return self._initialize_transformers()
            else:
                return self._initialize_gguf()
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            self._model_loaded = False
            self.is_initialized = False
            raise
    
    def _initialize_transformers(self) -> bool:
        """使用 Transformers 加载模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_path = Path(self.local_path)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {self.local_path}")
            
            logger.info(f"📦 正在加载 Transformers 模型: {self.local_path}")
            
            # 加载 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            # 加载模型
            self._model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map=self.device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            self._model.eval()
            self._model_loaded = True
            self.is_initialized = True
            
            logger.info("✅ Transformers 模型加载成功!")
            
            # 预热
            self._warmup_transformers()
            
            return True
            
        except ImportError:
            raise ImportError(
                "transformers 未安装！请运行: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"❌ Transformers 模型加载失败: {e}")
            raise
    
    def _initialize_gguf(self) -> bool:
        """使用 llama-cpp-python 加载 GGUF 模型（原有逻辑）"""
        try:
            # 检查模型文件是否存在
            model_file = Path(self.model_path)
            
            if not model_file.exists():
                logger.error(f"❌ 模型文件不存在: {self.model_path}")
                logger.info(f"💡 请从以下地址下载模型文件:")
                logger.info(f"   {self.download_url or 'https://huggingface.co/Qwen'}")
                
                # 尝试在models目录查找
                alt_paths = [
                    "./models/*.gguf",
                    "../models/*.gguf",
                    "./*.gguf"
                ]
                for pattern in alt_paths:
                    import glob
                    matches = glob.glob(pattern)
                    if matches:
                        self.model_path = matches[0]
                        logger.info(f"✅ 找到替代模型文件: {self.model_path}")
                        break
                else:
                    raise FileNotFoundError(
                        f"未找到GGUF模型文件。请下载后放置于 {self.model_path}"
                    )
            
            # 动态导入llama_cpp（避免未安装时报错）
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python 未安装！请运行: pip install llama-cpp-python"
                )
            
            logger.info(f"📦 正在加载模型: {self.model_path}")
            logger.info(f"   上下文窗口: {self.n_ctx}, 线程数: {self.n_threads}, GPU层: {self.n_gpu_layers}")
            
            # 创建Llama实例
            self._llm = Llama(
                model_path=str(Path(self.model_path).resolve()),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False  # 减少日志输出
            )
            
            self._model_loaded = True
            self.is_initialized = True
            
            logger.info("✅ 模型加载成功!")
            
            # 预热：进行一次空推理，消除首次延迟
            self._warmup_gguf()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            self._model_loaded = False
            self.is_initialized = False
            raise
    
    def _warmup_transformers(self):
        """Transformers 模型预热"""
        if not self._model_loaded or self._model is None:
            return
        
        try:
            import time
            import torch
            
            start = time.time()
            inputs = self._tokenizer("你好", return_tensors="pt")
            if hasattr(self._model, 'device'):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self._model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=self.temperature,
                    do_sample=True
                )
            
            elapsed = time.time() - start
            logger.info(f"🔥 Transformers 模型预热完成 ({elapsed:.2f}s)")
        except Exception as e:
            logger.warning(f"⚠️ Transformers 模型预热失败: {e}")
    
    def _warmup_gguf(self):
        """GGUF 模型预热 - 执行一次空推理"""
        if not self._model_loaded or self._llm is None:
            return
        
        try:
            import time
            start = time.time()
            self._llm("你好", max_tokens=10, stop=["\n"])
            elapsed = time.time() - start
            logger.info(f"🔥 GGUF 模型预热完成 ({elapsed:.2f}s)")
        except Exception as e:
            logger.warning(f"⚠️ GGUF 模型预热失败: {e}")
    
    def chat(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        聊天接口 - 发送Prompt并获取回复
        
        Args:
            prompt: 用户输入或完整prompt
            system_prompt: 可选的系统提示词
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大生成token数
            **kwargs: 其他llama_cpp参数
            
        Returns:
            模型生成的文本回复
        """
        if not self._model_loaded:
            raise RuntimeError("模型尚未初始化，请先调用 initialize()")
        
        if self.model_type == 'transformers':
            return self._chat_transformers(prompt, system_prompt, temperature, max_tokens, **kwargs)
        else:
            return self._chat_gguf(prompt, system_prompt, temperature, max_tokens, **kwargs)
    
    def _chat_transformers(self, prompt: str, system_prompt: str = None, 
                          temperature: float = None, max_tokens: int = None, **kwargs) -> str:
        """使用 Transformers 进行对话"""
        import torch
        
        with self._lock:  # 线程安全
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 应用聊天模板
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self._tokenizer(text, return_tensors="pt")
            
            # 输入长度校验
            input_length = inputs['input_ids'].shape[1]
            max_context = getattr(self, 'max_length', 2048)
            if input_length >= max_context:
                logger.warning(f"⚠️ 输入长度 ({input_length}) 接近模型上限 ({max_context})，可能影响生成质量")
            
            if hasattr(self._model, 'device'):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # 设置参数
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_new_tokens
            
            # 生成
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=tokens,
                    temperature=temp,
                    top_p=self.top_p,
                    do_sample=True if temp > 0 else False,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # 解码输出（只取新生成的部分）
            response = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
    
    def _chat_gguf(self, prompt: str, system_prompt: str = None,
                   temperature: float = None, max_tokens: int = None, **kwargs) -> str:
        """使用 llama-cpp-python 进行对话"""
        with self._lock:  # 线程安全
            # 构建消息列表
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # 设置参数
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # 调用模型
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=temp,
                top_p=self.top_p,
                max_tokens=tokens,
                **kwargs
            )
            
            # 提取回复文本
            result_text = ""
            if response and 'choices' in response and len(response['choices']) > 0:
                result_text = response['choices'][0]['message']['content']
            
            return result_text
    
    def generate(self, prompt: str, **kwargs) -> str:
        """简单的文本生成接口（非chat模式）"""
        if not self._model_loaded:
            raise RuntimeError("模型尚未初始化")
        
        if self.model_type == 'transformers':
            return self._generate_transformers(prompt, **kwargs)
        else:
            return self._generate_gguf(prompt, **kwargs)
    
    def _generate_transformers(self, prompt: str, **kwargs) -> str:
        """使用 Transformers 进行文本生成"""
        import torch
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, 'device'):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', self.max_new_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                top_p=kwargs.get('top_p', self.top_p),
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除 prompt 部分
        response = response[len(prompt):]
        return response.strip()
    
    def _generate_gguf(self, prompt: str, **kwargs) -> str:
        """使用 llama-cpp-python 进行文本生成（原有逻辑）"""
        response = self._llm(
            prompt,
            temperature=kwargs.get('temperature', self.temperature),
            top_p=kwargs.get('top_p', self.top_p),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            stop=kwargs.get('stop', ["\n\n", "</s>", "[/INST]"]),
            echo=False
        )
        
        if response and 'choices' in response:
            return response['choices'][0]['text']
        return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        if self.model_type == 'transformers':
            return {
                "model_type": "transformers",
                "model_path": self.local_path,
                "is_loaded": self._model_loaded,
                "max_length": self.max_length,
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "device_map": self.device_map
            }
        else:
            return {
                "model_type": "gguf",
                "model_path": self.model_path,
                "is_loaded": self._model_loaded,
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "has_gpu": self.n_gpu_layers > 0
            }
    
    def unload(self):
        """卸载模型释放内存"""
        if self.model_type == 'transformers':
            if self._model is not None:
                del self._model
                self._model = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            
            # 显式释放 GPU 显存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("🗑️ GPU 显存已释放")
            except Exception as e:
                logger.warning(f"⚠️ 显存释放失败: {e}")
        else:
            if self._llm is not None:
                del self._llm
                self._llm = None
        
        self._model_loaded = False
        self.is_initialized = False
        logger.info("🗑️ 模型已卸载")
    
    def __del__(self):
        """析构时自动卸载"""
        try:
            self.unload()
        except Exception:
            pass


# ==================== 降级模式：模拟推理 ====================

class MockModelManager(ModelManager):
    """
    模拟模型管理器（用于无模型时的演示）
    
    返回预定义的模板回复，无需实际模型
    """
    
    MOCK_RESPONSES = {
        "default": "我理解您的需求了，让我来帮您处理。",
        "greeting": "您好！我是智家伴，很高兴为您服务～🏠",
        "control_device": "好的，已为您控制设备。",
        "health": "已记录您的健康数据。请注意保持健康的生活习惯哦！",
        "reminder": "提醒设置完成！我会准时提醒您的⏰",
        "weather": "今天天气不错，适合外出活动呢☀️",
        "story": "从前有一只勇敢的小狮子...🦁",
        "error": "抱歉，我暂时无法处理这个请求。"
    }
    
    # 中文关键词映射
    KEYWORD_MAP = {
        "你好": "greeting", "您好": "greeting", "hello": "greeting",
        "开灯": "control_device", "关灯": "control_device", "空调": "control_device",
        "血压": "health", "血糖": "health", "心率": "health",
        "提醒": "reminder", "闹钟": "reminder",
        "天气": "weather", "气温": "weather",
        "故事": "story", "讲故事": "story",
    }
    
    def initialize(self) -> bool:
        """模拟初始化（始终成功）"""
        self.is_initialized = True
        self._model_loaded = True
        logger.info("🎭 使用模拟模式（Mock Model Manager）")
        return True
    
    def chat(self, prompt: str, **kwargs) -> str:
        """返回基于关键词匹配的模板回复"""
        lower_prompt = (prompt + "").lower()
        
        # 遍历中文关键词映射
        for keyword, response_key in self.KEYWORD_MAP.items():
            if keyword in prompt or keyword.lower() in lower_prompt:
                return self.MOCK_RESPONSES.get(response_key, self.MOCK_RESPONSES["default"])
        
        return self.MOCK_RESPONSES["default"]
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_path": "mock/simulation-mode",
            "is_loaded": True,
            "mode": "simulation"
        }
