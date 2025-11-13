import os
import openai
import logging
import datetime
from typing import Optional, Dict, Any, List
import json
import time
from datetime import datetime
# os.environ["OPENAI_API_KEY"] = "sk-Pth1hQVprCzicKWmB16396007fC44c568a74F3C8Fb484979"
# os.environ["OPENAI_API_KEY"] = "sk-d1X7CZ7Vs3HwELwY9101927aF49e495bA32a94C8B0337210"

openai.base_url = "http://127.0.0.1:8888/v1/"
# openai.default_headers = {"x-foo": "true"}
openai.default_headers={
                "Connection": "close",  # 避免保持连接
                "Keep-Alive": "timeout=0"  # 禁用keep-alive
            }
class OpenAIAPI:
    """
    OpenAI API封装类
    
    提供简单易用的接口来调用OpenAI模型
    """
    
    def __init__(
        self,
        key: str,
        base_url: str = openai.base_url,
        default_headers: Dict[str, str] = openai.default_headers,
        debug: bool = False,
        debug_dir: str = "Debug"
    ):
        """
        初始化OpenAI API客户端
        
        Args:
            base_url (str): API基础URL
            default_headers (dict): 默认请求头
            debug (bool): 是否启用debug模式
            debug_dir (str): debug日志保存目录
        """
        openai.api_key = key
        openai.base_url = base_url
        openai.default_headers = default_headers
        
        self.debug = debug
        self.debug_dir = debug_dir
        
        if self.debug:
            self._setup_debug_logging()
    
    def _setup_debug_logging(self):
        """设置debug日志"""
        # 确保Debug文件夹存在
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        # 创建logger
        self.logger = logging.getLogger('OpenAIAPI')
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建文件handler
            debug_filename = os.path.join(
                self.debug_dir, 
                f"openai_api_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(debug_filename, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 创建控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加handler到logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info(f"🔧 OpenAI API Debug模式已启用，日志保存到: {debug_filename}")
    
    def _log_request(self, model: str, content: str, **kwargs):
        """记录API请求信息"""
        if self.debug:
            self.logger.debug("=" * 50)
            self.logger.debug("📤 API请求信息:")
            self.logger.debug(f"Model: {model}")
            self.logger.debug(f"Content length: {len(content)} chars")
            self.logger.debug(f"Content preview: {content[:200]}...")
            self.logger.debug(f"Temperature: {kwargs.get('temperature', 'N/A')}")
            self.logger.debug(f"Max tokens: {kwargs.get('max_tokens', 'N/A')}")
            self.logger.debug("=" * 50)
    
    def _log_response(self, response: Dict[str, Any], duration: float):
        """记录API响应信息"""
        if self.debug:
            self.logger.debug("📥 API响应信息:")
            self.logger.debug(f"耗时: {duration:.2f}秒")
            self.logger.debug(f"Model: {response.get('model', 'N/A')}")
            self.logger.debug(f"ID: {response.get('id', 'N/A')}")
            self.logger.debug(f"Finish reason: {response.get('finish_reason', 'N/A')}")
            
            if 'usage' in response:
                usage = response['usage']
                self.logger.debug(f"Token使用情况:")
                self.logger.debug(f"  - Prompt tokens: {usage.get('prompt_tokens', 0)}")
                self.logger.debug(f"  - Completion tokens: {usage.get('completion_tokens', 0)}")
                self.logger.debug(f"  - Total tokens: {usage.get('total_tokens', 0)}")
            
            content = response.get('content', '')
            self.logger.debug(f"Response length: {len(content)} chars")
            self.logger.debug(f"Response preview: {content[:200]}...")
            self.logger.debug("=" * 50)
    
    def _log_error(self, error: Exception, model: str, content_length: int):
        """记录API错误信息"""
        if self.debug:
            self.logger.error("❌ API调用失败:")
            self.logger.error(f"Model: {model}")
            self.logger.error(f"Content length: {content_length}")
            self.logger.error(f"Error: {str(error)}")
            self.logger.error("=" * 50)
    
    def chat_completion(
        self, 
        content: str, 
        model: str,
        role: str = "user",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 8192,
    ) -> str:
        """
        调用聊天完成API
        
        Args:
            content (str): 用户输入内容
            model (str): 模型名称（必填）
            role (str): 消息角色，默认为user
            temperature (float): 温度参数，0.0最确定，1.0最有创意
            max_tokens (int): 最大token数量
            
        Returns:
            str: AI的回复内容
            
        Raises:
            Exception: API调用失败时抛出异常
        """
        import time
        start_time = time.time()
        
        self._log_request(model, content, temperature=temperature, max_tokens=max_tokens)
        
        try:
            completion = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": role, "content": content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                skip_special_tokens=True
            )
            
            duration = time.time() - start_time
            
            # 构建响应信息用于日志记录
            response_info = {
                "content": completion.choices[0].message.content,
                "model": completion.model,
                "id": completion.id,
                "finish_reason": completion.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }
            }
            
            self._log_response(response_info, duration)
            
            return completion.choices[0].message
        except Exception as e:
            self._log_error(e, model, len(content))
            raise Exception(f"API调用失败: {str(e)}")
    
    
    def get_response(self, model: str, prompt: str, api_log_path: str, max_tokens: int = 20480) -> Dict[str, Any]:
        """
        获取回复和tokens使用信息，并保存详细日志
        
        Args:
            model (str): 模型名称
            prompt (str): 用户输入内容
            api_log_path (str): API调用日志保存路径
            
        Returns:
            dict: 包含回复内容和tokens使用信息
        """
        start_time = time.time()
        # prompt += "Do not output the thought process; provide the answer directly.\n"
        # 确保日志目录存在
        os.makedirs(os.path.dirname(api_log_path), exist_ok=True)
        
        # 准备日志内容
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "prompt": prompt,
            "request": {
                "temperature": 0.0,
                "max_tokens": max_tokens
            }
        }
        
        try:
            # API会自动返回token数量，不需要本地计算
            
            # 创建普通响应而不是流式响应
            completion = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens
            )
            
            # 直接从响应中获取内容
            complete_response = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            response_id = completion.id
            
            # 使用API返回的token数量
            
            duration = time.time() - start_time
            
            response = {
                "content": complete_response,
                "role": "assistant",
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                },
                "model": model,
                "id": response_id,
                "duration": duration
            }
            
            # 更新日志内容
            log_entry.update({
                "response": response,
                "duration": f"{duration:.2f}秒",
                "status": "success"
            })
            
        except Exception as e:
            error_msg = str(e)
            if self.debug:
                self._log_error(e, model, len(prompt))
            
            # 记录错误信息到日志
            log_entry.update({
                "status": "error",
                "error": error_msg,
                "duration": f"{time.time() - start_time:.2f}秒"
            })
            
            raise Exception(f"API调用失败: {error_msg}")
        
        finally:
            # 将日志写入文件
            try:
                with open(api_log_path, 'a', encoding='utf-8') as f:
                    json.dump(log_entry, f, ensure_ascii=False, indent=2)
                    f.write('\n')  # 每个日志条目后添加换行
            except Exception as e:
                print(f"警告：无法写入API日志到{api_log_path}: {str(e)}")
        
        return response

# 使用示例
# if __name__ == "__main__":
#     try:
#         # 启用debug模式
#         api = OpenAIAPI(debug=True)
        
#         # 方法1：只获取文本内容
#         response1 = api.get_response(model="deepseek-r1-250528", prompt="Hello world!")
#         print(response1)
#         print(response1["content"])

        
#     except Exception as e:
#         print(f"错误: {e}")
        
        