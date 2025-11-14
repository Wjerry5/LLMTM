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
                "Connection": "close",  # Avoid keep-alive
                "Keep-Alive": "timeout=0"  # Disable keep-alive
            }
class OpenAIAPI:
    """
    OpenAI API Wrapper Class
    
    Provides a simple and easy-to-use interface to call OpenAI models
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
        Initializes the OpenAI API client
        
        Args:
            base_url (str): API base URL
            default_headers (dict): Default request headers
            debug (bool): Whether to enable debug mode
            debug_dir (str): Directory to save debug logs
        """
        openai.api_key = key
        openai.base_url = base_url
        openai.default_headers = default_headers
        
        self.debug = debug
        self.debug_dir = debug_dir
        
        if self.debug:
            self._setup_debug_logging()
    
    def _setup_debug_logging(self):
        """Sets up debug logging"""
        # Ensure the Debug folder exists
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        # Create logger
        self.logger = logging.getLogger('OpenAIAPI')
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid adding duplicate handlers
        if not self.logger.handlers:
            # Create file handler
            debug_filename = os.path.join(
                self.debug_dir, 
                f"openai_api_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(debug_filename, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info(f"🔧 OpenAI API Debug mode enabled, logging to: {debug_filename}")
    
    def _log_request(self, model: str, content: str, **kwargs):
        """Logs API request information"""
        if self.debug:
            self.logger.debug("=" * 50)
            self.logger.debug("📤 API Request Info:")
            self.logger.debug(f"Model: {model}")
            self.logger.debug(f"Content length: {len(content)} chars")
            self.logger.debug(f"Content preview: {content[:200]}...")
            self.logger.debug(f"Temperature: {kwargs.get('temperature', 'N/A')}")
            self.logger.debug(f"Max tokens: {kwargs.get('max_tokens', 'N/A')}")
            self.logger.debug("=" * 50)
    
    def _log_response(self, response: Dict[str, Any], duration: float):
        """Logs API response information"""
        if self.debug:
            self.logger.debug("📥 API Response Info:")
            self.logger.debug(f"Duration: {duration:.2f}s")
            self.logger.debug(f"Model: {response.get('model', 'N/A')}")
            self.logger.debug(f"ID: {response.get('id', 'N/A')}")
            self.logger.debug(f"Finish reason: {response.get('finish_reason', 'N/A')}")
            
            if 'usage' in response:
                usage = response['usage']
                self.logger.debug(f"Token Usage:")
                self.logger.debug(f"  - Prompt tokens: {usage.get('prompt_tokens', 0)}")
                self.logger.debug(f"  - Completion tokens: {usage.get('completion_tokens', 0)}")
                self.logger.debug(f"  - Total tokens: {usage.get('total_tokens', 0)}")
            
            content = response.get('content', '')
            self.logger.debug(f"Response length: {len(content)} chars")
            self.logger.debug(f"Response preview: {content[:200]}...")
            self.logger.debug("=" * 50)
    
    def _log_error(self, error: Exception, model: str, content_length: int):
        """Logs API error information"""
        if self.debug:
            self.logger.error("❌ API Call Failed:")
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
        Calls the chat completion API
        
        Args:
            content (str): User input content
            model (str): Model name (required)
            role (str): Message role, default is user
            temperature (float): Temperature parameter, 0.0 is most deterministic, 1.0 is most creative
            max_tokens (int): Maximum number of tokens
            
        Returns:
            str: The AI's response content
            
        Raises:
            Exception: Raises exception if API call fails
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
            
            # Build response info for logging
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
            raise Exception(f"API call failed: {str(e)}")
    
    
    def get_response(self, model: str, prompt: str, api_log_path: str, max_tokens: int = 20480) -> Dict[str, Any]:
        """
        Gets response and token usage information, and saves detailed logs
        
        Args:
            model (str): Model name
            prompt (str): User input content
            api_log_path (str): API call log save path
            
        Returns:
            dict: Contains response content and token usage information
        """
        start_time = time.time()
        prompt += "Do not output the thought process; provide the answer directly.\n"
        # Ensure log directory exists
        os.makedirs(os.path.dirname(api_log_path), exist_ok=True)
        
        # Prepare log content
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
            # API will automatically return token count, no local calculation needed
            
            # Create a normal response instead of a streaming response
            completion = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens
            )
            
            # Get content directly from the response
            complete_response = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            response_id = completion.id
            
            # Use token count returned by API
            
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
            
            # Update log content
            log_entry.update({
                "response": response,
                "duration": f"{duration:.2f}s",
                "status": "success"
            })
            
        except Exception as e:
            error_msg = str(e)
            if self.debug:
                self._log_error(e, model, len(prompt))
            
            # Log error information to log
            log_entry.update({
                "status": "error",
                "error": error_msg,
                "duration": f"{time.time() - start_time:.2f}s"
            })
            
            raise Exception(f"API call failed: {error_msg}")
        
        finally:
            # Write log to file
            try:
                with open(api_log_path, 'a', encoding='utf-8') as f:
                    json.dump(log_entry, f, ensure_ascii=False, indent=2)
                    f.write('\n')  # Add a newline after each log entry
            except Exception as e:
                print(f"Warning: Failed to write API log to {api_log_path}: {str(e)}")
        
        return response

# Example usage
# if __name__ == "__main__":
#     try:
#         # Enable debug mode
#         api = OpenAIAPI(debug=True)
        
#         # Method 1: Get only text content
#         response1 = api.get_response(model="deepseek-r1-250528", prompt="Hello world!")
#         print(response1)
#         print(response1["content"])

        
#     except Exception as e:
#         print(f"Error: {e}")