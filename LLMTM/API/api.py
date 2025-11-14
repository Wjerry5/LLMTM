import os
import openai
import logging
import datetime
from typing import Optional, Dict, Any, List

openai.base_url = ""
openai.default_headers = {"x-foo": "true"}

class OpenAIAPI:
    """
    OpenAI API Wrapper Class
    
    Provides a simple interface to call OpenAI models.
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
        Initialize OpenAI API client
        
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
        """Set up debug logging"""
        # Ensure Debug folder exists
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
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info(f"🔧 OpenAI API Debug mode enabled, logs saved to: {debug_filename}")
    
    def _log_request(self, model: str, content: str, **kwargs):
        """Log API request information"""
        if self.debug:
            self.logger.debug("=" * 50)
            self.logger.debug("API Request Information:")
            self.logger.debug(f"Model: {model}")
            self.logger.debug(f"Content length: {len(content)} chars")
            self.logger.debug(f"Content preview: {content[:200]}...")
            self.logger.debug(f"Temperature: {kwargs.get('temperature', 'N/A')}")
            self.logger.debug(f"Max tokens: {kwargs.get('max_tokens', 'N/A')}")
            self.logger.debug("=" * 50)
    
    def _log_response(self, response: Dict[str, Any], duration: float):
        """Log API response information"""
        if self.debug:
            self.logger.debug("API Response Information:")
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
        """Log API error information"""
        if self.debug:
            self.logger.error("API call failed:")
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
        Call chat completion API
        
        Args:
            content (str): User input content
            model (str): Model name (required)
            role (str): Message role, default is user
            temperature (float): Temperature parameter, 0.0 is most deterministic, 1.0 is most creative
            max_tokens (int): Maximum number of tokens
            
        Returns:
            str: AI response content
            
        Raises:
            Exception: Throws an exception when API call fails
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
                max_tokens=max_tokens
            )
            
            duration = time.time() - start_time
            
            # Build response information for logging
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
        Get response and token usage information, and save detailed logs
        
        Args:
            model (str): Model name
            prompt (str): User input content
            api_log_path (str): Path to save API call logs
            
        Returns:
            dict: Contains response content and token usage information
        """
        import time
        import json
        from datetime import datetime
        import tiktoken  # Used for local token calculation
        
        start_time = time.time()
        
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
            # Get encoding, use cl100k_base for custom models
            try:
                encoding = tiktoken.encoding_for_model(model)
            except Exception:
                # For unsupported models, use cl100k_base as the default encoder
                encoding = tiktoken.get_encoding("cl100k_base")
            
            # Calculate prompt tokens
            prompt_tokens = len(encoding.encode(prompt))
            
            # Create streaming response
            stream = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Collect full response
            full_content = []
            finish_reason = None
            response_id = None
            
            # Process streaming response
            try:
                for chunk in stream:
                    if self.debug:
                        self.logger.debug(f"Received chunk: {chunk}")
                    
                    # Get response_id
                    if hasattr(chunk, 'id'):
                        response_id = chunk.id
                    
                    # Safely get choices
                    choices = getattr(chunk, 'choices', [])
                    if not choices:
                        continue
                    
                    # Safely get the first choice
                    choice = choices[0]
                    if not choice:
                        continue
                    
                    # Process content
                    delta = getattr(choice, 'delta', None)
                    if delta:
                        content = getattr(delta, 'content', None)
                        if content is not None:
                            full_content.append(content)
                    
                    # Update finish_reason
                    if hasattr(choice, 'finish_reason') and choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
                        
            except Exception as e:
                if self.debug:
                    self.logger.error(f"Streaming processing error: {str(e)}")
                    self.logger.error(f"Error details: {e.__class__.__name__}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                raise
            
            # Merge full content
            complete_response = ''.join(full_content)
            
            # Calculate completion tokens
            completion_tokens = len(encoding.encode(complete_response))
            total_tokens = prompt_tokens + completion_tokens
            
            duration = time.time() - start_time
            
            response = {
                "content": complete_response,
                "role": "assistant",
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "token_encoding": "cl100k_base" if model not in tiktoken.model.MODEL_TO_ENCODING else encoding.name
                },
                "model": model,
                "id": response_id
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
            
            # Log error message
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
                print(f"Warning: Unable to write API log to {api_log_path}: {str(e)}")
        
        return response
