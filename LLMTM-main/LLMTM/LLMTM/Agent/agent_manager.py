import os
import logging
import re
import json
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Not needed for cloud models

import langchain
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent

from .tools import ALL_TOOLS
import openai

langchain.debug = True
import datetime
# --- Set your OpenAI API Key ---
openai.base_url = ""
openai.default_headers = {"x-foo": "true"}

# Set LangChain debug output to file
def setup_langchain_debug_logging():
    import sys
    import atexit
    
    if debug_dir is None:
        raise ValueError("debug_dir must be set before calling setup_langchain_debug_logging")
    
    # Ensure Debug folder exists
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Create a custom output stream that writes to both file and console
    class TeeOutput:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
            # Register to close the file on exit
            atexit.register(self.close)
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            if hasattr(self, 'log') and not self.log.closed:
                self.log.close()
    
    # Redirect standard output and standard error, save to Debug folder
    debug_filename = os.path.join(debug_dir, f"langchain_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create Tee output
    tee_output = TeeOutput(debug_filename)
    
    # Redirect stdout and stderr
    sys.stdout = tee_output
    sys.stderr = tee_output
    
    print(f"LangChain debug output will be saved to: {debug_filename}")
    print(f"Original stdout and stderr have been redirected")
    
    return debug_filename

# First define debug_dir, then call the setting function
debug_dir = None  # Initialize to None, set later in get_response
debug_file = None  # Initialize to None, set later in get_response

def extract_question(task, user_query):
    multi_motif_dyg = f"""Possible temporal motifs in the dynamic graph include:
"triangle": a 3-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u0, t2, a)]
"3-star": a 4-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u0, u2, t1, a), (u0, u3, t2, a)]
"4-path": a 4-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a)]
"4-cycle": a 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u0, t3, a)]
"4-tailedtriangle": 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u1, t3, a)]
"butterfly": a 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u3, t1, a), (u3, u2, t2, a), (u2, u0, t3, a)]
"4-chordalcycle": a 4-node, 5-edge, 14-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u1, t3, a), (u3, u0, t4, a)]
"4-clique": a 4-node, 6-edge, 15-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u1, u3, t2, a), (u2, u3, t3, a), (u3, u0, t4, a), (u0, u2, t5, a)]
"bitriangle": 6-node, 6-edge, 15-temporal motif with the edges[(u0, u1, t0, a), (u1, u3, t1, a), (u3, u5, t2, a), (u5, u4, t3, a), (u4, u2, t4, a), (u2, u0, t5, a)]
"""
    Question = ""
    def question_pattern(user_query):
        pattern = r'Question:\s*(.*?)(?=\s*Answer:|$)'
        matches = re.findall(pattern, user_query, re.DOTALL | re.IGNORECASE)
        return matches[-1].strip()
    if (task == "judge_multi_motif" or task == "multi_motif_first_time" or task == "multi_motif_count"):
        Question = multi_motif_dyg + question_pattern(user_query)
    else:
        Question = question_pattern(user_query)
    # print(Question)
    return Question

class AgentManager:
    def __init__(
        self,
        key = "",
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_new_tokens=8192,
        memory_k=5,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True,
        disable_memory=False
    ):
        self.key = key
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.memory_k = memory_k
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.handle_parsing_errors = handle_parsing_errors
        self.disable_memory = disable_memory
        self.log_handler = None
        self.original_stdout = None
        self.original_stderr = None
        self.tee_output = None

        self.llm = self._load_llm()
        self.memory = self._setup_memory()
        self.agent_executor = self._init_agent_executor()

    def _load_llm(self):

        llm = ChatOpenAI(
            api_key=self.key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            base_url=openai.base_url,         # Use the globally set value, but pass it as a parameter
            default_headers=openai.default_headers, # Use the globally set value, but pass it as a parameter
            stop=['\nObservation']
        )
        return llm

    def _setup_memory(self):
        if self.disable_memory:
            print("Memory functionality is explicitly disabled. Each query will be independent.")
            return None
        else:
            print(f"Memory functionality enabled with window size k={self.memory_k}.")
            return ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=self.memory_k,
                return_messages=True
            )

    def _create_agent_prompt(self):
        template = """You are a tool-using assistant for dynamic graph analysis.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]. The action name must be shown.
Action Input: the input to the action. The complete input parameters must be shown (Even if it has shown before, copy it here exactly). 
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES FOR TOOL USAGE:
- Answer ONLY what the user asked - NOTHING MORE.
- Use EXACTLY ONE tool - NEVER call multiple tools.
- Action Input MUST be a dictionary that maps the parameters of the tool to the data in the question.
- The 'edge_list' parameter MUST be a list of 4-element arrays: (u, v, t, operation), u, v and t are integers, operation is a string like "a" or "d".
- You MUST copy the edge list from the user's question EXACTLY as it appears. Do not add, remove, or modify any edges. 
- For the `motif_list` parameter MUST be a dictionary that maps motif name to a nested object containing its edge_pattern and time_window.
    - Motif_list Parameter Format: {{
        "motif_name": {{
        "edge_pattern": [(u0, v0, t0, operation), (u1, v1, t1, operation), ...],
        "time_window": time_window
        }}
    }}
    - time_window is the number extracted from "X-temporal motif" in the question.
    - edge_pattern is a list of tuples, each tuple is a 4-element array: (u0, v0, t0, operation), u0, v0, t0 are strings with u, u, t as the initials, extracted from "u0", "u1", "t0" in the question, operation is a string like "a" or "d".
- For the `motif_definitions` parameter MUST be is a dictionary that maps each motif name to a nested object containing its edge_pattern and time_window.
    - Motif_definitions Parameter Format: {{
        "motif_name1": {{
        "edge_pattern": [(u0, v0, t0, operation), (u1, v1, t1, operation), ...],
        "time_window": time_window
        }}
        "motif_name2": {{
        "edge_pattern": [(u0, v0, t0, operation), (u1, v1, t1, operation), ...],
        "time_window": time_window
        }}
        ...
    }}
    - time_window is the number extracted from "X-temporal motif" in the question.
    - edge_pattern is a list of tuples, each tuple is a 4-element array: (u0, v0, t0, operation), directly copied from the question.
- After you receive an Observation, IMMEDIATELY output Final Answer.
- The Final Answer MUST be the EXACT content from the last Observation, without any modification or conversational filler.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        return prompt

    def _init_agent_executor(self):
        agent = create_react_agent(
            llm=self.llm,
            tools=ALL_TOOLS,
            prompt=self._create_agent_prompt(),
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=self.verbose,
            memory=self.memory,
            max_iterations=self.max_iterations,
            handle_parsing_errors=self.handle_parsing_errors,
            return_intermediate_steps=True,  # Return intermediate steps for debugging
            early_stopping_method="force",  # Use a supported stopping method
        )
        return agent_executor

    def clear_memory(self):
        if self.memory is not None:
            self.memory.clear()
            print("Conversation history cleared")
        else:
            print("Memory functionality is disabled, no need to clear.")

    def _setup_logging(self, log_dir):
        """Set up independent logging for each task"""
        import sys
        import json
        import re
        
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create new log files
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f"langchain_debug_{timestamp}.log")
        stats_filename = os.path.join(log_dir, f"agent_stats_{timestamp}.json")
        
        # Save original stdout and stderr
        if self.original_stdout is None:
            self.original_stdout = sys.stdout
        if self.original_stderr is None:
            self.original_stderr = sys.stderr
            
        # Create new TeeOutput instance
        class TeeOutput:
            def __init__(self, filename, stats_filename, original_stdout):
                self.log = open(filename, 'w', encoding='utf-8')
                self.stats_file = stats_filename
                self.original_stdout = original_stdout
                self.stats = {
                    "execution_time": 0,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "steps": []
                }
                self.buffer = ""
            
            def write(self, message):
                # Directly write to original output and log without any processing
                self.original_stdout.write(message)
                self.log.write(message)
                self.log.flush()
                
                # Only accumulate [llm/end] messages in the buffer for subsequent processing
                if '[llm/end]' in message:
                    self.buffer += message
                    if message.strip().endswith('}'):  # Ensure JSON is complete
                        self._process_buffer()
                        self.buffer = ""
            
            def _process_buffer(self):
                try:
                    # Use regex to extract the JSON part
                    json_match = re.search(r'\{.*\}', self.buffer, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        if 'generations' in data and isinstance(data['generations'], list):
                            for generation in data['generations']:
                                if isinstance(generation, list):
                                    for msg in generation:
                                        if isinstance(msg, dict) and 'message' in msg:
                                            message_data = msg['message']
                                            if isinstance(message_data, dict) and 'kwargs' in message_data:
                                                kwargs = message_data['kwargs']
                                                if 'usage_metadata' in kwargs:
                                                    metadata = kwargs['usage_metadata']
                                                    if all(k in metadata for k in ['input_tokens', 'output_tokens', 'total_tokens']):
                                                        # Update statistics
                                                        self.stats["prompt_tokens"] += int(metadata['input_tokens'])
                                                        self.stats["completion_tokens"] += int(metadata['output_tokens'])
                                                        self.stats["total_tokens"] += int(metadata['total_tokens'])
                                                        self._save_stats()
                except Exception:
                    pass  # Silently handle statistics-related errors without affecting main functionality
            
            def flush(self):
                self.original_stdout.flush()
                self.log.flush()
            
            def close(self):
                if hasattr(self, 'log') and not self.log.closed:
                    self.log.close()
                self._save_stats()
            
            def _save_stats(self):
                try:
                    with open(self.stats_file, 'w', encoding='utf-8') as f:
                        json.dump(self.stats, f, indent=2)
                except Exception:
                    pass  # Silently handle statistics file write errors
            
            def update_stats(self, time=None, step_info=None):
                if time:
                    self.stats["execution_time"] = time
                if step_info:
                    self.stats["steps"].append(step_info)
                self._save_stats()
        
        # Close previous log files (if they exist)
        if self.tee_output is not None:
            self.tee_output.close()
        
        # Create a new TeeOutput instance
        self.tee_output = TeeOutput(log_filename, stats_filename, self.original_stdout)
        
        # Redirect output
        sys.stdout = self.tee_output
        sys.stderr = self.tee_output
        
        print(f"LangChain debug output will be saved to: {log_filename}")
        print(f"Agent statistics will be saved to: {stats_filename}")
        
        return log_filename

    def __del__(self):
        """Clean up resources"""
        import sys
        # Restore original stdout and stderr
        if hasattr(self, 'original_stdout') and self.original_stdout is not None:
            sys.stdout = self.original_stdout
        if hasattr(self, 'original_stderr') and self.original_stderr is not None:
            sys.stderr = self.original_stderr
        # Close log file
        if hasattr(self, 'tee_output') and self.tee_output is not None:
            self.tee_output.close()

    def get_response(self, task, prompt: str, agent_log_path: str, clear_history: bool = True):
        """
        Gets a response from the Agent with no retry mechanism.
        By default, clears historical memory before each new query.
        """
        start_time = time.time()
        
        # Set new log file
        log_file = self._setup_logging(agent_log_path)
        
        prompt = extract_question(task, prompt)
        if clear_history and not self.disable_memory:
            self.clear_memory()
        # print(prompt)
        # print(f"\n--- Processing query (single attempt) ---")
        try:
            inputs = {
                "input": prompt,
             }
            
            result = self.agent_executor.invoke(inputs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update statistics
            step_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "execution_time": execution_time,
            }
            if hasattr(self.tee_output, "update_stats"):
                self.tee_output.update_stats(
                    time=execution_time,
                    step_info=step_info
                )

            response_content = result.get("output", "").strip()

            if response_content:
                print(f"Query completed successfully.")
                print(f"⏱Execution time: {execution_time:.2f}s")
                print(f"Tokens used: {self.tee_output.stats if self.tee_output else 'N/A'}")
                return {
                    "content": response_content,
                    "success": True,
                    "info": "Single attempt completed.",
                    "time": execution_time,
                    "tokens": self.tee_output.stats if self.tee_output else {}
                }
            else:
                print(f"Query completed, but no content returned from Agent.")
                return {
                    "content": "Agent returned empty output.",
                    "success": False,
                    "info": "Single attempt completed.",
                    "time": execution_time
                }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Agent invocation failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {
                "content": f"Agent processing error: {str(e)}",
                "success": False,
                "info": "Error during single attempt.",
                "time": execution_time
            }
