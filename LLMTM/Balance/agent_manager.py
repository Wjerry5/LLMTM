#!/usr/bin/env python3
import os
import logging
import re
import json
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Not needed for cloud models
import langchain
# from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
# from langchain_community.memory import ConversationBufferWindowMemory 
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent

# try:
#     from langchain_community.callbacks import get_openai_callback
# except ImportError:
#     from langchain.callbacks import get_openai_callback # Compatible with older versions

from .tools import ALL_TOOLS
import openai
import sys  
import atexit
import datetime
import tiktoken
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, List, Dict, Optional


langchain.debug = True


# --- Set your OpenAI API Key ---
os.environ["OPENAI_API_KEY"] = "123"
openai.base_url = "http://127.0.0.1:8888/v1/"
# openai.default_headers = {"x-foo": "true"}
openai.default_headers={
                "Connection": "close",  # Avoid keep-alive
                "Keep-Alive": "timeout=0"  # Disable keep-alive
            }

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
            # Register closing the file on exit
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
    print(f"📝 LangChain debug output will be saved to: {debug_filename}")
    print(f"📝 Original stdout and stderr have been redirected")

    return debug_filename

# Define debug_dir first, then call the setup function

debug_dir = None  # Initialize to None, will be set later in get_response
debug_file = None  # Initialize to None, will be set later in get_response


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

        # Question += "Do not output the thought process; provide the answer directly.\n"

    # print(Question)
    return Question

# --- MODIFICATION 2: Add Tiktoken Callback Handler ---
# This class is used to calculate tokens on the *client-side*
class TiktokenCallbackHandler(BaseCallbackHandler):
    """
    A custom callback handler that estimates Tokens on the client-side using tiktoken.
    This captures *all* LLM calls during Agent execution.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            # Estimated encoding. "cl100k_base" is suitable for gpt-4, gpt-3.5-turbo
            self.encoding = tiktoken.get_encoding("cl100k_base")
            print(f"Token estimator loaded: cl100k_base")
        except Exception:
            self.encoding = tiktoken.get_encoding("p50k_base")
            print(f"Token estimator loaded (fallback): p50k_base")
        
        # These are accumulators for *this* call
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Calculates Prompt Tokens when LLM call starts."""
        for prompt in prompts:
            try:
                self.prompt_tokens += len(self.encoding.encode(prompt))
            except Exception:
                self.prompt_tokens += len(prompt.split()) 

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Calculates Completion Tokens when LLM call ends."""
        for generation_list in response.generations:
            for generation in generation_list:
                try:
                    self.completion_tokens += len(self.encoding.encode(generation.text))
                except Exception:
                    self.completion_tokens += len(generation.text.split())

    @property
    def stats(self) -> Dict[str, int]:
        """Get total statistics."""
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        # Return a dictionary with a structure *identical* to your TeeOutput.stats
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
# -----------------------------------------------------

class AgentManager:
    def __init__(
        self,
        model_name="pangu_auto",
        temperature=0.1,
        max_new_tokens=8192,
        memory_k=5,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True,
        disable_memory=False

    ):
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
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set your OpenAI API key.")

        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            base_url=openai.base_url,       # Use the globally set value, but pass via parameter
            default_headers=openai.default_headers, # Use the globally set value, but pass via parameter
            stop=['\nObservation']
        )
        return llm


    def _setup_memory(self):
        if self.disable_memory:
            print("🚫 Memory functionality is explicitly disabled. Each query will be independent.")
            return None
        else:
            print(f"✅ Memory functionality enabled with window size k={self.memory_k}.")
            return ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=self.memory_k,
                return_messages=True
            )

    def _create_agent_prompt(self):
        template = """You are a tool-using assistant for graph complexity analysis. Your job is to extract edge list from dynamic graph questions and predict whether the question can be answered directly by a Language Model (LLM) or requires an Agent with tools.

You have access to this tool:
{tools}

USE THE FOLLOWING FORMAT (Key Focus: After Question, You **MUST** output Action and Action Input; After Observation, You **MUST** output Final Answer):
Question: the input question I must answer
Thought: [I MUST output Action and Action Input.]
Action: the action to take, must be one of [{tool_names}]. The action name must be shown.
Action Input: {{"edge_list": [COPY edges from question]}}
Observation: the result of the tool.
Thought: [I MUST output the EXACT content from the last Observation as the Final Answer.]
Final Answer: [COPY EXACT content from the last Observation.]

EDGE EXTRACTION RULES:
- The 'edge_list' parameter MUST be a strict LIST of 4-element arrays: (u, v, t, operation), u, v and t are integers, NOT string, operation is a string like "a" or "d", directly COPIED from the question.
- Only include dynamic graph edges, Not temporal motif edges 

CRITICAL RULES FOR FINAL ANSWER:
- The Final Answer MUST be the EXACT content from the last Observation.
- DO NOT rephrase, explain, or convert the last Observation into a full sentence.
- If the last Observation is LLM, the Final Answer MUST be LLM.
- If the last Observation is Agent, the Final Answer MUST be Agent.
- Your task here is to be a data-processor, NOT a conversational assistant.

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
            return_intermediate_steps=True,  # Return intermediate steps, convenient for debugging
            early_stopping_method="force",  # Use a supported stopping method
        )

        return agent_executor


    def clear_memory(self):
        if self.memory is not None:
            self.memory.clear()
            print("🧹 Conversation history cleared")
        else:
            print("🚫 Memory functionality is disabled, no need to clear.")



    def _setup_logging(self, log_dir):
        """Set up independent logging for each task"""
        import json
        import re

        # Ensure log directory exists
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

        class TeeOutput:
            def __init__(self, filename, stats_filename, original_stdout):
                self.log = open(filename, 'w', encoding='utf-8')
                self.stats_file = stats_filename
                self.original_stdout = original_stdout
                # Your stats dictionary, containing your desired token names
                self.stats = {
                    "execution_time": 0,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "steps": []
                }
                self.buffer = "" # Your buffer logic   


            
            def write(self, message):
                # Write directly to original output and log, without any processing
                self.original_stdout.write(message)
                self.log.write(message)
                self.log.flush()
                # Only accumulate [llm/end] messages in the buffer for subsequent processing
                if '[llm/end]' in message:
                    self.buffer += message
                    if message.strip().endswith('}'):  # Ensure JSON is complete
                        self._process_buffer()
                        self.buffer = ""

            # def write(self, message):
            #     # Write directly to original output and log, without any processing
            #     self.original_stdout.write(message)
            #     self.log.write(message)
            #     self.log.flush()
            #     # self.buffer and _process_buffer are no longer needed

    

            def _process_buffer(self):
                try:
                    # Use regex to extract JSON part
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
                    pass  # Silently handle statistics-related errors, do not affect main functionality

            

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
                    pass  

            

            def update_stats(self, time=None, step_info=None, token_stats: Dict[str, int] = None):
                if time:
                    self.stats["execution_time"] = time
                if step_info:
                    self.stats["steps"].append(step_info)
                if token_stats:
                    self.stats.update(token_stats)
                self._save_stats()

    

        # Close previous log file (if it exists)

        if self.tee_output is not None:
            self.tee_output.close()

        # Create a new TeeOutput instance
        self.tee_output = TeeOutput(log_filename, stats_filename, self.original_stdout)

        # Redirect output
        sys.stdout = self.tee_output
        sys.stderr = self.tee_output

        print(f"📝 LangChain debug output will be saved to: {log_filename}")
        print(f"📊 Agent statistics will be saved to: {stats_filename}")

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
        Gets a response from the Agent, using tiktoken for stats.
        """
        start_time = time.time()
        log_file = self._setup_logging(agent_log_path) # TeeOutput is created here
        prompt = extract_question(task, prompt)
        if clear_history and not self.disable_memory:
            self.clear_memory()

        token_stats = {} # Initialize

        try:
            inputs = {
                "input": prompt,
            } 

            # --- 1. Create Tiktoken callback instance ---
            token_callback = TiktokenCallbackHandler()
            config = {"callbacks": [token_callback]}

            # --- 2. Run Agent, passing in config ---
            result = self.agent_executor.invoke(inputs, config=config)
            
            # --- 3. Get statistics *for this call* from the callback ---
            # (This will return a dictionary: {"total_tokens": ..., "prompt_tokens": ..., "completion_tokens": ...})
            token_stats = token_callback.stats

            execution_time = time.time() - start_time
            
            # --- 4. Update TeeOutput.stats ---
            step_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "execution_time": execution_time,
                "tokens": token_stats # (Optional) Also record in "steps"
            }
            if hasattr(self.tee_output, "update_stats"):
                # Pass the statistics results from tiktoken (token_stats)
                # update_stats will merge it into self.tee_output.stats
                self.tee_output.update_stats(
                    time=execution_time,
                    step_info=step_info,
                    token_stats=token_stats # ** Inject token statistics **
                )

            response_content = result.get("output", "").strip()
            if response_content:
                print(f"✅ Query completed successfully.")
                print(f"⏱️ Execution time: {execution_time:.2f}s")
                # Now self.tee_output.stats contains the estimates from tiktoken
                print(f"🎟️ Tokens used (tiktoken est.): {self.tee_output.stats}") 
                return {
                    "content": response_content,
                    "success": True,
                    "info": "Single attempt completed.",
                    "time": execution_time,
                    # The returned "tokens" dictionary now has the correct keys and values
                    "tokens": self.tee_output.stats if self.tee_output else {}
                }
            else:
                # ( ... failure logic unchanged ... )
                print(f"❌ Query completed, but no content returned from Agent.")
                return {
                    "content": "Agent returned empty output.",
                    "success": False,
                    "info": "Single attempt completed.",
                    "time": execution_time,
                    "tokens": self.tee_output.stats if self.tee_output else {} # Return even on failure
                }
        except Exception as e:
            # ( ... exception logic unchanged ... )
            execution_time = time.time() - start_time
            print(f"❌ Agent invocation failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {
                "content": f"Agent processing error: {str(e)}",
                "success": False,
                "info": "Error during single attempt.",
                "time": execution_time,
                "tokens": self.tee_output.stats if self.tee_output else {} # Return even on exception
            }

# import os
# # from agent_manager import AgentManager

if __name__ == "__main__":
    # Initialize AgentManager without max_retries parameter
    agent_manager = AgentManager(
        model_name="pangu_auto",
        temperature=0.1,
        max_new_tokens=2048,
        memory_k=5,
        verbose=True,
        max_iterations=4,  # Reduce max iterations, force Agent to finish quickly
        handle_parsing_errors=True,
        disable_memory=False
    )

    print("\n--- Dynamic Graph Agent Started (Pangu - Single Attempt) ---")
    print("Agent will clear historical memory for each new question.")
    print("Type 'exit' to quit.")


    message = """In an undirected dynamic graph, (u, v, t, a) means that node u and node v are linked with an undirected edge at time t, (u, v, t, d) means that node u and node v are deleted with an undirected edge at time t.\nA k-nosde, l-edge, \u03b4-temporal motif is a time-ordered sequence of k nodes and l distinct edges within \u03b4 duration, i.e., M = (u1, v1, t1, a), (u2, v2, t2, a) ..., (ul, vl, tl, a), this edges (u1, v1), (u2, v2) ..., (ul, vl) form a specific pattern, t1 < t2 < ... < tl are strictly increasing, and tl - t1 \u2264 \u03b4. That is each edge that forms a specific pattern occurs in specific order within a fixed time window. Each consecutive events shares at least one common node. When searching for a specific temporal motif in the undirected dynamic graph, it is necessary to match pattern, edge order and time window. Node IDs and exact timestamps are irrelevant. Meanwhile,you should only focus on added edges. Note that some patterns are symmetric, so the order of the corresponding timestamps may be unimportant.\nYour task is to determine whether the given undirected dynamic graph contains the given temporal motif? This means that there exists a subgraph within the undirected dynamic graph that matches both the structure and the temporal constraints of the temporal motif.\nGive the answer as 'Yes' or 'No' at the end of your response after 'Answer:'\nHere is what you need to answer:\nQuestion: Given an undirected dynamic graph with the edges [(5, 19, 0, 'a'), (4, 30, 1, 'a'), (16, 37, 2, 'a'), (48, 54, 3, 'a'), (4, 8, 4, 'a'), (18, 51, 4, 'a'), (44, 56, 4, 'a'), (2, 59, 5, 'a'), (5, 26, 7, 'a'), (9, 13, 7, 'a'), (20, 21, 8, 'a'), (33, 42, 9, 'a'), (47, 57, 9, 'a'), (34, 55, 13, 'a'), (2, 41, 16, 'a'), (0, 7, 17, 'a'), (23, 58, 17, 'a'), (6, 20, 18, 'a'), (3, 44, 19, 'a'), (9, 26, 19, 'a'), (3, 42, 21, 'a'), (1, 21, 26, 'a'), (47, 54, 26, 'a'), (41, 44, 27, 'a'), (4, 39, 30, 'a')].Given a 3-star temporal motif which is a 4-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u0, u2, t1, a), (u0, u3, t2, a)]. Whether the given undirected dynamic graph contains the given temporal motif?\nAnswer:\nDo not output the thought process; provide the answer directly.\n"""
    response = agent_manager.get_response("judge_contain_motif", message, clear_history=True,agent_log_path="/home/ma-user/work")

    if response["success"]:
        print("\n--- Agent Response (Success) ---")
        print(response["content"])
    else:
        print("\n--- Agent Response (Failed) ---")
        print(f"Failed to get a valid response. Error Info: {response['content']}")
    print(f"Processing Info: {response['info']}")

    print(f"--- Stats for this run ---")
    print(json.dumps(response.get('tokens', {}), indent=2))

    print("\n--- Agent Exited ---")