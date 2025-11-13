#!/usr/bin/env python3

import os
import logging
import re
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Not needed for cloud models

import langchain
from langchain.agents import initialize_agent, AgentType # We'll move away from initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import BaseMessage # For type hinting, good practice

from tools import ALL_TOOLS # Assuming your tools.py file exists and defines ALL_TOOLS
import openai

langchain.debug = True
logging.basicConfig(level=logging.DEBUG)
# --- Set your OpenAI API Key ---
os.environ["OPENAI_API_KEY"] = "sk-Pth1hQVprCzicKWmB16396007fC44c568a74F3C8Fb484979"
openai.base_url = "https://api.vveai.com/v1/"
openai.default_headers = {"x-foo": "true"}

class AgentManager:
    def __init__(
        self,
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_new_tokens=2048,
        memory_k=5,
        verbose=True,
        max_iterations=7,
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
            base_url=openai.base_url,
            default_headers=openai.default_headers
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
        # This part of the system message is static and contains the core instructions.
        # {tools} and {tool_names} are placeholders that will be filled dynamically.
        system_instructions = """You are a tool-using assistant for dynamic graph analysis.

You have access to the following tools:
{tools}

STRICT FORMAT - You MUST follow this EXACTLY:
Question: The user's input question.
Thought: You should think about which tool to use as Action and what is the Action Input.
Action: [REQUIRED] Tool_name (must be one of: {tool_names})
Action Input: [REQUIRED] Input Parameters 符合工具要求的 (JSON format)

STOP HERE! Do NOT output any other text after the Action Input.

Observation: The output from the tool.
Final Answer: The definitive answer to the user's question, derived directly from the Observation.

CRITICAL RULES FOR TOOL USAGE:
- Answer ONLY what the user asked - NOTHING MORE.
- Use EXACTLY ONE tool - NEVER call multiple tools.
- The 'edge_list' parameter MUST be a list of 4-element arrays: (u, v, t, operation), u, v and t are integers, operation is a string like "a" or "d".
- You MUST copy the edge list from the user's question EXACTLY as it appears. Do not add, remove, or modify any edges. 
- For the `motif_list` parameter MUST be a dictionary that maps motif name to a nested object containing its edge_pattern and time_window.
    - Motif_list参数格式: {{{{
        "motif_name1": {{{{
        "edge_pattern": [(u0, v0, t0, operation), (u1, v1, t1, operation), ...],
        "time_window": time_window
        }}}}
    }}}}
- For the `motif_definitions` parameter MUST be is a dictionary that maps each motif name to a nested object containing its edge_pattern and time_window.
    - Motif_definitions参数格式: {{{{
        "motif_name1": {{{{
        "edge_pattern": [(u0, v0, t0, operation), (u1, v1, t1, operation), ...],\n\
        "time_window": time_window
        }}}}
        "motif_name2": {{{{
        "edge_pattern": [(u0, v0, t0, operation), (u1, v1, t1, operation), ...],
        "time_window": time_window
        }}}}
        ...
    }}}}
- After you receive an Observation, IMMEDIATELY output Final Answer.
- The Final Answer MUST be the EXACT content from the last Observation, without any modification or conversational filler.

Begin!
"""
        # Create a ChatPromptTemplate using messages
        # SystemMessage sets the overall role and static instructions.
        # MessagesPlaceholder allows dynamic injection of chat history and agent scratchpad.
        # HumanMessage is for the current user input.
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_instructions),
                MessagesPlaceholder(variable_name="chat_history"), # For conversational memory
                HumanMessage(content="{input}"), # For the current user question
                MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's thoughts and tool interactions
            ]
        )
        return prompt

    def _init_agent_executor(self):
        # Prepare tools string for insertion into the prompt
        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in ALL_TOOLS])
        tool_names_str = ", ".join([tool.name for tool in ALL_TOOLS])
        print(f"Available Tools:\n{tools_str}")
        print(f"Tool Names: {tool_names_str}")

        # Get the base ChatPromptTemplate and partially fill it with static tool info
        base_prompt = self._create_agent_prompt()
        final_prompt = base_prompt.partial(tools=tools_str, tool_names=tool_names_str)

        print("\n--- Debugging: Final prompt content after partial filling (example render) ---")
        # Print an example of the rendered prompt to verify its structure and content
        print(final_prompt.format_messages(chat_history=[], input="test_input", agent_scratchpad=[]))
        print("---------------------------------------------------------------------------\n")

        # Create the React agent chain
        # This chain combines the prompt, the LLM, and the tool-calling logic.
        agent_chain = create_react_agent(
            llm=self.llm,
            tools=ALL_TOOLS,
            prompt=final_prompt,
        )

        # Initialize the AgentExecutor
        # The AgentExecutor wraps the agent_chain, manages memory, handles tool execution,
        # and controls the overall agent behavior (max_iterations, error handling).
        agent_executor = AgentExecutor(
            agent=agent_chain,
            tools=ALL_TOOLS,
            verbose=self.verbose,
            memory=self.memory, # AgentExecutor automatically uses this memory for 'chat_history'
            max_iterations=self.max_iterations,
            handle_parsing_errors=self.handle_parsing_errors,
        )
        return agent_executor

    def clear_memory(self):
        if self.memory is not None:
            self.memory.clear()
            print("🧹 对话历史已清空")
        else:
            print("🚫 内存功能已禁用，无需清空。")

    def get_response(self, prompt: str, clear_history: bool = True):
        """
        Gets a response from the Agent with no retry mechanism.
        By default, clears historical memory before each new query.
        """
        if clear_history and not self.disable_memory:
            self.clear_memory()

        print(f"\n--- Processing query (single attempt) ---")
        try:
            # For AgentExecutor, you typically only need to provide 'input'.
            # It will automatically handle 'chat_history' from its internal memory
            # and 'agent_scratchpad' as it runs the agent loop.
            inputs = {
                "input": prompt,
                "agent_scratchpad": []
            }
            
            print("\nDebugging: Inputs to agent_executor.invoke():")
            print(json.dumps(inputs, indent=2, default=str))

            result = self.agent_executor.invoke(inputs)

            response_content = result.get("output", "").strip()

            if response_content:
                print(f"✅ Query completed successfully.")
                return {
                    "content": response_content,
                    "success": True,
                    "info": "Single attempt completed."
                }
            else:
                print(f"❌ Query completed, but no content returned from Agent.")
                return {
                    "content": "Agent returned empty output.",
                    "success": False,
                    "info": "Single attempt completed."
                }

        except Exception as e:
            print(f"❌ Agent invocation failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {
                "content": f"Agent processing error: {str(e)}",
                "success": False,
                "info": "Error during single attempt."
            }
# main.py
import os
from agent_manager import AgentManager

if __name__ == "__main__":
    # Initialize AgentManager without max_retries parameter
    agent_manager = AgentManager(
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_new_tokens=2048,
        memory_k=5,
        verbose=True,
        max_iterations=7,
        handle_parsing_errors=True,
        disable_memory=False
    )

    print("\n--- Dynamic Graph Agent Started (GPT-4o Mini - Single Attempt) ---")
    print("Agent will clear historical memory for each new question.")
    print("Type 'exit' to quit.")

    
    message = """Possible  temporal motifs in the dynamic graph include:\n\"triangle\": a 3-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u0, t2, a)]\n\"3-star\": a 4-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u0, u2, t1, a), (u0, u3, t2, a)]\n\"4-path\": a 4-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a)]\n\"4-cycle\": a 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u0, t3, a)]\n\"4-tailedtriangle\": 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u1, t3, a)]\n\"butterfly\": a 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u3, t1, a), (u3, u2, t2, a), (u2, u0, t3, a)]\n\"4-chordalcycle\": a 4-node, 5-edge, 14-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u1, t3, a), (u3, u0, t4, a)]\n\"4-clique\": a 4-node, 6-edge, 15-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u1, u3, t2, a), (u2, u3, t3, a), (u3, u0, t4, a), (u0, u2, t5, a)]\n\"bitriangle\": 6-node, 6-edge, 15-temporal motif with the edges[(u0, u1, t0, a), (u1, u3, t1, a), (u3, u5, t4, a), (u5, u4, t2, a), (u4, u2, t3, a), (u2, u0, t5, a)]\nQuestion: Given an undirected dynamic graph with the edges[(1, 3, 0, a), (1, 7, 0, a), (8, 12, 0, a), (10, 16, 0, a), (1, 9, 1, a), (7, 19, 1, a), (10, 13, 1, a), (10, 18, 1, a), (2, 5, 2, a), (3, 10, 2, a), (6, 11, 2, a), (1, 7, 3, d), (10, 16, 3, d), (12, 19, 3, a), (2, 8, 4, a), (4, 6, 4, a), (7, 8, 4, a), (0, 1, 5, a), (1, 2, 5, a), (3, 7, 5, a), (1, 17, 6, a), (3, 7, 6, d), (8, 13, 6, a), (9, 18, 6, a), (1, 13, 7, a), (7, 9, 7, a), (7, 10, 7, a), (8, 9, 7, a), (9, 10, 7, a), (16, 18, 7, a), (0, 10, 8, a), (11, 12, 8, a), (11, 16, 8, a), (14, 19, 8, a), (0, 14, 9, a), (1, 18, 9, a), (7, 8, 9, d), (7, 16, 9, a), (8, 17, 9, a), (8, 19, 9, a), (9, 19, 9, a), (11, 16, 9, d), (14, 18, 9, a), (1, 18, 10, d), (3, 10, 10, d), (4, 18, 10, a), (9, 19, 10, d), (11, 12, 10, d), (13, 14, 10, a), (0, 1, 11, d), (0, 4, 11, a), (0, 17, 11, a), (1, 13, 11, d), (4, 7, 11, a), (8, 13, 11, d), (9, 10, 11, d), (0, 4, 12, d), (0, 9, 12, a), (1, 10, 12, a), (1, 15, 12, a), (2, 8, 12, d), (4, 7, 12, d), (4, 12, 12, a), (4, 18, 12, d), (5, 9, 12, a), (7, 16, 12, d), (8, 9, 12, d), (10, 17, 12, a), (13, 14, 12, d), (0, 17, 13, d), (1, 10, 13, d), (1, 14, 13, a), (1, 15, 13, d), (5, 12, 13, a), (7, 9, 13, d), (8, 16, 13, a), (9, 15, 13, a), (12, 13, 13, a), (0, 9, 14, d), (1, 14, 14, d), (3, 6, 14, a), (4, 12, 14, d), (6, 8, 14, a), (8, 10, 14, a), (8, 16, 14, d), (9, 18, 14, d), (12, 13, 14, d), (14, 19, 14, d), (15, 17, 14, a)]. What motifs present in the given undirected dynamic graph?\n"""
    # Call Agent to get response (clear_history=True by default)
    response = agent_manager.get_response(message, clear_history=True)

    if response["success"]:
        print("\n--- Agent Response (Success) ---")
        print(response["content"])
    else:
        print("\n--- Agent Response (Failed) ---")
        print(f"Failed to get a valid response. Error Info: {response['content']}")
    print(f"Processing Info: {response['info']}")

    print("\n--- Agent Exited ---")