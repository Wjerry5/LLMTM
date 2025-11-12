from .base import DyGraphTask
import numpy as np
import re
import random
from collections import defaultdict

class DyGraphTaskWhenDirectLink(DyGraphTask):
    """
    Task: Ask for the time when two given nodes are first directly connected, and the time when they are first disconnected.
    Corresponds to original scripts: When.py, QA.py (Operation), evaluate.py (Operation)
    """
    def generate_qa(self, info, *args, **kwargs):
        """
        Generate QA pairs. Transplanted from QA.py (Operation).
        Select an edge that has at least one add and one delete operation, find the first add and first delete times.
        """
        context_orig = info['edge_index']
        # Ensure u, v, t are integers, op is string
        context_typed = []
        for item in context_orig:
            try:
                u, v, t, op = item
                # Normalize edge representation (u <= v)
                u_int, v_int = int(u), int(v)
                if u_int > v_int:
                    u_int, v_int = v_int, u_int
                context_typed.append((u_int, v_int, int(t), str(op)))
            except ValueError as e:
                print(f"Warning (WhenDirectLink - QA): Skipping unparseable line: {item} - {e}")
                continue
            except Exception as e:
                print(f"Warning (WhenDirectLink - QA): Error processing line: {item} - {e}")
                continue

        if not context_typed:
            print("Error (WhenDirectLink - QA): No valid context data.")
            # In this case, cannot generate valid QA, can return None or throw error
            # Return None to let caller (like gen in runner.py) know to skip this instance
            return None

        # Sort by time
        context_typed.sort(key=lambda x: x[2])

        # Find all edges that have at least one 'a' and one 'd' operation
        edge_operations = defaultdict(list)
        for u, v, t, op in context_typed:
            edge = (u, v) # Already normalized
            edge_operations[edge].append((t, op))

        eligible_edges = []
        for edge, operations in edge_operations.items():
            has_add = any(op == 'a' for _, op in operations)
            has_delete = any(op == 'd' for _, op in operations)
            if has_add and has_delete:
                eligible_edges.append(edge)

        if not eligible_edges:
            print(f"Warning (WhenDirectLink - QA): No edges found with both add and delete operations.")
            # Similarly, cannot generate valid QA
            return None

        # Randomly select a qualified edge
        selected_edge = random.choice(eligible_edges)
        u_query, v_query = selected_edge # Extract query node pair

        # Find the first add and first delete operation times for this edge
        operations = edge_operations[selected_edge]
        first_add_time = -1
        first_delete_time = -1

        # Find first 'a'
        for t, op in operations:
            if op == 'a':
                first_add_time = t
                break # Stop when first one is found

        # Find first 'd' (must occur after first 'a')
        if first_add_time != -1:
             for t, op in operations:
                  if op == 'd' and t >= first_add_time: # Ensure delete is not earlier than add
                       first_delete_time = t
                       break # Stop when first one is found

        # Must find both first add and first delete times
        if first_add_time == -1 or first_delete_time == -1:
             print(f"Warning (WhenDirectLink - QA): Edge {selected_edge} not found valid first add/delete pair.")
             return None # Cannot generate valid QA

        # Answer is (first add time, first delete time)
        answer = (first_add_time, first_delete_time)

        # context for prompt is the original unsorted, untyped list
        qa = {
            "context": context_orig, # Use original format passed to prompt
            "query": (u_query, v_query),
            "answer": answer,
            "task": self.task
        }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        # Transplanted from When.py
        return (f"Your task is to answer when the given two nodes are first connected, and after that, when they are first disconnected in dynamic graph? Two nodes are connected if and only if there exists a direct edge between them, ignoring indirect connections. (The problem guarantees that the two nodes are connected at least once and disconnected at least once).")
        # return (f"Your task is to answer when the given two nodes are first connected, and after that, when they are first disconnected in dynamic graph? Two nodes are connected when the direct edge between them added. The two edges are disconnected when the direct edge deleted. (The problem guarantees that the two nodes are connected directly at least once and deleted at least once).")

    def generate_instructor_answer(self, *args, **kwargs):
        # Transplanted from When.py
        return "Give the answer as two integer numbers at the last of your response after 'Answer:'."
    def generate_prompt_examplars(self, num, *args, **kwargs):
        # Transplanted from When.py, ensure correct format
        qa_examples_raw = [
            (
                [(0, 5, 0, 'a'), (2, 7, 0, 'a'), (6, 8, 0, 'a'), (0, 5, 1, 'd'), (5, 7, 2, 'a'), (7, 8, 2, 'a'), (1, 2, 3, 'a'), (0, 2, 4, 'a'), (0, 3, 4, 'a'), (0, 6, 4, 'a'), (1, 2, 4, 'd'), (2, 3, 4, 'a'), (4, 5, 4, 'a'), (6, 7, 4, 'a')], # context
                (0, 5), # query
                """**Chain of Thought:**
1. Objective: Locate two specific timestamps for the edge between node 0 and node 5. First, the earliest time this edge is added (operation 'a'). Second, the earliest time this edge is deleted (operation 'd').
2. Scan for Connection: Go through all edges in the provided list. Identify the tuple (u, v, t, 'a') where u and v are 0 and 5 (order doesn't matter for undirected graphs). Record the t value from the first such occurrence. This is first_connected_time.
3. Scan for Disconnection: Similarly, go through all edges in the provided list. Identify the tuple (u, v, t, 'd') where u and v are 0 and 5. Record the t value from the first such occurrence. This is first_disconnected_time.
4. Final Output: Present first_connected_time and first_disconnected_time.""", # COT
                (0, 1) # answer - (first_add_time, first_delete_time)
            )
        ]
        # 
        qa_formatted = []
        for c, q, s, a in qa_examples_raw: 
             answer_str = f"{a[0]}, {a[1]}"
             qa_formatted.append([list(c), list(q), s, answer_str])  


        if num == 0:
            return ""
        examples = []
        for c, q, s, a_str in qa_formatted[:num]:  
             context_prompt = self.generate_context_prompt(c) 
             question_prompt = self.generate_prompt_question(q) 
             if self.args.add_cot == 1:
                 example = f"{context_prompt}{question_prompt}\n{s}\nAnswer: {a_str}"
             else:
                 example = f"{context_prompt}{question_prompt} Answer: {a_str}" # 
             examples.append(example)

        if num == 1:
            prompt = "Here is an example: " + "\n".join(examples)
        else:
            prompt = f"Here are {num} examples: " + "\n".join(examples)
        return prompt


    def generate_prompt_question(self, query = None, *args, **kwargs):
        # Transplanted from When.py
        u, v = query
        return f" When are node {u} and node {v} first connected, and after that, when they are first disconnected in dynamic graph?"

    def evaluate(self, qa, response, use_agent):
        """
       
        :
            metric: 1 (correct), 0 (incorrect but format correct), -1 (cannot parse format)
            extracted_answer: parsed (t1, t2) tuple or None
        """
        true_times = qa['answer'] # (t_add, t_del)

        # Find content after the last "Answer:" marker
        answer_marker = "Answer:"
        answer_start_index = response.rfind(answer_marker)
        if answer_start_index != -1 or use_agent == 1:
            if use_agent == 1:
                extracted_part = response[0:].strip()
            else:
                extracted_part = response[answer_start_index + len(answer_marker):].strip()
            extracted_part = extracted_part.lower()
            
            # Try to match format: "The nodes are first connected at time \\boxed{2} and then disconnected at time \\boxed{3}."
            match_nodes_boxed = re.search(r'nodes are first connected at time \\+boxed\{(\d+)\}.*?disconnected at time \\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_nodes_boxed:
                try:
                    appear_time = int(match_nodes_boxed.group(1))
                    disconnect_time = int(match_nodes_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First connected at time \\boxed{3}, first disconnected at time \\boxed{4}."
            match_first_boxed = re.search(r'first connected at time \\+boxed\{(\d+)\}.*?first disconnected at time \\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_first_boxed:
                try:
                    appear_time = int(match_first_boxed.group(1))
                    disconnect_time = int(match_first_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "### Final Answer\n- **First Connected:** \\boxed{2}\n- **First Disconnected:** \\boxed{4}"
            match_final_answer_boxed = re.search(r'###\s*final answer.*?\*\*first connected\*\*.*?\\+boxed\{(\d+)\}.*?\*\*first disconnected\*\*.*?\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_final_answer_boxed:
                try:
                    appear_time = int(match_final_answer_boxed.group(1))
                    disconnect_time = int(match_final_answer_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "**First Connected:** \\boxed{2}\n- **First Disconnected:** \\boxed{4}"
            # 
            match_connected_disconnected_boxed = re.search(r'\*\*first connected\*\*:.*?\\+boxed\{(\d+)\}.*?-\s*\*\*first disconnected\*\*:.*?\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_connected_disconnected_boxed:
                try:
                    appear_time = int(match_connected_disconnected_boxed.group(1))
                    disconnect_time = int(match_connected_disconnected_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 
            match_markdown_boxed_precise = re.search(r'\*\*first connected\*\*:\s*\\+boxed\{(\d+)\}.*?\*\*first disconnected\*\*:\s*\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_markdown_boxed_precise:
                try:
                    appear_time = int(match_markdown_boxed_precise.group(1))
                    disconnect_time = int(match_markdown_boxed_precise.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "**First Connected:** \\boxed{2}\n- **First Disconnected:** \\boxed{4}" (more general matching)
            match_markdown_boxed = re.search(r'\*\*first connected\*\*.*?\\+boxed\{(\d+)\}.*?\*\*first disconnected\*\*.*?\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_markdown_boxed:
                try:
                    appear_time = int(match_markdown_boxed.group(1))
                    disconnect_time = int(match_markdown_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First connected at t=1, first disconnected at t=3."
            match_simple_t_format = re.search(r'first connected at t=(\d+).*?first disconnected at t=(\d+)', extracted_part, re.DOTALL)
            if match_simple_t_format:
                try:
                    appear_time = int(match_simple_t_format.group(1))
                    disconnect_time = int(match_simple_t_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First connected at **t=1**, first disconnected at **t=2**"
            match_t_format = re.search(r'first connected at \*\*t=(\d+)\*\*.*?first disconnected at \*\*t=(\d+)\*\*', extracted_part, re.DOTALL)
            if match_t_format:
                try:
                    appear_time = int(match_t_format.group(1))
                    disconnect_time = int(match_t_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First connected at time: 0  \nFirst disconnected at time: 2"
            match_time_colon_newline = re.search(r'first connected at time:\s*(\d+).*?first disconnected at time:\s*(\d+)', extracted_part, re.DOTALL)
            if match_time_colon_newline:
                try:
                    appear_time = int(match_time_colon_newline.group(1))
                    disconnect_time = int(match_time_colon_newline.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First Connection: 2  \nFirst Disconnection: 3"
            match_connection_disconnection = re.search(r'first connection:\s*(\d+).*?first disconnection:\s*(\d+)', extracted_part, re.DOTALL)
            if match_connection_disconnection:
                try:
                    appear_time = int(match_connection_disconnection.group(1))
                    disconnect_time = int(match_connection_disconnection.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First Connection Time: 3  \nFirst Disconnection Time: 4"  
            match_connection_time = re.search(r'first connection time:\s*(\d+).*?first disconnection time:\s*(\d+)', extracted_part, re.DOTALL)
            if match_connection_time:
                try:
                    appear_time = int(match_connection_time.group(1))
                    disconnect_time = int(match_connection_time.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First Connection: 3\n- First Disconnect: 4"
            match_connection_format = re.search(r'first connection:\s*(\d+).*?first disconnect:\s*(\d+)', extracted_part, re.DOTALL)
            if match_connection_format:
                try:
                    appear_time = int(match_connection_format.group(1))
                    disconnect_time = int(match_connection_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "The nodes first connect at time `3` and then disconnect at time `4`."
            match_nodes_connect_backtick = re.search(r'nodes first connect at time `(\d+)`.*?disconnect at time `(\d+)`', extracted_part, re.DOTALL)
            if match_nodes_connect_backtick:
                try:
                    appear_time = int(match_nodes_connect_backtick.group(1))
                    disconnect_time = int(match_nodes_connect_backtick.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "The nodes first connect at time 0 and disconnect at time 4."
            match_nodes_connect_simple = re.search(r'nodes first connect at time (\d+).*?disconnect at time (\d+)', extracted_part, re.DOTALL)
            if match_nodes_connect_simple:
                try:
                    appear_time = int(match_nodes_connect_simple.group(1))
                    disconnect_time = int(match_nodes_connect_simple.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "The nodes are first connected at timestamp 0 and then disconnected at timestamp 4."
            match_nodes_timestamp_then = re.search(r'nodes are first connected at timestamp (\d+).*?then disconnected at timestamp (\d+)', extracted_part, re.DOTALL)
            if match_nodes_timestamp_then:
                try:
                    appear_time = int(match_nodes_timestamp_then.group(1))
                    disconnect_time = int(match_nodes_timestamp_then.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "The nodes are first connected at timestamp 2 and first disconnected at timestamp 4."
            match_nodes_timestamp = re.search(r'nodes are first connected at timestamp (\d+).*?first disconnected at timestamp (\d+)', extracted_part, re.DOTALL)
            if match_nodes_timestamp:
                try:
                    appear_time = int(match_nodes_timestamp.group(1))
                    disconnect_time = int(match_nodes_timestamp.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "The nodes are first connected at time 2 and then disconnected at time 3."
            match_nodes_are_connected = re.search(r'nodes are first connected at time (\d+).*?disconnected at time (\d+)', extracted_part, re.DOTALL)
            if match_nodes_are_connected:
                try:
                    appear_time = int(match_nodes_are_connected.group(1))
                    disconnect_time = int(match_nodes_are_connected.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "- First connected at time 0.\n- First disconnected at time 1."
            match_list_time = re.search(r'-\s*first connected at time (\d+).*?-\s*first disconnected at time (\d+)', extracted_part, re.DOTALL)
            if match_list_time:
                try:
                    appear_time = int(match_list_time.group(1))
                    disconnect_time = int(match_list_time.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "- First connected at timestamp 3.\n- First disconnected at timestamp 4."
            match_list_timestamp = re.search(r'-\s*first connected at timestamp (\d+).*?-\s*first disconnected at timestamp (\d+)', extracted_part, re.DOTALL)
            if match_list_timestamp:
                try:
                    appear_time = int(match_list_timestamp.group(1))
                    disconnect_time = int(match_list_timestamp.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First Connected: 4  \nFirst Disconnected: 4"
            match_first_connected_disconnected = re.search(r'first connected:\s*(\d+).*?first disconnected:\s*(\d+)', extracted_part, re.DOTALL)
            if match_first_connected_disconnected:
                try:
                    appear_time = int(match_first_connected_disconnected.group(1))
                    disconnect_time = int(match_first_connected_disconnected.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First connected at time `t = 0`, first disconnected at time `t = 1`."
            match_t_equals_format = re.search(r'first connected at time `t = (\d+)`.*?first disconnected at time `t = (\d+)`', extracted_part, re.DOTALL)
            if match_t_equals_format:
                try:
                    appear_time = int(match_t_equals_format.group(1))
                    disconnect_time = int(match_t_equals_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "Connect Time: 1, Disconnect Time: 3"
            match_connect_disconnect_time = re.search(r'connect time:\s*(\d+).*?disconnect time:\s*(\d+)', extracted_part, re.DOTALL)
            if match_connect_disconnect_time:
                try:
                    appear_time = int(match_connect_disconnect_time.group(1))
                    disconnect_time = int(match_connect_disconnect_time.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # Try to match format: "First connected at 3, first disconnected at 4"
            match_simple_connected_at = re.search(r'first connected at (\d+),\s*first disconnected at (\d+)', extracted_part, re.DOTALL)
            if match_simple_connected_at:
                try:
                    appear_time = int(match_simple_connected_at.group(1))
                    disconnect_time = int(match_simple_connected_at.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            match_any_two_boxed = re.search(r'\\+boxed\{(\d+)\}.*?\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_any_two_boxed:
                try:
                    appear_time = int(match_any_two_boxed.group(1))
                    disconnect_time = int(match_any_two_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            match_boxed = re.search(r'\\+boxed\{(\d+)\},\s*\\+boxed\{(\d+)\}', extracted_part)
            if match_boxed:
                try:
                    appear_time = int(match_boxed.group(1))
                    disconnect_time = int(match_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 
                    return -1, None # Format error
            
            # Try to match format: "**Answer:**\n- First connected at time **1**.\n- First disconnected at time **3**."
            match_markdown_format = re.search(r'first connected at time \*\*(\d+)\*\*.*?first disconnected at time \*\*(\d+)\*\*', extracted_part, re.IGNORECASE | re.DOTALL)
            if match_markdown_format:
                try:
                    appear_time = int(match_markdown_format.group(1))
                    disconnect_time = int(match_markdown_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 
                    return -1, None # Format error
            
            # Try to match format: "First connected at time 0, first disconnected at time 4."
            match_simple_format = re.search(r'first connected at time (\d+),\s*first disconnected at time (\d+)', extracted_part, re.IGNORECASE)
            if match_simple_format:
                try:
                    appear_time = int(match_simple_format.group(1))
                    disconnect_time = int(match_simple_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 
                    return -1, None # Format error
            
            # Try to match format: "First connected at time: 1, first disconnected at time: 5"
            match_colon_format = re.search(r'first connected at time:\s*(\d+),\s*first disconnected at time:\s*(\d+)', extracted_part, re.IGNORECASE)
            if match_colon_format:
                try:
                    appear_time = int(match_colon_format.group(1))
                    disconnect_time = int(match_colon_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 
                    return -1, None # Format error
            
            # Try to match format: "First connected at time `3`, first disconnected at time `4`"
            match_formatted = re.search(r'first connected at time `(\d+)`, first disconnected at time `(\d+)`', extracted_part, re.IGNORECASE)
            if match_formatted:
                try:
                    appear_time = int(match_formatted.group(1))
                    disconnect_time = int(match_formatted.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:

                    return -1, None 

            match_answer_direct = re.search(r'^\s*(\d+),\s*(\d+)', extracted_part)
            if match_answer_direct:
                try:
                    appear_time = int(match_answer_direct.group(1))
                    disconnect_time = int(match_answer_direct.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:

                    return -1, None 
            
            patterns = [
                r'\*\*answer:\*\*\s*(\d+),\s*(\d+)(?:[.\s]*$|[.\s]*\n.*?(?:therefore|thus|hence|so),?\s+(?:the\s+)?answer\s+is\s*\1,\s*\2)',  # **Answer:** 2, 3\nTherefore, the answer is 2, 3
                r'\*\*answer:\*\*\s*(\d+),\s*(\d+)[.\s]*$',  # **Answer:** 2, 3
                r'answer:\s*(\d+),\s*(\d+)',  # Answer: 2, 3
            ]
            
            for pattern in patterns:
                match = re.search(pattern, extracted_part, re.IGNORECASE)
                if match:
                    try:
                        appear_time = int(match.group(1))
                        disconnect_time = int(match.group(2))
                        predicted_times = [appear_time, disconnect_time]
                        print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                        metric = 1 if predicted_times == true_times else 0
                        return metric, predicted_times
                    except ValueError:
                        continue
            if use_agent == 1:
              
                try:
                    # 
                    clean_part = extracted_part.strip().strip('[]')
                    numbers = [int(x.strip()) for x in clean_part.split(',')]
                    if len(numbers) == 2:
                        appear_time = numbers[0]
                        disconnect_time = numbers[1]
                        predicted_times = [appear_time, disconnect_time]
                        print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                        metric = 1 if predicted_times == true_times else 0
                        return metric, predicted_times
                    else:
                        return -1, None
                except (ValueError, IndexError):
                    return -1, None
            # 
            match = re.search(r'(\d+)[,\s]+(\d+)\s*$', extracted_part)
            if match:
                try:
                    appear_time = int(match.group(1))
                    disconnect_time = int(match.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 
                    return -1, None # Format error
            else:
                # If two numbers in the correct format not found
                return -1, None # Format error
        else:
            # If "Answer:" marker not found
             return -1, None # Format error 