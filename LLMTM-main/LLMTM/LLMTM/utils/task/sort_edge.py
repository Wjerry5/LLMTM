from .base import DyGraphTask
import numpy as np
import re
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
def judge_ordered(con):
    con = np.array(con)
    for i in range(len(con)-1):
        if int(con[i,2]) > int(con[i+1,2]):
            return False
    return True
    
def parse_quadruples_from_response_content(content):
    """Parse quadruple list from model response content"""
    quadruples_list = []
    
    # Directly find all list formats [...], and take the last one
    list_matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
    
    if list_matches:
        # Take the last matched list
        list_content = list_matches[-1].strip()
        # Try multiple matching patterns
        tuple_matches = []
        
        # Pattern 1: Standard format (0, 1, 0, 'a') or (0, 1, 0, a)
        tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', list_content)
        
        # Pattern 2: More flexible format, allowing optional commas
        if not tuple_matches:
            tuple_matches = re.findall(r'\(\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*([ad])\s*\)', list_content)
        
        # Pattern 3: Most flexible format, just number-number-number-letter combination
        if not tuple_matches:
            tuple_matches = re.findall(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([ad])', list_content)
        
        for t in tuple_matches:
            try:
                # Parse as (int, int, int, str)
                quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
                quadruples_list.append(quad)
            except (ValueError, IndexError):
                 pass  # Silently skip tuples with parsing errors
    
    return quadruples_list

class DyGraphTaskSortEdge(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info.get('edge_index', [])
        # Shuffle the content in context
        import random
        context_shuffled = context.copy()
        random.shuffle(context_shuffled)
        context = context_shuffled
        # Ensure context is a list of (int, int, int, str)
        context_typed = [(int(u), int(v), int(t), str(op)) for u, v, t, op in context]

        qa = {
            "context": context_typed,
            # query is usually empty for sorting tasks
            "query": [],
            # answer can be empty here, as evaluation is based on completeness and ordering
            "answer": [],
            "task": self.task
        }
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to SORT the edges by time from earliest to latest, Ensure that the number of edges remains consistent with the original."
        # return f"**Your task** is to **SORT** the edges by time from **Earliest to Latest**. Ensure that **the number of edges remains CONSISTENT** with the original input.\n\n"
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a list of 4-tuples at the end of your response after 'Answer:'. Do not output any other content (such as code, text or explanation)."
        # return "**Give the answer** as a **list of 4-tuples** at the end of your response, after `Answer:`.\n"
#     def generate_instructor_cot(self, *args, **kwargs):
# #         return """## Step-by-Step Reasoning:
# # 1. **Sort the Edges**: Sort all edges by the third element `t`, from smallest to largest, if there are multiple edges with the same timestamp, preserve their original order.
# # 2. **Edge Consistency**: Do not remove or duplicate any edges. The number of edges in the output must exactly match the input.
# # 3. **Output the Answer**: Return the sorted list of edges as a list of 4-tuples at the end of response."""
#         return """## Step-by-Step Reasoning:
# 1. **Determine Timestamps**: Scan the raw data to identify all the timestamps(the third element `t`) that appear, and sort them in ascending order.
# 2. **Group Data**: For each timestamp, scan the raw data to obtain the set of edges with the same timestamp.
# 3. **Merge Edge Sets**: Combine all the edge sets in the order of their timestamps to form the final list.
# 4. **Edge Consistency**: Ensure the number of edges in the output must exactly match the input.
# 5. **Output the Answer**: Return the sorted list of edges as a list of 4-tuples at the end of the response."""
#     
    def generate_prompt_examplars(self, num, *args, **kwargs):
        # Transplanted from temporal_understand.py, and ensure correct format
        # Note: Need to ensure the tuple elements here are of correct type (int, int, int, str)
        qa_examples_raw = [
            (
            [(1, 8, 3, 'a'), (2, 9, 0, 'd'), (0, 7, 2, 'a'), (5, 6, 4, 'd'), (4, 7, 1, 'a'), (3, 8, 5, 'a'), (1, 6, 2, 'd'), (0, 5, 0, 'a'), (4, 9, 3, 'a'), (2, 5, 1, 'a'), (6, 7, 4, 'a'), (0, 3, 3, 'd'), (1, 5, 5, 'd'), (2, 8, 4, 'a'), (3, 9, 0, 'a')], # context (will be formatted by make_qa_example)
            [], # query
            """**Chain of Thought:**
1. My goal is to sort the list of edges based on their timestamp. The timestamp is the third element in each tuple.
2. I'll go through the input list and group the edges by their timestamp, from the earliest to the latest.
    * t=0: I found three edges: (2, 9, 0, d), (0, 5, 0, a), and (3, 9, 0, a). I'll keep them in their original relative order.
    * t=1: I found two edges: (4, 7, 1, a) and (2, 5, 1, a). I'll keep them in their original relative order.
    * t=2: I found two edges: (0, 7, 2, a) and (1, 6, 2, d). I'll keep them in their original relative order.
    * t=3: I found three edges: (1, 8, 3, a), (4, 9, 3, a), and (0, 3, 3, d). I'll keep them in their original relative order.
    * t=4: I found two edges: (5, 6, 4, d) and (6, 7, 4, a). I'll keep them in their original relative order.
    * t=5: I found two edges: (3, 8, 5, a) and (1, 5, 5, d). I'll keep them in their original relative order.
3. Now, I will combines these groups in chronological order (0, 1, 2, 3, 4, 5) to create the final sorted list.
4. I'll do a final check: the input had 15 edges, and my sorted output also has 15 edges. The count is consistent.
5. This leads to the final answer.\n\n""", # COT
            [(2, 9, 0, 'd'), (0, 5, 0, 'a'), (3, 9, 0, 'a'), (4, 7, 1, 'a'), (2, 5, 1, 'a'), (0, 7, 2, 'a'), (1, 6, 2, 'd'), (1, 8, 3, 'a'), (4, 9, 3, 'a'), (0, 3, 3, 'd'), (5, 6, 4, 'd'), (6, 7, 4, 'a'), (2, 8, 4, 'a'), (3, 8, 5, 'a'), (1, 5, 5, 'd')]# answer (will be formatted by make_qa_example)
            )
        ]
        # Convert internal representation to match make_qa_example's expected format list[list, list, list, list] - Fix: correctly handle 4 elements
        qa_formatted = [ [list(c), list(q), s, list(a)] for c, q, s, a in qa_examples_raw]

        return self.make_qa_example(num, qa_formatted)
        
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f" Sort the edges by time from earliest to latest." # Removed "in the dynamic graph" to match original script
    
    
    
    def evaluate(self, qa, response, *args, **kwargs):
        """
        Evaluate the completeness and ordering of model responses.
        Transplanted from evaluate.py (Temporal).
        Return value:
            metric: 1 (correct), -1 (ordering error), -2 (completeness error), -3 (parsing error)
            extracted_answer_list: Parsed quadruple list or None
        """
        # Get use_agent parameter, default to False
        use_agent = kwargs.get('use_agent', False)
        # 1. Parse model response
        if use_agent:
            quads = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*"(\w+)"\]', response)
            if not quads:
                quads = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*'(\w+)'\]", response)
            if not quads:
                quads = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\w+)\]', response)
            response_quads_list = [(int(u), int(v), int(t), str(op)) for u, v, t, op in quads]
        else:
            response_quads_list = parse_quadruples_from_response_content(response)
        if response_quads_list is None or not response_quads_list: # If parsing failed or empty list
            # Check if original context is also empty
            if not qa.get("context"):
                 return 1, [] # If original context is also empty, treat as empty list match, count as correct
            return -3, None # Parsing failed

        # 2. Completeness check
        # Get original context set from qa (should be created in generate_qa)
        original_quads_set = set(tuple(item) for item in qa.get("context"))
        response_quads_set = set(response_quads_list)
        print(original_quads_set)
        print(response_quads_set)
        is_complete = (original_quads_set == response_quads_set)

        if not is_complete:
            return -2, response_quads_list # Completeness error

        # 3. Sorting check (only when completeness passes)
        is_sorted = judge_ordered(response_quads_list) # Use judge_ordered to check

        if is_sorted:
            return 1, response_quads_list # Correct
        else:
            return -1, response_quads_list # Sorting error

        