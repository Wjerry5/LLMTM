from .base import DyGraphTask
import numpy as np
import re
import random
from collections import defaultdict

# Helper function: Simulate graph evolution to find edges existing at a specific time
# Ported and modified from QA_1.py's process_temporal_graph
def get_existing_edges(temporal_edges_typed, selected_time):
    """
    Processes a single temporal graph data, finds edges existing at selected_time.

    Parameters:
    temporal_edges_typed - List of typed quadruples sorted by time [(u, v, t, op), ...] (int, int, int, str)
    selected_time - Target timestamp (int)

    Returns:
    existing_edges_at_t - List of original add quadruples for edges existing at selected_time [(u, v, t_add, 'a'), ...]
    """
    if not temporal_edges_typed:
        return []

    current_edges = {} # {(u, v): original_add_quadruple}

    # Iterate through all events with time <= selected_time
    for u, v, t_event, op in temporal_edges_typed:
        if t_event > selected_time:
             break # Because it's sorted, subsequent events have larger times, no need to process

        edge = tuple(sorted((u, v))) # Normalize edge representation (int, int)

        if op == 'a':
            # Record add operation, store original quadruple (int, int, int, str)
            current_edges[edge] = (u, v, t_event, op)
        elif op == 'd':
            # If edge exists, delete it
            if edge in current_edges:
                del current_edges[edge]

    # Extract the original quadruples of edges existing at the end of selected_time
    existing_edges_at_t = list(current_edges.values())

    return existing_edges_at_t

# Helper function: Parse quadruples from a list-formatted string
# Ported from evaluate_1.py
# def parse_quadruples_from_list_string(list_string):
#     """Parses a set of quadruples from a list-formatted string"""
#     quadruples = set()
#     # Find all tuples, e.g., (0, 1, 0, 'a') or (0, 1, 0, a)
#     # Slightly looser regex, allows optional quotes
#     tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', list_string)
#     for t in tuple_matches:
#         try:
#             quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
#             quadruples.add(quad)
#         except ValueError:
#              print(f"  Warning (WhatEdgeAtTime - Parse): Skipping incorrectly parsed tuple: {t}")
#     return quadruples

def _parse_text_for_quadruples(text_to_search):
    """
    Helper function: Runs all regex patterns on the given text block to find tuples.
    """
    quadruples_list = []
    
    # Pattern 1: Standard format (0, 1, 0, 'a') or (0, 1, 0, a)
    tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', text_to_search)
    
    # Pattern 2: Looser format (retaining your original logic)
    if not tuple_matches:
        tuple_matches = re.findall(r'\(\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*([ad])\s*\)', text_to_search)
    
    # Pattern 3: Loosest format (retaining your original logic)
    if not tuple_matches:
        tuple_matches = re.findall(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([ad])', text_to_search)

    for t in tuple_matches:
        try:
            quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
            quadruples_list.append(quad)
        except (ValueError, IndexError):
            pass
    
    return quadruples_list

def parse_quadruples_from_response_content(content):
    """
    Parses a list of quadruples from the model response content in priority order
    P1: [...] -> P2: Answer: -> P3: Full text
    """
    
    # --- Priority 1: Find [...] ---
    # Find all [...] and take the last one
    list_matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
    if list_matches:
        last_list_content = list_matches[-1].strip()
        result = _parse_text_for_quadruples(last_list_content)
        # Only return if the content inside [...] *actually* contains tuples
        if result:
            return result
    
    # --- Priority 2: Find "Answer:" ---
    # (If P1 fails, execute P2)
    # re.IGNORECASE ignores case, re.DOTALL makes . match newlines
    answer_match = re.search(r'Answer:\s*(.*)', content, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        result = _parse_text_for_quadruples(answer_content)
        if result:
            return result
            
    # --- Priority 3: Find in full content ---
    # (If P1 and P2 both fail, execute P3)
    # The helper function will parse the complete content string here
    return _parse_text_for_quadruples(content)

class DyGraphTaskWhatEdgeAtTime(DyGraphTask):
    """
    Task: Ask what edges exist in the dynamic graph at a given time point.
    Corresponds to original scripts: What.py, QA_1.py, evaluate_1.py
    """
    def generate_qa(self, info, *args, **kwargs):
        """
        Generates QA pairs. Ported from QA_1.py.
        Randomly selects a time point, calculates the edges existing at that time.
        """
        context_orig = info['edge_index']
        # Type conversion and sorting
        context_typed = []
        timestamps = set()
        for item in context_orig:
            try:
                u, v, t, op = item
                u_int, v_int, t_int = int(u), int(v), int(t)
                op_str = str(op)
                context_typed.append((u_int, v_int, t_int, op_str))
                timestamps.add(t_int)
            except ValueError as e:
                print(f"Warning (WhatEdgeAtTime - QA): Skipping unparseable row: {item} - {e}")
                continue
            except Exception as e:
                print(f"Warning (WhatEdgeAtTime - QA): Error processing row: {item} - {e}")
                continue

        if not context_typed or not timestamps:
            print("Error (WhatEdgeAtTime - QA): No valid context data or timestamps.")
            return None # Cannot generate QA

        # Sort by time
        context_typed.sort(key=lambda x: x[2])

        # Randomly select a valid timestamp
        selected_time = random.choice(list(timestamps))

        # Calculate edges existing at that time point (using helper function)
        existing_edges = get_existing_edges(context_typed, selected_time)

        # The answer is the list of existing edges
        answer = existing_edges # Already in [(u, v, t_add, 'a'), ...] format

        qa = {
            "context": context_orig, # Context in original format
            "query": selected_time, # Query is the time point
            "answer": answer,       # Answer is the list of quadruples
            "task": self.task
        }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        # Ported from What.py
        # return (f"Your task is to answer what edges exist at a given moment in dynamic graph? (The order of edges doesn't matter)")
        return (f"Your task is to answer what edges exist at a given moment in dynamic graph? (The order of edges doesn't matter)")
    def generate_instructor_answer(self, *args, **kwargs):
        # Ported from What.py
        return "Give the answer as a list of 4-tuples (u, v, t, a/d) at the end of your response after 'Answer:'."

    def generate_prompt_examplars(self, num, *args, **kwargs):
        # Ported from What.py, ensure format is correct
        qa_examples_raw = [
            (
                [(0, 9, 0, 'a'), (1, 9, 0, 'a'), (2, 5, 0, 'a'), (1, 2, 1, 'a'), (2, 6, 1, 'a'), (3, 7, 1, 'a'), (1, 9, 2, 'd'), (4, 5, 2, 'a'), (4, 7, 2, 'a'), (7, 8, 2, 'a')],
                1, # query (time)
                """**Chain of Thought:**
1. Objective: Determine which edges are active (connected) at the specific query_time of 1. An edge is active if it was added at or before time 1 and has NOT been deleted at or before time 1.
2. Initialize Edge Status Tracker: Prepare a record to track the current state of each unique edge (u, v). Initially, assume no edges exist.
3. Process Events Chronologically (up to query_time=1):
    * Go through each event (u, v, t, operation) in the input list.
    * Crucial Filter: If t > 1, ignore this event as it happens after the moment we are interested in.
    * For events where t <= 1:
        * If operation is 'a': Add the edge (u, v, t, a) to our set of currently existing edges.
        * If operation is 'd': Remove the edge (u, v, t, a) from our set of currently existing edges.
4. Final Active Edges at t=1: After processing all events that occurred at or before time 1, the edges remaining in our collection are the ones that exist at t=1.
5. Output: Present this final list of active edges.""", # CoT (Thought process) - for compatibility with base.py's 4-element format
                [(0, 9, 0, 'a'), (1, 9, 0, 'a'), (2, 5, 0, 'a'), (1, 2, 1, 'a'), (2, 6, 1, 'a'), (3, 7, 1, 'a')], # answer (edges existing at t=1)
            )
        ]
        # Convert format - Fix: correctly handle 4 elements
        qa_formatted = [ [list(c), q, s, list(a)] for c, q, s, a in qa_examples_raw]
        return self.make_qa_example(num, qa_formatted)
        
    def generate_prompt_question(self, query = None, *args, **kwargs):
        # Ported from What.py
        selected_time = query
        return f" What edges exist at time {selected_time} in dynamic graph?"

    def evaluate(self, qa, response, use_agent):
        """
        Evaluates the model response. Ported from evaluate_1.py.
        Parses the list of quadruples from the model output and compares with the answer set.
        Returns:
            metric: 1 (correct), 0 (wrong but correct format), -1 (cannot parse format)
            extracted_answer: The parsed set of quadruples or None
        """
        true_quads_set = set(tuple( (int(u), int(v), int(t), str(op)) ) for u,v,t,op in qa['answer'])

#         # Try multiple ways to extract the answer
#         extracted_part = None
        
# # Find content after the last "Answer:"
#         answer_marker = "Answer:"
#         answer_indices = [m.start() for m in re.finditer(re.escape(answer_marker), response)]
        
#         if len(answer_indices) >= 2:
#             # If there are multiple Answers, take the content between the second-to-last and the last Answer
#             start_index = answer_indices[-2]
#             end_index = answer_indices[-1]
#             extracted_part = response[start_index + len(answer_marker):end_index].strip()
#         elif len(answer_indices) == 1:
#             # If there is only one Answer, take the content from that Answer to the end
#             start_index = answer_indices[0]
#             extracted_part = response[start_index + len(answer_marker):].strip()
#         else:
#             # If the Answer marker is not found
#             return -1, None
#         # extracted_part = response.split(answer_marker)[-1].strip()
#         # Try to parse list content


#         # 1. First, try to find content after the last "Answer:"
#         # answer_markers = ["Answer:", "答案:", "The answer is:", "最终答案:", ""]
#         # for marker in answer_markers:
#         #     if marker in response:
#         #         answer_indices = [m.start() for m in re.finditer(re.escape(marker), response)]
#         #         if answer_indices:
#         #             # Take content after the last marker
#         #             start_index = answer_indices[-1]
#         #             extracted_part = response[start_index + len(marker):].strip()
#         #             break
        
#         # # 2. If no marker is found, try to parse the entire response
#         # if extracted_part is None:
#         #     # Try to extract directly from the response
#         #     extracted_part = response.strip()
        
#         # # 3. If the extracted content contains multiple paragraphs, prioritize the last non-empty paragraph
#         # if extracted_part:
#         #     paragraphs = [p.strip() for p in extracted_part.split('\n\n') if p.strip()]
#         #     if paragraphs:
#         #         extracted_part = paragraphs[-1]

#         # 4. Try to parse list content
        

        extracted_part = parse_quadruples_from_response_content(response)
        predicted_quads_set = set(extracted_part)

        if predicted_quads_set is not None: # If parsing was successful (could be an empty set)
            metric = 1 if predicted_quads_set == true_quads_set else 0
            if metric == 0:
                print(f"  Ground truth set: {true_quads_set}") # For debugging
                print(f"  Model's set: {predicted_quads_set}") # For debugging
                print(f"  Extracted text: {extracted_part}") # For debugging
            return metric, predicted_quads_set
        else:
            # Parsing returned None, usually means internal warning, but we treat as format error
            print(f"  Warning: Could not parse valid quadruples from the following text:\n{extracted_part}")
            return -1, None # Format error