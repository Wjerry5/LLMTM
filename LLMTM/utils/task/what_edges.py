from .base import DyGraphTask
import numpy as np
import re
import random

# Helper function: Simulate graph evolution to find edges existing at specific time
# Transplanted and modified from QA_1.py's process_temporal_graph
def get_existing_edges(temporal_edges_typed, selected_time):
    """
    Process single temporal graph data, find edges existing at selected_time moment.
    
    Parameters:
    temporal_edges_typed - Sorted typed quadruple list [(u, v, t, op), ...] (int, int, int, str)
    selected_time - Target timestamp (int)
    
    Returns:
    existing_edges_at_t - Original add quadruple list of edges existing at selected_time [(u, v, t_add, 'a'), ...]
    """
    if not temporal_edges_typed:
        return []

    current_edges = {} # {(u, v): original_add_quadruple}

    # Iterate through all events with time <= selected_time
    for u, v, t_event, op in temporal_edges_typed:
        if t_event > selected_time:
             break # Since sorted, subsequent events have larger times, no need to process

        edge = tuple(sorted((u, v))) # Normalize edge representation (int, int)

        if op == 'a':
            # Record add operation, store original quadruple (int, int, int, str)
            current_edges[edge] = (u, v, t_event, op)
        elif op == 'd':
            # If edge exists, delete it
            if edge in current_edges:
                del current_edges[edge]

    # Extract original quadruples of edges existing at the end of selected_time
    existing_edges_at_t = list(current_edges.values())

    return existing_edges_at_t

# Helper function: Parse quadruples from list format string
# Transplanted from evaluate_1.py
def parse_quadruples_from_list_string(list_string):
    """Parse quadruple set from list format string"""
    quadruples = set()
    # Find all tuples, e.g. (0, 1, 0, 'a') or (0, 1, 0, a)
    # Slightly flexible regex, allowing quotes to be optional
    tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', list_string)
    for t in tuple_matches:
        try:
            quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
            quadruples.add(quad)
        except ValueError:
             print(f"  Warning (WhatEdgeAtTime - parsing): Skipping tuple with parsing error: {t}")
    return quadruples


class DyGraphTaskWhatEdgeAtTime(DyGraphTask):
    """
    Task: Ask for edges that exist at a given time point in the dynamic graph.
    Corresponds to original scripts: What.py, QA_1.py, evaluate_1.py
    """
    def generate_qa(self, info, *args, **kwargs):
        """
        Generate QA pairs. Transplanted from QA_1.py.
        Randomly select a time point and calculate the edges that exist at that time point.
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
                print(f"Warning (WhatEdgeAtTime - QA): Skipping unparseable line: {item} - {e}")
                continue
            except Exception as e:
                print(f"Warning (WhatEdgeAtTime - QA): Error processing line: {item} - {e}")
                continue

        if not context_typed or not timestamps:
            print("Error (WhatEdgeAtTime - QA): No valid context data or timestamps.")
            return None # Cannot generate QA

        # Sort by time
        context_typed.sort(key=lambda x: x[2])

        # Randomly select a valid timestamp
        selected_time = random.choice(list(timestamps))

        # Calculate edges that exist at this time point (using helper function)
        existing_edges = get_existing_edges(context_typed, selected_time)

        # Answer is the list of existing edges
        answer = existing_edges # Already in [(u, v, t_add, 'a'), ...] format

        qa = {
            "context": context_orig, # Original format context
            "query": selected_time, # Query is the time point
            "answer": answer,       # Answer is quadruple list
            "task": self.task
        }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        # Transplanted from What.py
        # return (f"Your task is to answer what edges exist at a given moment in dynamic graph? (The order of edges doesn't matter)")
        return (f"Your task is to answer what edges exist at a given moment in dynamic graph? (The order of edges doesn't matter)")
    def generate_instructor_answer(self, *args, **kwargs):
        # Transplanted from What.py
        return "Give the answer as a list of 4-tuples (u, v, t, a/d) at the end of your response after 'Answer:'."

    def generate_prompt_examplars(self, num, *args, **kwargs):
        # Transplanted from What.py, ensure correct format
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
5. Output: Present this final list of active edges.""", # CoT (Chain of Thought) - To be compatible with base.py's 4-element format
                [(0, 9, 0, 'a'), (1, 9, 0, 'a'), (2, 5, 0, 'a'), (1, 2, 1, 'a'), (2, 6, 1, 'a'), (3, 7, 1, 'a')], # answer (edges existing at t=1)
            )
        ]
        # Convert format - Fix: correctly handle 4 elements
        qa_formatted = [ [list(c), q, s, list(a)] for c, q, s, a in qa_examples_raw]
        return self.make_qa_example(num, qa_formatted)
        
    def generate_prompt_question(self, query = None, *args, **kwargs):
        # Transplanted from What.py
        selected_time = query
        return f" What edges exist at time {selected_time} in dynamic graph?"

    def evaluate(self, qa, response, use_agent):
        """
        Evaluate model response. Transplanted from evaluate_1.py.
        Parse the quadruple list output by the model and compare with the answer set.
        Return value:
            metric: 1 (correct), 0 (incorrect but format correct), -1 (unable to parse format)
            extracted_answer: Parsed quadruple set or None
        """
        true_quads_set = set(tuple( (int(u), int(v), int(t), str(op)) ) for u,v,t,op in qa['answer'])

        # Try multiple ways to extract answer
        extracted_part = None
        
# Find content after the last "Answer:"
        # answer_marker = "Answer:"
        # answer_indices = [m.start() for m in re.finditer(re.escape(answer_marker), response)]
        
        # if len(answer_indices) >= 2:
        #     # If there are multiple Answers, take content between second-to-last and last Answer
        #     start_index = answer_indices[-2]
        #     end_index = answer_indices[-1]
        #     extracted_part = response[start_index + len(answer_marker):end_index].strip()
        # elif len(answer_indices) == 1:
        #     # If there is only one Answer, take content from that Answer to the end
        #     start_index = answer_indices[0]
        #     extracted_part = response[start_index + len(answer_marker):].strip()
        # else:
        #     # If no Answer marker found
        #     return -1, None
        # extracted_part = response.split(answer_marker)[-1].strip()
        # Try to parse list content


        # 1. First try to find content after the last "Answer:"
        answer_markers = ["Answer:", "The answer is:", "Final answer:"]
        for marker in answer_markers:
            if marker in response:
                answer_indices = [m.start() for m in re.finditer(re.escape(marker), response)]
                if answer_indices:
                    # Take content after the last marker
                    start_index = answer_indices[-1]
                    extracted_part = response[start_index + len(marker):].strip()
                    break
        
        # 2. If no marker found, try to parse entire response directly
        if extracted_part is None:
            # Try to extract directly from response
            extracted_part = response.strip()
        
        # 3. If extracted content contains multiple paragraphs, prioritize using the last non-empty paragraph
        if extracted_part:
            paragraphs = [p.strip() for p in extracted_part.split('\n\n') if p.strip()]
            if paragraphs:
                extracted_part = paragraphs[-1]

        # 4. Try to parse list content
        predicted_quads_set = parse_quadruples_from_list_string(extracted_part)

        if predicted_quads_set is not None: # If parsing successful (may be empty set)
            metric = 1 if predicted_quads_set == true_quads_set else 0
            if metric == 0:
                print(f"  Ground truth set: {true_quads_set}") # For debugging
                print(f"  Model set: {predicted_quads_set}") # For debugging
                print(f"  Extracted text: {extracted_part}") # For debugging
            return metric, predicted_quads_set
        else:
            # Parsing returning None usually means internal warnings, but here we treat it as format error
            print(f"  Warning: Cannot parse valid quadruples from the following text:\n{extracted_part}")
            return -1, None # Format error 