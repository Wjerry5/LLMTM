from LLMTM.utils.modif_judge_count import judge, check_temporal_constraints
from .base import DyGraphTask
import re
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

class DyGraphTaskModifyDyG(DyGraphTask):
    """
    Task: Modify the dynamic graph to contain the specified motif
    """
    def __init__(self, task, args):
        super().__init__(task, args)
        self.motif_name = args.motif_name

    def generate_qa(self, info, *args, **kwargs):
        """
        Generates QA pairs. Determines the true answer ("Yes" or "No") based on generation info. (English comments)
        """
        context = info.get('edge_index', []) # List of dynamic graph edges to be judged
        original_motif = info.get('target_motif', []) # Motif definition
        # Ensure context and predefined_motif are lists of the correct type
        context = [(int(u), int(v), int(t), str(op)) for u, v, t, op in context]
        N = info.get('motif_nodes')
        M = info.get('motif_edges')
        T = info.get('W')
        answer = info.get('answer',[])
        qa = {
            "context": context,      # Graph to be judged
            "query": [original_motif, [N, M, T], [self.motif_name]], # For judgment tasks, query is usually embedded in the question template
            "answer": answer,           # True answer "Yes" or "No"
            "task": self.task,
            }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        """Generate task instructions (English comments)"""
        return "Your task is to answer How to modify the given undirected dynamic graph so that it contains the given temporal motif?"
    def generate_instructor_answer(self, *args, **kwargs):
        """Generate answer format instructions (English comments)"""

        return "Give the answer as a list of 4-tuples at the end of your response after 'Answer:'"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        """Generate Few-shot examples (English comments)"""
        if self.motif_name == "4-cycle":
            qa_examples_raw = [
                (
                    [(8, 9, 3, 'a'), (3, 7, 4, 'a'), (0, 3, 5, 'a'), (0, 3, 5, 'd'), (3, 7, 5, 'd'), (0, 5, 6, 'a'), (0, 5, 6, 'd'), (2, 7, 6, 'a'), (8, 9, 7, 'd'), (2, 7, 8, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')],
                    [4, 4, 5],
                    ["4-cycle"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "4-cycle" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 2, 3, 5, 7, 8, 9
    * Edges: (8, 9, 3, 'a'), (3, 7, 4, 'a'), (0, 3, 5, 'a'), (0, 5, 6, 'a'), (2, 7, 6, 'a')
3. Since the current graph definitely doesn't contain "4-cycle", I will now search the graph for the "4-cycle with its last edge removed" temporal motif.
    * The pattern for a "4-cycle with its last edge removed" is a sequence of four edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 4.
    * In this undirected graph, the edges (7, 3, 4, 'a'), (3, 0, 5, 'a'), (0, 5, 6, 'a' ) form the above pattern.
    * The timestamps 4, 5, 6 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 6 - 4 = 2, 2 < 4, which satisfies the time window constraint.
4. Therefore, I will add the last edge (5, 7, 7, 'a') within the time window to make the graph contain a complete "4-cycle" temporal motif.
5. This edge is the final answer.""",
                    [(5, 7, 7, 'a')] # answer (will be formatted by make_qa_example)
                )
            ]
        elif self.motif_name == "4-tailedtriangle":
            qa_examples_raw = [
                (
                    [(1, 6, 0, 'a'), (0, 1, 4, 'a'), (1, 6, 4, 'a'), (1, 2, 5, 'a'), (2, 3, 6, 'a'), (2, 3, 6, 'd'), (0, 1, 7, 'd'), (6, 8, 7, 'a'), (1, 2, 8, 'd'), (6, 8, 8, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')],
                    [4, 4, 5],
                    ["4-tailedtriangle"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "4-tailedtriangle" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 6, 8
    * Edges: (1, 6, 0, 'a'), (0, 1, 4, 'a'), (1, 6, 4, 'a'), (1, 2, 5, 'a'), (2, 3, 6, 'a'), (6, 8, 7, 'a')
3. Since the current graph definitely doesn't contain "4-tailedtriangle", I will now search the graph for the "4-tailedtriangle with its last edge removed" temporal motif.
    * The pattern for a "4-tailedtriangle with its last edge removed" is a sequence of five edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 4.
    * In this undirected graph, the edges (0, 1, 4, 'a'), (1, 2, 5, 'a'), (2, 3, 6, 'a') form the above pattern.
    * The timestamps 4, 5, 6 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 6 - 4 = 2, 2 < 4, which satisfies the time window constraint.
4. Therefore, I will add the last edge (3, 1, 7, 'a') within the time window to make the graph contain a complete "4-tailedtriangle" temporal motif.
5. This edge is the final answer.""",
                    [(3, 1, 7, 'a')] # answer (will be formatted by make_qa_example)
                )
            ]
        elif self.motif_name == "4-clique":
            qa_examples_raw = [
                (
                    [(2, 4, 1, 'a'), (0, 3, 4, 'a'), (0, 7, 5, 'a'), (0, 8, 7, 'a'), (7, 8, 8, 'a'), (0, 7, 9, 'd'), (6, 8, 10, 'a'), (4, 7, 11, 'a'), (3, 8, 12, 'a'), (0, 8, 14, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')],
                    [4, 6, 10],
                    ["4-clique"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "4-clique" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 2, 3, 4, 6, 7, 8
    * Edges: (2, 4, 1, 'a'), (0, 3, 4, 'a'), (0, 7, 5, 'a'), (0, 8, 7, 'a'), (7, 8, 8, 'a'), (6, 8, 10, 'a'), (4, 7, 11, 'a'), (3, 8, 12, 'a')
3. Since the current graph definitely doesn't contain "4-clique", I will now search the graph for the "4-clique with its last edge removed" temporal motif.
    * The pattern for a "4-clique with its last edge removed" is a sequence of five edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 9.
    * In this undirected graph, the edges (3, 0, 4, 'a'), (0, 7, 5, 'a'), (0, 8, 7, 'a'), (7, 8, 8, 'a'), (8, 3, 12, 'a') form the above pattern.
    * The timestamps 4, 5, 7, 8, 12 are strictly increasing. 
    * The difference between the maximum and minimum timestamp is 12 - 4 = 8, 8 < 9, which satisfies the time window constraint.
4. Therefore, I will add the last edge (3, 7, 13, 'a') within the time window to make the graph contain a complete "4-clique" temporal motif.
5. This edge is the final answer.""",
                    [(3, 7, 13, 'a')] # answer (will be formatted by make_qa_example)
                )
            ]
        elif self.motif_name == "4-chordalcycle":
            qa_examples_raw = [
                (
                    [(3, 4, 2, 'a'), (6, 8, 3, 'a'), (2, 8, 4, 'a'), (0, 2, 7, 'a'), (0, 2, 8, 'd'), (0, 7, 8, 'a'), (2, 7, 10, 'a'), (1, 5, 13, 'a'), (3, 4, 13, 'd'), (2, 8, 14, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')],
                    [4, 5, 10],
                    ["4-chordalcycle"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "4-chordalcycle" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5, 6, 7, 8
    * Edges: (3, 4, 2, 'a'), (6, 8, 3, 'a'), (2, 8, 4, 'a'), (0, 2, 7, 'a'), (0, 7, 8, 'a'), (2, 7, 10, 'a'), (1, 5, 13, 'a')
3. Since the current graph definitely doesn't contain "4-chordalcycle", I will now search the graph for the "4-chordalcycle with its last edge removed" temporal motif.
    * The pattern for a "4-chordalcycle with its last edge removed" is a sequence of four edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 9.
    * In this undirected graph, the edges (8, 2, 4, 'a'), (2, 0, 7, 'a'), (0, 7, 8, 'a'), (7, 2, 10, 'a') form the above "4-chordalcycle with its last edge removed" pattern.
    * The timestamps 4, 7, 8, 10 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 10 - 4 = 6, 6 < 9, which satisfies the time window constraint.
4. Therefore, I will add the last edge (7, 8, 11, 'a') within the time window to make the graph contain a complete "4-chordalcycle" temporal motif.
5. This edge is the final answer.""",
                    [(7, 8, 11, 'a')] # answer (will be formatted by make_qa_example)
                )
            ]
        elif self.motif_name == "bitriangle":
            qa_examples_raw = [
                (
                    [(3, 4, 2, "a"), (2, 8, 4, "a"), (2, 7, 6, "a"), (2, 7, 7, "d"), (7, 9, 7, "a"), (5, 6, 8, "a"), (6, 9, 8, "a"), (0, 6, 10, "a"), (0, 6, 10, "d"), (2, 8, 14, "d")],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')],
                    [6, 6, 10],
                    ["bitriangle"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "bitriangle" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 2, 3, 4, 5, 6, 7, 8, 9
    * Edges: (3, 4, 2, 'a'), (2, 8, 4, 'a'), (2, 7, 6, 'a'), (7, 9, 7, 'a'), (5, 6, 8, 'a'), (6, 9, 8, 'a'), (0, 6, 10, 'a')
3. Since the current graph definitely doesn't contain "bitriangle", I will now search the graph for the "bitriangle with its last edge removed" temporal motif.
    * The pattern for a "bitriangle with its last edge removed" is a sequence of five edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 9.
    * In this undirected graph, the edges (8, 2, 4, 'a'), (2, 7, 6, 'a'), (7, 9, 7, 'a'), (9, 6, 8, 'a'), (6, 0, 10, 'a') form the above pattern.
    * The timestamps 4, 6, 7, 8, 10 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 10 - 4 = 6, 6 < 9, which satisfies the time window constraint.
4. Therefore, I will add the last edge (0, 8, 11, 'a') within the time window to make the graph contain a complete "bitriangle" temporal motif.
5. This edge is the final answer.""",
                    [(0, 8, 11, 'a')] # answer (will be formatted by make_qa_example)
                )
            ]

        # Convert to the format make_qa_example needs: list[dict, list, str]
        # Need a dictionary containing motif_definition and context as the first element
        qa_formatted = []
        for graph_ctx, motif_def, NMT, motif_name, cot, answer in qa_examples_raw:
             example_qa_dict = {
                 "context": graph_ctx,
                 "query": [motif_def , NMT, motif_name], # query is still empty
                 "cot": cot,
                 "answer" : answer
             }
             qa_formatted.append([example_qa_dict["context"], example_qa_dict["query"], example_qa_dict["cot"], example_qa_dict["answer"]]) # Second list is query (empty), third is answer

        return self.make_qa_example(num, qa_formatted)

    def generate_prompt_question(self, query = None, *args, **kwargs):
        """
        Generates the question containing the Motif definition and graph context. (English comments)
        kwargs should contain 'motif_definition' and 'context'.
        """
        print(f"query: {query}")
        motif_def = query[0]
        N = query[1][0]
        M = query[1][1]
        T = query[1][2]
        motif_name = query[2][0]
        def format_list_for_prompt(data_list):
            return "[" + ", ".join([f"({str(u)}, {str(v)}, {str(t)}, {str(op)})" for u, v, t, op in data_list]) + "]"

        motif_def_str = format_list_for_prompt(motif_def)

        # Use the question template you provided
        return f" Given a {motif_name} temporal motif which is a {N}-node, {M}-edge, {T}-temporal motif with the edges{motif_def_str}. How to modify the given undirected dynamic graph so that it contains the given temporal motif?"

    def evaluate(self, qa, response, w):
        """
        Evaluates whether the generated answer is correct. (English comments)

        Args:
            qa (dict): Dictionary containing the true answer 'answer' ("Yes" or "No").
            response (str): The complete response string generated by the model.

        Returns:
            tuple: (metric, extracted_answer)
                   metric: 1 (correct), 0 (incorrect), -3 (parsing failure)
                   extracted_answer: The parsed "Yes" or "No" (lowercase), or None
        """
        metric = -3 # Default to parsing failure
        response_quads_list = parse_quadruples_from_response_content(response)
        context = qa.get("context")
        original_motif = qa.get("query")[0]
        processed_motif = []
        for item in original_motif:
            if len(item) == 4:
                u, v, t_rel, event_type = item
                # Check if node identifiers need conversion
                if isinstance(u, str) and u.startswith('u'):
                    try:
                        u = int(u[1:])  # Extract 0 from 'u0'
                    except ValueError:
                        pass  # Keep as is
                if isinstance(v, str) and v.startswith('u'):
                    try:
                        v = int(v[1:])  # Extract 1 from 'u1'
                    except ValueError:
                        pass  # Keep as is
                if isinstance(t_rel, str) and t_rel.startswith('t'):
                    try:
                        t_rel = int(t_rel[1:])  # Extract 0 from 't0'
                    except ValueError:
                        pass  # Keep as is
                processed_motif.append((u, v, t_rel, event_type))
            else:
                processed_motif.append(item)  # Keep as is
        print(judge(context, processed_motif, w))
        print(response_quads_list)
        if response_quads_list and judge(context + response_quads_list, processed_motif, w) == "Yes":
            print("Yes")
            metric = 1
        else:
            metric = 0
        return metric, response_quads_list