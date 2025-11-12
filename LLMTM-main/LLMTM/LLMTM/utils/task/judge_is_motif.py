from .base import DyGraphTask
import re
import json 
import networkx as nx
from collections import defaultdict

# --- New helper function for checking temporal constraints ---
def check_temporal_constraints(mapping, sorted_motif_events, context_times_all, motif_time_window):
    """
    Check if there exists a timestamp sequence that satisfies temporal and window constraints for a given isomorphic mapping.
    (Using iterative method to find valid sequences)
    """
    possible_sequences = [[]] 

    for k, (u_m, v_m, t_rel) in enumerate(sorted_motif_events):
        u_c = mapping.get(u_m)
        v_c = mapping.get(v_m)
        if u_c is None or v_c is None: return False

        context_edge = tuple(sorted((u_c, v_c)))
        candidate_times = context_times_all.get(context_edge, [])
        if not candidate_times: return False

        next_possible_sequences = []
        for sequence in possible_sequences:
            last_t = sequence[-1] if sequence else -1
            t_start = sequence[0] if sequence else None

            for t_c in candidate_times:
                if t_c > last_t:  # Check order
                    if t_start is None:  # First event
                        next_possible_sequences.append(sequence + [t_c])
                    else:  # Subsequent events, check time window
                        if motif_time_window <= 0 or (t_c - t_start) < motif_time_window:
                            next_possible_sequences.append(sequence + [t_c])

        possible_sequences = next_possible_sequences
        if not possible_sequences:
            return False  # No valid sequence can be extended

    return True # Successfully built at least one complete sequence

# --- Replace old judge function ---
def judge(context, motif_definition, motif_time_window):
    """
    Precisely determine if there exists an instance of motif_definition in the context graph,
    satisfying topology, temporal order, and time window constraints (considering all timestamps).
    """
    context_a = [e for e in context if len(e) == 4 and e[3] == 'a']
    motif_a = [e for e in motif_definition if len(e) == 4 and e[3] == 'a']

    if not motif_a: return "No"

    # 1. Parse Motif
    G_motif_template = nx.Graph()
    motif_nodes_orig = set()
    sorted_motif_events = []
    try:
        temp_motif_events = []
        for u, v, t_rel, op in motif_a:
            u_node, v_node = int(u), int(v)
            motif_nodes_orig.add(u_node)
            motif_nodes_orig.add(v_node)
            G_motif_template.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t_rel))
        sorted_motif_events = sorted(temp_motif_events, key=lambda x: x[2])
    except Exception as e:
        print(f"Error (judge): Failed to parse motif_definition: {e}")
        return "Error"

    # 2. Parse Context (store all timestamps)
    G_context = nx.Graph()
    context_times_all = defaultdict(list)
    context_nodes = set()
    if not context_a: return "No"
    try:
        for u, v, t, op in context_a:
            u_node, v_node = int(u), int(v)
            context_nodes.add(u_node)
            context_nodes.add(v_node)
            edge = tuple(sorted((u_node, v_node)))
            G_context.add_edge(u_node, v_node)
            context_times_all[edge].append(int(t))
        for edge in context_times_all:
            context_times_all[edge].sort()
    except Exception as e:
         print(f"Error (judge): Failed to parse context: {e}")
         return "Error"

    # 3. Check Isomorphism
    if G_context.number_of_nodes() < G_motif_template.number_of_nodes() or \
       G_context.number_of_edges() < G_motif_template.number_of_edges():
        return "No"

    matcher = nx.isomorphism.GraphMatcher(G_context, G_motif_template)

    # 4. Iterate through each isomorphism mapping to check temporal constraints
    for mapping in matcher.subgraph_isomorphisms_iter():
        if set(mapping.keys()) != set(G_motif_template.nodes()):
             continue
        # Call new helper function for precise temporal check
        reversed_mapping = {v: k for k, v in mapping.items()}
        if check_temporal_constraints(reversed_mapping, sorted_motif_events, context_times_all, motif_time_window):
            return "Yes" 

    return "No"

class DyGraphTaskJudgeMotif(DyGraphTask):
    """
    Task: Determine whether the given dynamic graph is the specified temporal motif.
    """
    def __init__(self, task, args):
        super().__init__(task, args)
        self.motif_name = args.motif_name

    def generate_qa(self, info, *args, **kwargs):
        """
        Generate QA pairs. Determine the true answer ("Yes" or "No") based on the generated information.
        """
        context = info.get('edge_index', []) # Dynamic graph edge list to be judged
        original_motif = info.get('original_motif', []) # Motif definition
        # Ensure context and predefined_motif are correctly typed lists
        context = [(int(u), int(v), int(t), str(op)) for u, v, t, op in context]
        original_motif = [(str(u), str(v), str(t), str(op)) for u, v, t, op in original_motif]
        
        processed_motif = []
        for item in original_motif:
            if len(item) == 4:
                u, v, t_rel, event_type = item
                # Check if node identifiers need to be converted
                if isinstance(u, str) and u.startswith('u'):
                    try:
                        u = int(u[1:])  # Extract 0 from 'u0'
                    except ValueError:
                        pass  # Keep as is
                if isinstance(v, str) and v.startswith('u'):
                    try:
                        v = int(v[1:])  # Extract 1 from 'u1'
                    except ValueError:
                        pass 
                if isinstance(t_rel, str) and t_rel.startswith('t'):
                    try:
                        t_rel = int(t_rel[1:])  # Extract 0 from 't0'
                    except ValueError:
                        pass 
                processed_motif.append((u, v, t_rel, event_type))
            else:
                processed_motif.append(item)  
        
        predefined_motif = processed_motif
        # Calculate the number of nodes and edges of predefined_motif
        nodes = set()
        for u, v, _, _ in predefined_motif:
            nodes.add(u)
            nodes.add(v)
        N = len(nodes)  # Number of nodes
        M = len(predefined_motif)  # Number of edges
        T = info.get('motif_time_window', 0)
        answer = judge(context, predefined_motif, T)

        qa = {
            "context": context,       # Graph to be judged
            "query": [original_motif, [N, M, T], [self.motif_name]], # For judgment tasks, query is usually embedded in the question template
            "answer": answer,               # True answer "Yes" or "No"
            "task": self.task,
            }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        """Generate task instruction"""
        return "Your task is to answer whether the given undirected dynamic graph is the given temporal motif?"

    def generate_instructor_answer(self, *args, **kwargs):
        """Generate answer format instruction"""

        return "Give the answer as 'Yes' or 'No' at the end of your response after 'Answer:'"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        """Generate Few-shot examples"""
        

        if self.motif_name == "bitriangle":
            qa_examples_raw = [
                (
                    [(0, 1, 2, 'a'), (1, 3, 3, 'a'), (3, 5, 7, 'a'), (1, 3, 8, 'd'), (4, 5, 9, 'a'), (2, 4, 10, 'a'), (0, 2, 11, 'a'), (0, 2, 15, 'd'), (2, 4, 17, 'd'), (0, 1, 18, 'd'), (4, 5, 19, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')],
                    [6, 6, 10],
                    ["bitriangle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a bitriangle motif.
2. I'll start by identifying the nodes and and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5
    * Edges: (0, 1, 2, 'a'), (1, 3, 3, 'a'), (3, 5, 7, 'a'), (4, 5, 9, 'a'), (2, 4, 10, 'a'), (0, 2, 11, 'a')
3. Next, I'll check if the graph is a bitriangle pattern.
    * A bitriangle pattern is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 10.
    * In this undirected graph, the edges (0, 1, 2, 'a'), (1, 3, 3, 'a'), (3, 5, 7, 'a'), (5, 4, 9, 'a'), (4, 2, 10, 'a'), (2, 0, 11, 'a') form a closed cycle.
    * The timestamps 2, 3, 7, 9, 10, 11 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 11 - 2 = 9, 9 < 10, which satisfies the time window constraint.
4. Therefore, the graph is a bitriangle motif.""", #cot
                    ["Yes"]
                )
            ]
        elif self.motif_name == "triangle":
            qa_examples_raw = [
                (
                    [(0, 2, 0, 'a'), (0, 1, 1, 'a'), (0, 1, 2, 'd'), (1, 2, 4, 'a'), (0, 2, 6, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')],
                    [3, 3, 5],
                    ["triangle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a triangle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2
    * Edges: (0, 2, 0, 'a'), (0, 1, 1, 'a'), (1, 2, 4, 'a')
3. Next, I'll check if the graph is a triangle pattern.
    * A triangle pattern is a sequence of three edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 5.
    * In this undirected graph, the edges (2, 0, 0, 'a'), (0, 1, 1, 'a'), (1, 2, 4, 'a') form a closed cycle.
    * The timestamps 0, 1, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 5, which satisfies the time window constraint.
4. Therefore, the graph is a triangle motif.""", #cot
                    ["Yes"] 
                )
            ]
        elif self.motif_name == "3-star":
            qa_examples_raw = [
                (
                    [(0, 2, 0, 'a'), (0, 3, 1, 'a'), (0, 3, 2, 'd'), (0, 1, 4, 'a')],
                    [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')],
                    [4, 3, 5],
                    ["3-star"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a 3-star motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3
    * Edges: (0, 2, 0, 'a'), (0, 3, 1, 'a'), (0, 1, 4, 'a')
3. Next, I'll check if the graph is a 3-star pattern.
    * A 3-star pattern is star structure where all edges are connected to a central node, and the timestamps are strictly increasing, the difference between the maximum and minimum timestamp is at most 5.
    * In this undirected graph, the edges (0, 2, 0, 'a'), (0, 3, 1, 'a'), (0, 1, 4, 'a') form a star structure.
    * The timestamps 0, 1, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 5, which satisfies the time window constraint.
4. Therefore, the graph is a 3-star motif.""", #cot
                    ["Yes"]
                )
            ]
        elif self.motif_name == "4-path":
            qa_examples_raw = [
                (
                    [(0, 1, 0, 'a'), (1, 2, 2, 'a'), (2, 3, 4, 'a'), (2, 3, 6, 'd'), (0, 1, 8, 'd'), (1, 2, 8, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')],
                    [4, 3, 5],
                    ["4-path"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a 4-path motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3
    * Edges: (0, 1, 0, 'a'), (1, 2, 2, 'a'), (2, 3, 4, 'a')
3. Next, I'll check if the graph is a 4-path pattern.
    * A 4-path pattern is a sequence of three edges forming a path, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 5.
    * In this undirected graph, the edges (0, 1, 0, 'a'), (1, 2, 2, 'a'), (2, 3, 4, 'a') form a path.   
    * The timestamps 0, 2, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 5, which satisfies the time window constraint.
4. Therefore, the graph is a 4-path motif.""", #cot
                    ["Yes"] 
                )
            ]
        elif self.motif_name == "4-cycle":
            qa_examples_raw = [
                (
                    [(1, 2, 0, 'a'), (0, 1, 2, 'a'), (0, 3, 3, 'a'), (2, 3, 4, 'a'), (1, 2, 6, 'd'), (2, 3, 7, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')],
                    [4, 4, 5],
                    ["4-cycle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a 4-cycle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3
    * Edges: (1, 2, 0, 'a'), (0, 1, 2, 'a'), (0, 3, 3, 'a'), (2, 3, 4, 'a')
3. Next, I'll check if the graph is a 4-cycle pattern.
    * A 4-cycle pattern is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 5.
    * In this undirected graph, the edges (2, 1, 0, 'a'), (1, 0, 2, 'a'), (0, 3, 3, 'a'), (3, 2, 4, 'a') form a closed cycle.   
    * The timestamps 0, 2, 3, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 5, which satisfies the time window constraint.
4. Therefore, the graph is a 4-cycle motif.""", #cot
                    ["Yes"] 
                )
            ]
        elif self.motif_name == "4-tailedtriangle":
            qa_examples_raw = [
                (
                    [(0, 1, 0, 'a'), (1, 3, 1, 'a'), (2, 3, 2, 'a'), (1, 2, 4, 'a'), (1, 2, 5, 'd'), (0, 1, 9, 'd'), (2, 3, 9, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')],
                    [4, 4, 5],
                    ["4-tailedtriangle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a 4-tailedtriangle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3
    * Edges: (0, 1, 0, 'a'), (1, 3, 1, 'a'), (2, 3, 2, 'a'), (1, 2, 4, 'a')
3. Next, I'll check if the graph is a 4-tailedtriangle pattern.
    * A 4-tailedtriangle pattern is a sequence of four edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u1'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 5.
    * In this undirected graph, the edges (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 2, 'a'), (2, 1, 4, 'a') form a 4-tailedtriangle pattern.   
    * The timestamps 0, 1, 2, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 5, which satisfies the time window constraint.
4. Therefore, the graph is a 4-tailedtriangle motif.""", #cot
                    ["Yes"] 
                )
            ]
        elif self.motif_name == "butterfly":
            qa_examples_raw = [
                (
                    [(0, 2, 0, 'a'), (0, 1, 1, 'a'), (1, 3, 3, 'a'), (2, 3, 4, 'a'), (2, 3, 5, 'd'), (0, 2, 6, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')],
                    [4, 4, 5],
                    ["butterfly"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a butterfly motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3
    * Edges: (0, 2, 0, 'a'), (0, 1, 1, 'a'), (1, 3, 3, 'a'), (2, 3, 4, 'a')
3. Next, I'll check if the graph is a butterfly pattern.
    * A butterfly pattern is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 5.
    * In this undirected graph, the edges (2, 0, 0, 'a'), (0, 1, 1, 'a'), (1, 3, 3, 'a'), (3, 2, 4, 'a') form a closed cycle.
    * The timestamps 0, 1, 3, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 5, which satisfies the time window constraint.
4. Therefore, the graph is a butterfly motif.""", #cot
                    ["Yes"] 
                )
            ]
        elif self.motif_name == "4-chordalcycle":
            qa_examples_raw = [
                (
                    [(0, 3, 7, 'a'), (2, 3, 9, 'a'), (1, 2, 11, 'a'), (1, 2, 12, 'd'), (1, 3, 13, 'a'), (2, 3, 13, 'd'), (0, 1, 15, 'a'), (0, 1, 17, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')],
                    [4, 5, 10],
                    ["4-chordalcycle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a 4-chordalcycle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3
    * Edges: (0, 3, 7, 'a'), (2, 3, 9, 'a'), (1, 2, 11, 'a'), (1, 3, 13, 'a'), (0, 1, 15, 'a')
3. Next, I'll check if the graph is a 4-chordalcycle pattern.
    * A 4-chordalcycle pattern is a sequence of five edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3' ), ('u3', 'u1'), ('u3', 'u0'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 10.
    * In this undirected graph, the edges (0, 3, 7, 'a'), (3, 2, 9, 'a'), (2, 1, 11, 'a'), (1, 3, 13, 'a'), (1, 0, 15, 'a') form a 4-chordalcycle pattern.
    * The timestamps 7, 9, 11, 13, 15 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 15 - 7 = 8, 8 < 10, which satisfies the time window constraint.
4. Therefore, the graph is a 4-chordalcycle motif.""", #cot
                    ["Yes"] 
                )
            ]
        elif self.motif_name == "4-clique":
            qa_examples_raw = [
                (
                    [(0, 1, 1, 'a'), (1, 2, 2, 'a'), (1, 3, 3, 'a'), (2, 3, 7, 'a'), (0, 3, 8, 'a'), (0, 2, 9, 'a'), (1, 3, 11, 'd'), (0, 3, 14, 'd'), (0, 1, 16, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')],
                    [4, 6, 10],
                    ["4-clique"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph is a 4-clique motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3
    * Edges: (0, 1, 1, 'a'), (1, 2, 2, 'a'), (1, 3, 3, 'a'), (2, 3, 7, 'a'), (0, 3, 8, 'a'), (0, 2, 9, 'a')
3. Next, I'll check if the graph is a 4-clique pattern.
    * A 4-clique pattern is a sequence of six edges ('u0', 'u1'), ('u1', 'u2'), ('u1', 'u3'), ('u2', 'u3'), ('u3', 'u0'), ('u0', 'u2'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 10.
    * In this undirected graph, the edges (0, 1, 1, 'a'), (1, 2, 2, 'a'), (1, 3, 3, 'a'), (2, 3, 7, 'a'), (3, 0, 8, 'a'), (0, 2, 9, 'a') form a 4-clique pattern.
    * The timestamps 1, 2, 3, 7, 8, 9 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 9 - 1 = 8, 8 < 10, which satisfies the time window constraint.
4. Therefore, the graph is a 4-clique motif.""", #cot
                    ["Yes"] 
                )
            ]
            

        # Convert to the format required by make_qa_example
        qa_formatted = []
        for item in qa_examples_raw:
            try:
                if len(item) == 6:
                    graph_ctx, motif_def, NMT, motif_name, s, answer = item
                    qa_formatted.append([graph_ctx, [motif_def, NMT, motif_name], s, answer])
                else:
                    print(f"Warning: Example format incorrect, expected 6 elements, actually has {len(item)}")
            except Exception as e:
                print(f"Warning: Error processing example: {e}")
                continue

        return self.make_qa_example(num, qa_formatted)

    def generate_prompt_question(self, query = None, *args, **kwargs):
        """
        Generate a question containing motif definition and graph context.
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

        return f"Given a {motif_name} temporal motif which is a {N}-node, {M}-edge, {T}-temporal motif with the edges{motif_def_str}. Whether the given undirected dynamic graph is the given motif?"

    def evaluate(self, qa, response, use_agent):
        """
        Evaluate whether the model response is "Yes" or "No" and compare with the true answer.

        Args:
            qa (dict): Dictionary containing the true answer 'answer' ("Yes" or "No").
            response (str): Complete response string generated by the model.

        Returns:
            tuple: (metric, extracted_answer)
                   metric: 1 (correct), 0 (incorrect), -3 (parsing failed)
                   extracted_answer: Parsed "Yes" or "No" (lowercase), or None
        """
        ground_truth = qa.get("answer", "").strip().lower() # Get true answer and convert to lowercase
        extracted_answer = None
        metric = -3 # Default to parsing failed
        if use_agent:
            raw = str(response).strip()
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    for k in ("result", "answer", "final_answer", "final", "output"):
                        if k in data:
                            raw = str(data[k])
                            break
            except Exception:
                pass
            extracted_answer = raw.strip().lower()
            metric = 1 if extracted_answer == ground_truth else 0
            return metric, extracted_answer
        # Find the content after the last "Answer:" marker
        answer_marker = "Answer:"
        answer_start_index = response.rfind(answer_marker)

        if answer_start_index != -1:
            answer_part = response[answer_start_index + len(answer_marker):].strip()
            # Extract "Yes" or "No", ignoring case and possible punctuation
            # Use more flexible regex to find "yes" or "no" (case insensitive)
            match = re.search(r'\b(yes|no)\b', answer_part, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).lower()
                # print(f"  Evaluation: GT='{ground_truth}', Extracted='{extracted_answer}'") # Debug
                if extracted_answer == ground_truth:
                    metric = 1 # Correct match
                else:
                    metric = 0 # Wrong answer
            else:
                print(f"  Evaluation: Parsing failed - Could not find 'Yes' or 'No' in '{answer_part}'")
                metric = -3 # Parsing failed
        else:
            print("  Evaluation: Parsing failed - Could not find 'Answer:' marker in response")
            metric = -3 # Parsing failed

        return metric, extracted_answer
        
        