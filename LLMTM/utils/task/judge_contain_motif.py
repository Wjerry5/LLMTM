from .base import DyGraphTask
import re
import json # For formatting lists in prompts
import networkx as nx
from collections import defaultdict # Ensure import
from ..modify_judge_count import judge

PREDEFINED_MOTIFS = {
    "triangle":     {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')], "T": 4},             # k=3, l=3
    "3-star":       {"edge": [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')], "T": 3},             # k=4, l=3 (center node is 0)
    "4-path":       {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')], "T": 3},             # k=4, l=3
    "4-cycle":      {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')], "T": 6}, # k=4, l=4
    "4-tailedtriangle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')], "T": 7}, # k=4, l=4
    "butterfly":    {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')], "T": 6}, # k=4, l=4 (Same topology as 4-cycle but different temporal order? Ambiguous based on image analysis, temporarily excluded)
    "4-chordalcycle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')], "T": 14}, # k=4, l=5
    "4-clique":     {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')], "T": 27}, # k=4, l=6
    "bitriangle":   {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')], "T": 14}, # k=6, l=6
}


class DyGraphTaskJudgeContainMotif(DyGraphTask):
    """
    Task: Determine whether the given dynamic graph contains the specified temporal motif.
    """
    def __init__(self, task, args):
        super().__init__(task, args)
        self.motif_name = args.motif_name

    def generate_qa(self, info, *args, **kwargs):
        """
        Generate QA pairs. Determine the true answer ("Yes" or "No") based on the generated information.
        """
        context = info.get('edge_index', []) # Dynamic graph edge list to be judged
        original_motif = PREDEFINED_MOTIFS[self.motif_name]["edge"] # Motif definition
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
                        pass  # Keep as is
                if isinstance(t_rel, str) and t_rel.startswith('t'):
                    try:
                        t_rel = int(t_rel[1:])  # Extract 0 from 't0'
                    except ValueError:
                        pass  # Keep as is
                processed_motif.append((u, v, t_rel, event_type))
            else:
                processed_motif.append(item)  # Keep as is
        
        predefined_motif = processed_motif
        # Calculate the number of nodes and edges in predefined_motif
        nodes = set()
        for u, v, _, _ in predefined_motif:
            nodes.add(u)
            nodes.add(v)
        N = len(nodes)  # Number of nodes
        M = len(predefined_motif)  # Number of edges
        T = PREDEFINED_MOTIFS[self.motif_name]["T"]
        answer = judge(context, predefined_motif, T)
        print(f"answer: {answer}")
        qa = {
            "context": context,       # Graph to be judged
            "query": [original_motif, [N, M, T], [self.motif_name]], # For judgment tasks, query is usually embedded in the question template
            "answer": answer,               # True answer "Yes" or "No"
            "task": self.task,
            }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        """Generate task instruction"""
        return "Your task is to determine whether the given undirected dynamic graph contains the given temporal motif? This means that there exists a subgraph within the undirected dynamic graph that matches both the structure and the temporal constraints of the temporal motif."

    def generate_instructor_answer(self, *args, **kwargs):
        """Generate answer format instruction"""

        return "Give the answer as 'Yes' or 'No' at the end of your response after 'Answer:'"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        """Generate Few-shot examples"""
        

        if self.motif_name == "bitriangle":
            qa_examples_raw = [
                (
                    
#                     """**Chain of Thought:**
# 1. Objective: Determine if the provided dynamic graph contains a bitriangle motif.
# 2. Definition of a Bitriangle: A bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14.
# 3. Pattern Search and Identification:
#     * In this undirected graph, the edges (1, 2), (1, 17), (17, 22), (19, 22), (16, 19), (2, 16) form a bitriangle motif.
#     * when (1, 2) is the first edge, the timestamps are strictly increasing.
#     * The difference between the maximum and minimum timestamp is 14, which satisfies the time window constraint.
# 4. Conclusion: The graph contains a bitriangle motif.
# """,
#                     """**Chain of Thought:**
# 1. My goal is to determine if the given dynamic graph contains a bitriangle motif.
# 2. I'll start by identifying the nodes and added edges in the graph.
#     * Nodes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
#     * Edges: (1, 2), (3, 23), (5, 10), (8, 23), (12, 14), (14, 24), (15, 24), (1, 5), (5, 9), (15, 18), (9, 15), (12, 20), (16, 23), (6, 12), (6, 21), (10, 14), (15, 23), (3, 7), (6, 22), (8, 24), (10, 20), (20, 23), (0, 1), (0, 20), (1, 17), (11, 12), (1, 12), (6, 13), (11, 23), (16, 20), (16, 21), (1, 9), (5, 8), (6, 10), (9, 22), (10, 19), (13, 18), (17, 24), (22, 24), (0, 10), (5, 19), (6, 9), (10, 11), (11, 24), (18, 21), (0, 14), (3, 12), (4, 15), (4, 18), (7, 23), (10, 12), (2, 13), (5, 24), (13, 14), (14, 18), (0, 4), (0, 17), (2, 12), (7, 11), (17, 22), (18, 23), (20, 22), (0, 9), (1, 8), (7, 12), (17, 21), (19, 22), (1, 10), (4, 6), (5, 20), (7, 15), (15, 21), (16, 19), (18, 20), (2, 16), (14, 19), (0, 21), (5, 23), (8, 14), (9, 11), (13, 16), (13, 17), (1, 4), (7, 13), (3, 18), (5, 16), (6, 18), (7, 8), (15, 20), (1, 13), (1, 20), (3, 6), (8, 13), (7, 19), (11, 21), (14, 17)
# 3. Next, I'll check if the graph contains a bitriangle motif.
#     * A bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14.
#     * In this undirected graph, the edges (1, 2), (1, 17), (17, 22), (19, 22), (16, 19), (2, 16) form a bitriangle motif.
#     * when (1, 2) is the first edge, the timestamps are strictly increasing.
#     * The difference between the maximum and minimum timestamp is 14, which satisfies the time window constraint.
# 4. Therefore, the graph contains a bitriangle motif. 
#     """,
        [(1, 2, 0, 'a'), (1, 6, 1, 'a'), (2, 3, 1, 'a'), (3, 4, 2, 'a'), (4, 5, 3, 'a'), (5, 6, 4, 'a'), (6, 1, 5, 'a'), (1, 3, 6, 'a'), (3, 5, 6, 'a'), (5, 1, 6, 'a'), (1, 2, 7, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')],
                    [6, 6, 14],
                    ["bitriangle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a bitriangle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5, 6
    * Edges: (1, 2, 0, 'a'), (1, 6, 1, 'a'), (2, 3, 1, 'a'), (3, 4, 2, 'a'), (4, 5, 3, 'a'), (5, 6, 4, 'a'), (6, 1, 5, 'a'), (1, 3, 6, 'a'), (3, 5, 6, 'a'), (5, 1, 6, 'a')
3. Next, I'll check if the graph contains a bitriangle motif.
    * A bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14.
    * In this undirected graph, the edges (1, 2, 0, 'a'), (2, 3, 1, 'a'), (3, 4, 2, 'a'), (4, 5, 3, 'a'), (5, 6, 4, 'a'), (6, 1, 5, 'a') form a bitriangle pattern.
    * The timestamps 0, 1, 2, 3, 4, 5 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 5 - 0 = 5, 5 < 14, which satisfies the time window constraint.
4. Therefore, the graph contains a bitriangle motif. 
    """,
                    ["Yes"]
                )
            ]
        elif self.motif_name == "triangle":
            qa_examples_raw = [
                (
                    [(1, 4, 0, 'a'), (2, 3, 1, 'a'), (4, 2, 2, 'a'), (2, 1, 3, 'a'), (0, 3, 4, 'a'), (0, 3, 5, 'd')], 
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')],
                    [3, 3, 4],
                    ["triangle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a triangle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4
    * Edges: (1, 4, 0, 'a'), (2, 3, 1, 'a'), (4, 2, 2, 'a'), (2, 1, 3, 'a'), (0, 3, 4, 'a')
3. Next, I'll check if the graph contains a triangle motif.
    * A triangle motif is a sequence of three edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 4.
    * In this undirected graph, the edges (1, 4, 0, 'a'), (4, 2, 2, 'a'), (2, 1, 3, 'a') form a triangle pattern.
    * The timestamps 0, 2, 3 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 3 - 0 = 3, 3 < 4, which satisfies the time window constraint.
4. Therefore, the graph contains a triangle motif. 
    """,
                    ["Yes"]
                )
            ]
        elif self.motif_name == "3-star":
            qa_examples_raw = [
                (
                    [(0, 1, 0, 'a'), (1, 3, 1, 'a'), (0, 3, 1, 'a'), (0, 4, 2, 'a'), (2, 3, 3, 'a'), (2, 4, 4, 'a'), (0, 3, 5, 'd')], 
                    [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')],
                    [4, 3, 3],
                    ["3-star"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a 3-star motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4
    * Edges: (0, 1, 0, 'a'), (1, 3, 1, 'a'), (0, 3, 1, 'a'), (0, 4, 2, 'a'), (2, 3, 3, 'a'), (2, 4, 4, 'a')
3. Next, I'll check if the graph contains a 3-star motif.
    * A 3-star motif is star structure where all edges are connected to a central node, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3.
    * In this undirected graph, the edges (0, 1, 0, 'a'), (0, 3, 1, 'a'), (0, 4, 2, 'a') form a 3-star pattern.
    * The timestamps 0, 1, 2 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 2 - 0 = 2, 2 < 3, which satisfies the time window constraint.
4. Therefore, the graph contains a 3-star motif. 
    """,
                    ["Yes"]
                )
            ]
        elif self.motif_name == "4-path":
            qa_examples_raw = [
                (
                    [(0, 4, 0, 'a'), (1, 2, 1, 'a'), (1, 3, 1, 'a'), (2, 3, 2, 'a'), (3, 4, 3, 'a'), (1, 2, 4, 'd'), (5, 6, 5, 'a'), (5, 6, 6, 'd')], # context ( make_qa_example )
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')],
                    [4, 3, 3],
                    ["4-path"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a 4-path motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5, 6
    * Edges: (0, 4, 0, 'a'), (1, 2, 1, 'a'), (1, 3, 1, 'a'), (2, 3, 2, 'a'), (3, 4, 3, 'a'), (5, 6, 5, 'a')
3. Next, I'll check if the graph contains a 4-path motif.
    * A 4-path motif is a sequence of three edges forming a path, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3.
    * In this undirected graph, the edges (1, 2, 1, 'a'), (2, 3, 2, 'a'), (3, 4, 3, 'a') form a 4-path pattern.
    * The timestamps 1, 2, 3 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 3 - 1 = 2, 2 < 3, which satisfies the time window constraint.
4. Therefore, the graph contains a 4-path motif. 
    """,
                    ["Yes"]
                )
            ]
        elif self.motif_name == "4-cycle":
            qa_examples_raw = [
                (
                    [(0, 1, 0, 'a'), (0, 3, 2, 'a'), (1, 2, 3, 'a'), (1, 3, 3, 'a'), (2, 3, 4, 'a'), (3, 0, 5, 'a'), (3, 4, 6, 'a'), (0, 1, 6, 'd')], # context ( make_qa_example )
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')],
                    [4, 4, 6],
                    ["4-cycle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a 4-cycle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3
    * Edges: (0, 1, 0, 'a'), (0, 3, 2, 'a'), (1, 2, 3, 'a'), (1, 3, 3, 'a'), (2, 3, 4, 'a'), (3, 0, 5, 'a'), (3, 4, 6, 'a')
3. Next, I'll check if the graph contains a 4-cycle motif.
    * A 4-cycle motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6.
    * In this undirected graph, the edges (0, 1, 0, 'a'), (1, 2, 3, 'a'), (2, 3, 4, 'a'), (3, 0, 5, 'a') form a 4-cycle pattern.
    * The timestamps 0, 3, 4, 5 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 5 - 0 = 5, 5 < 6, which satisfies the time window constraint.
4. Therefore, the graph contains a 4-cycle motif. 
    """,
                    ["Yes"]
                )
            ]
        elif self.motif_name == "4-tailedtriangle":
            qa_examples_raw = [
                (
                    [(0, 1, 0, 'a'), (0, 2, 1, 'a'), (1, 2, 2, 'a'), (2, 3, 3, 'a'), (3, 1, 4, 'a'), (3, 0, 5, 'a'), (0, 2, 6, 'd'), (4, 5, 7, 'a'), (4, 6, 8, 'a'), (4, 5, 9, 'd')], # context ( make_qa_example )
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')],
                    [4, 4, 7],
                    ["4-tailedtriangle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a 4-tailedtriangle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5, 6
    * Edges: (0, 1, 0, 'a'), (0, 2, 1, 'a'), (1, 2, 2, 'a'), (2, 3, 3, 'a'), (3, 1, 4, 'a'), (3, 0, 5, 'a'), (4, 5, 7, 'a'), (4, 6, 8, 'a')
3. Next, I'll check if the graph contains a 4-tailedtriangle motif.
    * A 4-tailedtriangle motif is a sequence of four edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u1'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 7.
    * In this undirected graph, the edges (0, 1, 0, 'a'), (1, 2, 2, 'a'), (2, 3, 3, 'a'), (3, 1, 4, 'a') form a 4-tailedtriangle pattern.
    * The timestamps 0, 2, 3, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 7, which satisfies the time window constraint.
4. Therefore, the graph contains a 4-tailedtriangle motif. 
    """,
                    ["Yes"] 
                )
            ]

        elif self.motif_name == "4-chordalcycle":
            qa_examples_raw = [
                (
                    [(0, 1, 0, 'a'), (1, 2, 1, 'a'), (1, 4, 2, 'a'), (2, 3, 2, 'a'), (3, 1, 3, 'a'), (3, 0, 4, 'a'), (0, 2, 5, 'a'), (4, 5, 7, 'a'), (4, 6, 8, 'a'), (4, 5, 9, 'd')], # context ( make_qa_example )
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')],
                    [4, 5, 14],
                    ["4-chordalcycle"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a 4-chordalcycle motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5, 6
    * Edges: (0, 1, 0, 'a'), (1, 2, 1, 'a'), (1, 4, 2, 'a'), (2, 3, 2, 'a'), (3, 1, 3, 'a'), (3, 0, 4, 'a'), (0, 2, 5, 'a'), (4, 5, 7, 'a'), (4, 6, 8, 'a')
3. Next, I'll check if the graph contains a 4-chordalcycle motif.
    * A 4-chordalcycle motif is a sequence of five edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3' ), ('u3', 'u1'), ('u3', 'u0'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14.
    * In this undirected graph, the edges (0, 1, 0, 'a'), (1, 2, 1, 'a'), (2, 3, 2, 'a'), (3, 1, 3, 'a'), (3, 0, 4, 'a') form a 4-chordalcycle pattern.
    * The timestamps 0, 1, 2, 3, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 14, which satisfies the time window constraint.
4. Therefore, the graph contains a 4-chordalcycle motif. 
    """,
                    ["Yes"] 
                )
            ]
        elif self.motif_name == "4-clique":
            qa_examples_raw = [
                (
                    [(0, 1, 0, 'a'), (1, 2, 1, 'a'), (1, 6, 2, 'a'), (1, 3, 3, 'a'), (2, 3, 4, 'a'), (3, 0, 5, 'a'), (0, 2, 6, 'a'), (0, 2, 7, 'd'), (4, 5, 8, 'a'), (4, 5, 10, 'd')], # context ( make_qa_example )           
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')],
                    [4, 6, 27],
                    ["4-clique"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a 4-clique motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5, 6
    * Edges: (0, 1, 0, 'a'), (1, 2, 1, 'a'), (1, 6, 2, 'a'), (1, 3, 3, 'a'), (2, 3, 4, 'a'), (3, 0, 5, 'a'), (0, 2, 6, 'a')
3. Next, I'll check if the graph contains a 4-clique motif.
    * A 4-clique motif is a sequence of six edges ('u0', 'u1'), ('u1', 'u2'), ('u1', 'u3'), ('u2', 'u3'), ('u3', 'u0'), ('u0', 'u2'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 27.
    * In this undirected graph, the edges (0, 1, 0, 'a'), (1, 2, 1, 'a'), (1, 3, 3, 'a'), (2, 3, 4, 'a'), (3, 0, 5, 'a'), (0, 2, 6, 'a') form a 4-clique pattern.
    * The timestamps 0, 1, 3, 4, 5, 6 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 6 - 0 = 6, 6 < 27, which satisfies the time window constraint.
4. Therefore, the graph contains a 4-clique motif. 
    """,
                    ["Yes"]
                )
            ]
        elif self.motif_name == "butterfly":
            qa_examples_raw = [
                (
                    [(0, 1, 0, 'a'), (0, 4, 1, 'a'), (1, 3, 2, 'a'), (3, 2, 3, 'a'), (2, 0, 4, 'a'), (0, 2, 5, 'd'), (0, 1, 6, 'd'), (4, 5, 7, 'a')], # context ( make_qa_example )           
                     [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')],
                    [4, 4, 6],
                    ["butterfly"],
                    """**Chain of Thought:**
1. My goal is to determine if the given dynamic graph contains a butterfly motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5
    * Edges: (0, 1, 0, 'a'), (0, 4, 1, 'a'), (1, 3, 2, 'a'), (3, 2, 3, 'a'), (2, 0, 4, 'a'), (0, 2, 5, 'd'), (0, 1, 6, 'd'), (4, 5, 7, 'a')
3. Next, I'll check if the graph contains a butterfly motif.
    * A butterfly motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6.
    * In this undirected graph, the edges (0, 1, 0, 'a'), (1, 3, 2, 'a'), (3, 2, 3, 'a'), (2, 0, 4, 'a') form a butterfly pattern.
    * The timestamps 0, 2, 3, 4 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 4 - 0 = 4, 4 < 6, which satisfies the time window constraint.
4. Therefore, the graph contains a butterfly motif. 
    """,
                    ["Yes"]
                )
            ]
            

        # Convert to the format required by make_qa_example: list[dict, list, str]
        # Need a dictionary containing motif_definition and context as the first element
        qa_formatted = []
        for graph_ctx, motif_def, NMT, motif_name, cot, answer in qa_examples_raw:
             example_qa_dict = {

                 "context": graph_ctx,
                 "query": [motif_def , NMT, motif_name], # query is still empty
                 "cot": cot,
                 "answer" : answer
             }
             qa_formatted.append([example_qa_dict["context"], example_qa_dict["query"], example_qa_dict["cot"],example_qa_dict["answer"]]) # The second list is query (empty), the third is answer

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

        return f"Given a {motif_name} temporal motif which is a {N}-node, {M}-edge, {T}-temporal motif with the edges{motif_def_str}. Whether the given undirected dynamic graph contains the given temporal motif?\nGive me the final answer directly. Do not show your reasoning or steps."

    def evaluate(self, qa, response):
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

        answer_part = response[0:].strip()
        match = re.search(r'\b(yes|no)\b', answer_part, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).lower()
            if extracted_answer == ground_truth:
                metric = 1 
            else:
                metric = 0 
        
        return metric, extracted_answer
        
        