from .base import DyGraphTask
import re
import json  # For formatting lists in prompts
import networkx as nx
from collections import defaultdict  # Ensure import
from ..modify_judge_count import count, judge


class DyGraphTaskMultiMotifCount(DyGraphTask):
    """
    Task: Determine whether the given dynamic graph is the specified temporal motif.
    """
    def __init__(self, task, args):
        super().__init__(task, args)
        self.motif_name = args.motif_name

    def generate_qa(self, info, *args, **kwargs):
        """
        Generate QA pairs. Determine the true answer based on the generated information.
        """
        context = info.get('edge_index', []) # Dynamic graph edge list to be judged
        PREDEFINED_MOTIFS = {
    "triangle":     {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')], "T": 3},             # k=3, l=3
    "3-star":       {"edge": [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')], "T": 3},             # k=4, l=3 (center node is 0)
    "4-path":       {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')], "T": 3},             # k=4, l=3
    "4-cycle":      {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')], "T": 6}, # k=4, l=4
    "4-tailedtriangle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')], "T": 6}, # k=4, l=4
    "butterfly":    {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')], "T": 6}, # k=4, l=4
    "4-chordalcycle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')], "T": 14}, # k=4, l=5
    "4-clique":     {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')], "T": 15}, # k=4, l=6
    "bitriangle":   {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')], "T": 15}, # k=6, l=6
}
        answer = []
        context = [(int(u), int(v), int(t), str(op)) for u, v, t, op in context]
        for motif_name, motif_definition in PREDEFINED_MOTIFS.items():
            original_motif = [(str(u), str(v), str(t), str(op)) for u, v, t, op in motif_definition["edge"]]
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
            if judge(context, predefined_motif, motif_definition["T"]) == "Yes":
                answer.append([motif_name, count(context, predefined_motif, motif_definition["T"])])
        
        print(answer)

        qa = {
            "context": context,       # Graph to be judged
            "query": None, # For judgment tasks, query is usually embedded in the question template
            "answer": answer,               # [["triangle", 2], ["3-star", 3], ["4-path", 4]]
            "task": self.task,
            }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        """Generate task instruction"""
        return "Your task is to answer How many times does each of the above temporal motifs appear in the given undirected dynamic graph? Omit temporal motifs that do not appear in the undirected dynamic graph."

    def generate_instructor_answer(self, *args, **kwargs):
        """Generate answer format instruction"""

        return "Give the answer as a list of tuples consisting of the name of a temporal motif and an integer at the end of your response after 'Answer:'"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        """Generate Few-shot examples"""
        qa_examples_raw = [
            (
                [(0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 2, 4, 'a'), (3, 4, 4, 'a'), (2, 4, 4, 'a'), (2, 1, 5, 'a'), (4, 1, 5, 'a'), (2, 0, 6, 'a'), (4, 5, 7, 'a'), (0, 1, 8, 'd'), (5, 0, 9, 'a')],
                [],
                """**Chain of Thought:**  
1. My goal is to determine the number of all temporal motifs present in the given undirected dynamic graph.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5
    * Edges: (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 2, 4, 'a'), (3, 4, 4, 'a'), (2, 4, 4, 'a'), (2, 1, 5, 'a'), (4, 1, 5, 'a'), (2, 0, 6, 'a'), (4, 5, 7, 'a'), (5, 0, 9, 'a')
3. Next, I'll iterate through all the possible existing motifs mentioned in the question. For each motif, Initialize count to 0. Regarding the motif matching, we'll first consider the pattern matching, then check the temporal strictly increasing, finally check the time window constraint. If the motif is found, I'll increment the count by 1.
    * triangle motif is a sequence of three edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. I found (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a') are triangle motifs. Hence, count = 1.
    * 3-star motif is a star structure where all edges are connected to a central node, and the timestamps are strictly increasing, the difference between the maximum and minimum timestamp is at most 3. I found (1, 3, 1, 'a'), (3, 0, 2, 'a'),  (3, 2, 3, 'a') and (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 4, 4, 'a') and (3, 2, 4, 'a'), (2, 1, 5, 'a'), (2, 0, 6, 'a') and (3, 4, 4, 'a'), (4, 1, 5, 'a'), (4, 5, 7, 'a') and (2, 4, 4, 'a'), (4, 1, 5, 'a'), (4, 5, 7, 'a') are 3-star motifs. Hence, count = 5.
    * 4-path motif is a sequence of three edges forming a path, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. I found(0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 4, 'a') and (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 4, 4, 'a') are 4-path motifs. Hence, count = 2.
    * 4-cycle motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. I found (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 4, 'a'), (2, 0, 6, 'a') is 4-cycle motifs. Hence, count = 1.
    * 4-tailedtriangle motif is a sequence of four edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u1'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. I found (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 4, 'a'), (2, 1, 5, 'a') is 4-tailedtriangle motifs. Hence, count = 1.
    * butterfly motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. I found (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 4, 'a'), (2, 0, 6, 'a') is butterfly motifs. Hence, count = 1.
    * 4-chordalcycle motif is a sequence of five edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3' ), ('u3', 'u1'), ('u3', 'u0'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14. I found (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 4, 'a'), (2, 1, 5, 'a'), (2, 0, 6, 'a') is 4-chordalcycle motifs. Hence, count = 1.
    * 4-clique motif is a sequence of six edges ('u0', 'u1'), ('u1', 'u2'), ('u1', 'u3'), ('u2', 'u3'), ('u3', 'u0'), ('u0', 'u2'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. No motif is found.
    * bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. I found (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 4, 'a'), (2, 4, 4, 'a'), (4, 5, 7, 'a'), (5, 0, 9, 'a') is bitriangle motifs. Hence, count = 1.
4. Therefore, the number of all temporal motifs present are: (triangle, 2), (3-star, 2), (4-path, 2), (4-cycle, 1), (4-tailedtriangle, 1), (butterfly, 1), (4-chordalcycle, 1), (bitriangle, 1).""",
                [["triangle", 1], ["3-star", 5], ["4-path", 2], ["4-cycle", 1], ["4-tailedtriangle", 1], ["butterfly", 1], ["4-chordalcycle", 1], ["bitriangle", 1]] 
            )
        ]
        return self.make_qa_example(num, qa_examples_raw)

    def generate_prompt_question(self, query = None, *args, **kwargs):
        """
        Generate a question containing Motif definition and graph context.
        kwargs should contain 'motif_definition' and 'context'.
        """

        return f" How many times does each of the above temporal motifs appear in the given undirected dynamic graph?"

    def evaluate(self, qa, response, use_agent):
        """
        Evaluate whether the generated answer is correct.
        Returns:
            metric: 1 for completely correct, 0 for incorrect, -3 for parsing failure
            extracted_answer: the answer extracted from the response
        """
        ground_truth = qa.get("answer", [])  # Get ground-truth answers
        extracted_answer = None
        metric = -1 # Default to parsing failure

        # Find content after the last "Answer:" marker
        answer_markers = ["Answer:", "**Answer:**"]
        answer_text = None
        
        # Try different answer markers
        for marker in answer_markers:
            answer_start_index = response.rfind(marker)
            if answer_start_index != -1:
                answer_text = response[answer_start_index + len(marker):].strip()
                break
                
        if answer_text or use_agent:
            if use_agent:
                answer_text = response[0:].strip()

            code_block_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', answer_text, re.DOTALL)
            if code_block_match:
                answer_text = code_block_match.group(1).strip()
            else:
                # If no code block, clean all code block markers
                answer_text = re.sub(r'```.*?```|```[\s\S]*$', '', answer_text, flags=re.MULTILINE)
            
            # Clean the text before the answer, only keep the last list-like answer
            answer_text = re.sub(r'^.*?\[', '[', answer_text.strip(), flags=re.DOTALL)
            
            try:
                # Try to parse the answer text as a list
                extracted_answer = eval(answer_text)
                
                # Convert tuple format to list format
                if isinstance(extracted_answer, list):
                    converted_answer = []
                    for item in extracted_answer:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            motif_name = str(item[0]).lower().strip()  # Normalize motif name
                            try:
                                count = int(float(item[1]))  # Handle possible float
                                converted_answer.append([motif_name, count])
                            except (ValueError, TypeError):
                                continue
                    extracted_answer = converted_answer

                # Check if the format is correct (each element in the list is [motif_name, time] format)
                if isinstance(extracted_answer, list) and \
                   all(isinstance(x, list) and len(x) == 2 and \
                       isinstance(x[0], str) and isinstance(x[1], (int, float)) \
                       for x in extracted_answer):
                    
                    # Convert extracted answer and ground truth to sets for comparison (ignore order)
                    extracted_set = {(x[0].lower(), x[1]) for x in extracted_answer}
                    ground_truth_set = {(x[0].lower(), x[1]) for x in ground_truth}
                    # INSERT_YOUR_CODE
                    # Calculate the sum of the ratios of extracted/ground truth for elements with the same motif_name
                    result = 0.0
                    for motif, gt_count in ground_truth_set:
                        for motif2, ex_count in extracted_set:
                            if motif == motif2 and ex_count <= gt_count:
                                try:
                                    result += float(ex_count) / float(gt_count)
                                except Exception:
                                    pass
                    metric = float(result / len(ground_truth_set))
                    # print(result, len(ground_truth_set), metric)
            except Exception as e:
                print(f"Error parsing answer: {str(e)}")
                metric = -1
                extracted_answer = answer_text   
        return metric, extracted_answer


        

        