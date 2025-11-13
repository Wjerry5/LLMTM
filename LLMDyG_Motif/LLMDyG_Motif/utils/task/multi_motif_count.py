from .base import DyGraphTask
import re
import json # 用于格式化列表以在 prompt 中显示
import networkx as nx
from collections import defaultdict # 确保导入
from ..modif_judge_count import count, judge


class DyGraphTaskMultiMotifCount(DyGraphTask):
    """
    任务：判断给定的动态图是否为指定的时序 Motif。
    """
    def __init__(self, task, args):
        super().__init__(task, args)
        self.motif_name = args.motif_name

    def generate_qa(self, info, *args, **kwargs):
        """
        生成 QA 对。根据生成信息确定真实答案 ("Yes" 或 "No")。(中文注释)
        """
        context = info.get('edge_index', []) # 待判断的动态图边列表
        PREDEFINED_MOTIFS = {
    "triangle":     {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')], "T": 3},             # k=3, l=3
    "3-star":       {"edge": [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')], "T": 3},             # k=4, l=3 (中心节点为 0)
    "4-path":       {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')], "T": 3},             # k=4, l=3
    "4-cycle":      {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')], "T": 6}, # k=4, l=4
    "4-tailedtriangle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')], "T": 6}, # k=4, l=4
    "butterfly":    {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')], "T": 6}, # k=4, l=4 (与 4-cycle 拓扑相同，时序不同？根据图像解析有歧义，暂不包含)
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
                    # 检查是否需要转换节点标识符
                    if isinstance(u, str) and u.startswith('u'):
                        try:
                            u = int(u[1:])  # 从 'u0' 提取 0
                        except ValueError:
                            pass  # 保持原样
                    if isinstance(v, str) and v.startswith('u'):
                        try:
                            v = int(v[1:])  # 从 'u1' 提取 1
                        except ValueError:
                            pass  # 保持原样
                    if isinstance(t_rel, str) and t_rel.startswith('t'):
                        try:
                            t_rel = int(t_rel[1:])  # 从 't0' 提取 0
                        except ValueError:
                            pass  # 保持原样
                    processed_motif.append((u, v, t_rel, event_type))
                else:
                    processed_motif.append(item)  # 保持原样
        
            predefined_motif = processed_motif
            if judge(context, predefined_motif, motif_definition["T"]) == "Yes":
                answer.append([motif_name, count(context, predefined_motif, motif_definition["T"])])
        
        print(answer)

        qa = {
            "context": context,       # 待判断的图
            "query": None, # 对于判断任务，query 通常嵌入在问题模板中
            "answer": answer,               # [["triangle", 2], ["3-star", 3], ["4-path", 4]]
            "task": self.task,
            }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        """生成任务指令 (中文注释)"""
        return "Your task is to answer How many times does each of the above temporal motifs appear in the given undirected dynamic graph? Omit temporal motifs that do not appear in the undirected dynamic graph."

    def generate_instructor_answer(self, *args, **kwargs):
        """生成答案格式指令 (中文注释)"""

        return "Give the answer as a list of tuples consisting of the name of a temporal motif and an integer at the end of your response after 'Answer:'"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        """生成 Few-shot 示例 (中文注释)"""
        qa_examples_raw = [
            (
#                 [(1, 3, 0, 'a'), (1, 7, 0, 'a'), (8, 12, 0, 'a'), (10, 16, 0, 'a'), (1, 9, 1, 'a'), (7, 19, 1, 'a'), (10, 13, 1, 'a'), (10, 18, 1, 'a'), (2, 5, 2, 'a'), (3, 10, 2, 'a'), (6, 11, 2, 'a'), (1, 7, 3, 'd'), (10, 16, 3, 'd'), (12, 19, 3, 'a'), (2, 8, 4, 'a'), (4, 6, 4, 'a'), (7, 8, 4, 'a'), (0, 1, 5, 'a'), (1, 2, 5, 'a'), (3, 7, 5, 'a'), (1, 17, 6, 'a'), (3, 7, 6, 'd'), (8, 13, 6, 'a'), (9, 18, 6, 'a'), (1, 13, 7, 'a'), (7, 9, 7, 'a'), (7, 10, 7, 'a'), (8, 9, 7, 'a'), (9, 10, 7, 'a'), (16, 18, 7, 'a'), (0, 10, 8, 'a'), (11, 12, 8, 'a'), (11, 16, 8, 'a'), (14, 19, 8, 'a'), (0, 14, 9, 'a'), (1, 18, 9, 'a'), (7, 8, 9, 'd'), (7, 16, 9, 'a'), (8, 17, 9, 'a'), (8, 19, 9, 'a'), (9, 19, 9, 'a'), (11, 16, 9, 'd'), (14, 18, 9, 'a'), (1, 18, 10, 'd'), (3, 10, 10, 'd'), (4, 18, 10, 'a'), (9, 19, 10, 'd'), (11, 12, 10, 'd'), (13, 14, 10, 'a'), (0, 1, 11, 'd'), (0, 4, 11, 'a'), (0, 17, 11, 'a'), (1, 13, 11, 'd'), (4, 7, 11, 'a'), (8, 13, 11, 'd'), (9, 10, 11, 'd'), (0, 4, 12, 'd'), (0, 9, 12, 'a'), (1, 10, 12, 'a'), (1, 15, 12, 'a'), (2, 8, 12, 'd'), (4, 7, 12, 'd'), (4, 12, 12, 'a'), (4, 18, 12, 'd'), (5, 9, 12, 'a'), (7, 16, 12, 'd'), (8, 9, 12, 'd'), (10, 17, 12, 'a'), (13, 14, 12, 'd'), (0, 17, 13, 'd'), (1, 10, 13, 'd'), (1, 14, 13, 'a'), (1, 15, 13, 'd'), (5, 12, 13, 'a'), (7, 9, 13, 'd'), (8, 16, 13, 'a'), (9, 15, 13, 'a'), (12, 13, 13, 'a'), (0, 9, 14, 'd'), (1, 14, 14, 'd'), (3, 6, 14, 'a'), (4, 12, 14, 'd'), (6, 8, 14, 'a'), (8, 10, 14, 'a'), (8, 16, 14, 'd'), (9, 18, 14, 'd'), (12, 13, 14, 'd'), (14, 19, 14, 'd'), (15, 17, 14, 'a')],
#                 [],
#                 """**Chain of Thought:**  
# 1. My goal is to determine the number of all temporal motifs present in the given undirected dynamic graph.
# 2. I'll start by identifying the nodes and added edges in the graph.
#     * Nodes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
#     * Edges: (1, 3), (1, 7), (8, 12), (10, 16), (1, 9), (7, 19), (10, 13), (10, 18), (2, 5), (3, 10), (6, 11), (12, 19), (2, 8), (4, 6), (7, 8), (0, 1), (1, 2), (3, 7), (1, 17), (8, 13), (9, 18), (1, 13), (7, 9), (7, 10), (8, 9), (9, 10), (16, 18), (0, 10), (11, 12), (11, 16), (14, 19), (0, 14), (1, 18), (7, 16), (8, 17), (8, 19), (9, 19), (14, 18), (4, 18), (13, 14), (0, 4), (0, 17), (4, 7), (0, 9), (1, 10), (1, 15), (5, 9), (10, 17), (1, 14), (5, 12), (8, 16), (9, 15), (12, 13), (3, 6), (6, 8), (8, 10), (15, 17)
# 3. Next, I'll iterate through all the possible existing motifs mentioned in the question. For each motif, Initialize count to 0. Regarding the motif matching, we'll first consider the pattern matching, then check the temporal strictly increasing, finally check the time window constraint. If the motif is found, I'll increment the count by 1.
#     * triangle motif is a sequence of three edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. No motif is found.
#     * 3-star motif is a star structure where all edges are connected to a central node, and the timestamps are strictly increasing, the difference between the maximum and minimum timestamp is at most 3. After motif matching, count = 17.
#     * 4-path motif is a sequence of three edges forming a path, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. After motif matching, count = 24.
#     * 4-cycle motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. After motif matching, count = 5.
#     * 4-tailedtriangle motif is a sequence of four edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u1'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. After motif matching, count = 3.
#     * butterfly motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. After motif matching, count = 5. 
#     * 4-chordalcycle motif is a sequence of five edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3' ), ('u3', 'u1'), ('u3', 'u0'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14. After motif matching, count = 1.
#     * 4-clique motif is a sequence of six edges ('u0', 'u1'), ('u1', 'u2'), ('u1', 'u3'), ('u2', 'u3'), ('u3', 'u0'), ('u0', 'u2'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. No motif is found.
#     * bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. After motif matching, count = 2.
# 4. Therefore, the number of all temporal motifs present are: (3-star, 17), (4-path, 24), (4-cycle, 5), (4-tailedtriangle, 3), (butterfly, 5), (4-chordalcycle, 1), (bitriangle, 2).""",
#                 [["3-star", 17], ["4-path", 24], ["4-cycle", 5], ["4-tailedtriangle", 3], ["butterfly", 5], ["4-chordalcycle", 1], ["bitriangle", 2]] 

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
        生成包含 Motif 定义和图上下文的问题。(中文注释)
        kwargs 应该包含 'motif_definition' 和 'context'。
        """

        # 使用你提供的问题模板
        return f" How many times does each of the above temporal motifs appear in the given undirected dynamic graph?"

    def evaluate(self, qa, response):
        """
        评估生成的答案是否正确。
        返回:
            metric: 1表示完全正确,0表示错误,-3表示解析失败
            extracted_answer: 从回答中提取的答案
        """
        ground_truth = qa.get("answer", []) # 获取真实答案
        extracted_answer = None
        metric = -1 # 默认为解析失败

        # 查找最后一个 "Answer:" 标记后的内容
        answer_markers = ["Answer:", "**Answer:**", "答案："]
        answer_text = None
        
        # 尝试不同的答案标记
        for marker in answer_markers:
            answer_start_index = response.rfind(marker)
            if answer_start_index != -1:
                answer_text = response[answer_start_index + len(marker):].strip()
                break
                
        if answer_text:
            # 清理代码块标记和其后的注释内容
            # 首先尝试提取代码块中的内容
            code_block_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', answer_text, re.DOTALL)
            if code_block_match:
                answer_text = code_block_match.group(1).strip()
            else:
                # 如果没有代码块，清理所有代码块标记
                answer_text = re.sub(r'```.*?```|```[\s\S]*$', '', answer_text, flags=re.MULTILINE)
            
            # 清理答案前的说明文字，只保留最后一个列表形式的答案
            answer_text = re.sub(r'^.*?\[', '[', answer_text.strip(), flags=re.DOTALL)
            
            try:
                # 尝试将答案文本解析为列表
                extracted_answer = eval(answer_text)
                
                # 将元组格式转换为列表格式
                if isinstance(extracted_answer, list):
                    converted_answer = []
                    for item in extracted_answer:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            motif_name = str(item[0]).lower().strip()  # 标准化motif名称
                            try:
                                count = int(float(item[1]))  # 处理可能的浮点数
                                converted_answer.append([motif_name, count])
                            except (ValueError, TypeError):
                                continue
                    extracted_answer = converted_answer

                # 检查格式是否正确(列表中的每个元素都是[motif_name, time]格式)
                if isinstance(extracted_answer, list) and \
                   all(isinstance(x, list) and len(x) == 2 and \
                       isinstance(x[0], str) and isinstance(x[1], (int, float)) \
                       for x in extracted_answer):
                    
                    # 将提取的答案和真实答案转换为集合进行比较(忽略顺序)
                    extracted_set = {(x[0].lower(), x[1]) for x in extracted_answer}
                    ground_truth_set = {(x[0].lower(), x[1]) for x in ground_truth}
                    # INSERT_YOUR_CODE
                    # 计算 motif_name 相同的元素，extracted/ground_truth 的比值之和
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


        

        