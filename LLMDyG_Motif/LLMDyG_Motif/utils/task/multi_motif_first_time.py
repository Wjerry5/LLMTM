from .base import DyGraphTask
import re
import json # 用于格式化列表以在 prompt 中显示
import networkx as nx
from collections import defaultdict # 确保导入
from ..modif_judge_count import motif_and_first_time, judge


class DyGraphTaskMultiMotifFirstTime(DyGraphTask):
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
            print(predefined_motif)
            print(motif_definition["T"])
            print(context)
            print(judge(context, predefined_motif, motif_definition["T"]))
            first_time = motif_and_first_time(context, predefined_motif, motif_definition["T"])
            if first_time != 0x3f3f3f3f:
                answer.append([motif_name, first_time])
        
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
        return "Your task is to answer When does each of the above temporal motifs first appear in the given undirected dynamic graph? The appearance time of a motif refers to the time when the last edge of the motif occurs. Omit temporal motifs that do not appear in the undirected dynamic graph."

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
# 1. My goal is to determine the first appearing time of all temporal motifs present in the given undirected dynamic graph.
# 2. I'll start by identifying the nodes and added edges in the graph.
#     * Nodes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
#     * Edges: (1, 3), (1, 7), (8, 12), (10, 16), (1, 9), (7, 19), (10, 13), (10, 18), (2, 5), (3, 10), (6, 11), (12, 19), (2, 8), (4, 6), (7, 8), (0, 1), (1, 2), (3, 7), (1, 17), (8, 13), (9, 18), (1, 13), (7, 9), (7, 10), (8, 9), (9, 10), (16, 18), (0, 10), (11, 12), (11, 16), (14, 19), (0, 14), (1, 18), (7, 16), (8, 17), (8, 19), (9, 19), (14, 18), (4, 18), (13, 14), (0, 4), (0, 17), (4, 7), (0, 9), (1, 10), (1, 15), (5, 9), (10, 17), (1, 14), (5, 12), (8, 16), (9, 15), (12, 13), (3, 6), (6, 8), (8, 10), (15, 17)
# 3. Next, I'll iterate through all the possible existing motifs mentioned in the question. For each motif, Set first_time to infinity. Regarding the temporal motif matching, we'll first consider the pattern matching, then check the temporal strictly increasing, finally check the time window constraint. If a motif is found, update first_time to the minimum of the current first_time and the time when the last edge of the motif occurs.
#     * triangle motif is a sequence of three edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. No motif is found.
#     * 3-star motif is a star structure where all edges are connected to a central node, and the timestamps are strictly increasing, the difference between the maximum and minimum timestamp is at most 3. Hence (10, 16, 0, 'a'), (10, 13, 1, 'a'), (3, 10, 2, 'a') forms a 3-star motif, and the first time is 2.
#     * 4-path motif is a sequence of three edges forming a path, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. Hence (1, 7, 0, 'a'), (7, 19, 1, 'a') (12, 19, 3, 'a') forms a 4-path motif, and the first time is 3.
#     * 4-cycle motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (2, 8, 4, 'a'), (1, 2, 5, 'a'), (1, 17, 6, 'a'), (8, 17, 9, 'a') forms a 4-cycle motif, and the first time is 9.
#     * 4-tailedtriangle motif is a sequence of four edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u1'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (10, 13, 1, 'a'), (3, 10, 2, 'a'), (3, 7, 5, 'a'), (7, 10, 7, 'a') forms a 4-tailedtriangle motif, and the first time is 7.
#     * butterfly motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence  (2, 8, 4, 'a'), (1, 2, 5, 'a'), (1, 17, 6, 'a'), (8, 17, 9, 'a') forms a butterfly motif, and the first time is 9.
#     * 4-chordalcycle motif is a sequence of five edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3' ), ('u3', 'u1'), ('u3', 'u0'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14. Hence (10, 16, 0, 'a'), (3, 10, 2, 'a'), (3, 7, 5, 'a'), (7, 10, 7, 'a'), (7, 16, 9, 'a') forms a 4-chordalcycle motif, and the first time is 9.
#     * 4-clique motif is a sequence of six edges ('u0', 'u1'), ('u1', 'u2'), ('u1', 'u3'), ('u2', 'u3'), ('u3', 'u0'), ('u0', 'u2'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. No motif is found.
#     * bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. Hence (8, 12, 0, 'a'), (2, 8, 4, 'a'), (1, 2, 5, 'a'), (1, 18, 9, 'a'), (4, 18, 10, 'a'), (4, 12, 12, 'a') forms a bitriangle motif, and the first time is 12.
# 4. Therefore, the first appearing time of all temporal motifs are: (3-star, 2), (4-path, 3), (4-cycle, 9), (4-tailedtriangle, 7), (butterfly, 9), (4-chordalcycle, 9), (bitriangle, 12).""",
#                 [["3-star", 2], ["4-path", 3], ["4-cycle", 9], ["4-tailedtriangle", 7], ["butterfly", 9], ["4-chordalcycle", 9], ["bitriangle", 12]] 
[(0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 2, 3, 'a'), (2, 4, 4, 'a'), (2, 1, 5, 'a'), (2, 0, 6, 'a'), (4, 5, 7, 'a'), (0, 1, 8, 'd'), (5, 0, 9, 'a')],
                [],
                """**Chain of Thought:**  
1. My goal is to determine the first appearing time of all temporal motifs present in the given undirected dynamic graph.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5
    * Edges: (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 2, 3, 'a'), (2, 4, 4, 'a'), (2, 1, 5, 'a'), (2, 0, 6, 'a'), (4, 5, 7, 'a'), (5, 0, 9, 'a')
3. Next, I'll iterate through all the possible existing motifs mentioned in the question. For each motif, Set first_time to infinity. Regarding the temporal motif matching, we'll first consider the pattern matching, then check the temporal strictly increasing, finally check the time window constraint. If a motif is found, update first_time to the minimum of the current first_time and the time when the last edge of the motif occurs.
    * triangle motif is a sequence of three edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a') forms a triangle motif, and the first time is 2.
    * 3-star motif is a star structure where all edges are connected to a central node, and the timestamps are strictly increasing, the difference between the maximum and minimum timestamp is at most 3. Hence (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 2, 3, 'a') forms a 3-star motif, and the first time is 3.
    * 4-path motif is a sequence of three edges forming a path, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a') forms a 4-path motif, and the first time is 3.
    * 4-cycle motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 0, 6, 'a') forms a 4-cycle motif, and the first time is 6.
    * 4-tailedtriangle motif is a sequence of four edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u1'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 1, 5, 'a') forms a 4-tailedtriangle motif, and the first time is 5.
    * butterfly motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 0, 6, 'a') forms a butterfly motif, and the first time is 6.
    * 4-chordalcycle motif is a sequence of five edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3' ), ('u3', 'u1'), ('u3', 'u0'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 1, 5, 'a'), (2, 0, 6, 'a') forms a 4-chordalcycle motif, and the first time is 6.
    * 4-clique motif is a sequence of six edges ('u0', 'u1'), ('u1', 'u2'), ('u1', 'u3'), ('u2', 'u3'), ('u3', 'u0'), ('u0', 'u2'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. No motif is found.
    * bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 4, 4, 'a'), (4, 5, 7, 'a'), (5, 0, 9, 'a') forms a bitriangle motif, and the first time is 9.
4. Therefore, the first appearing time of all temporal motifs are: (triangle, 2), (3-star, 3), (4-path, 3), (4-cycle, 6), (4-tailedtriangle, 5), (butterfly, 6), (4-chordalcycle, 6), (bitriangle, 9).""",
                [["triangle", 2], ["3-star", 3], ["4-path", 3], ["4-cycle", 6], ["4-tailedtriangle", 5], ["butterfly", 6], ["4-chordalcycle", 6], ["bitriangle", 9]] 
            )
        ]
        return self.make_qa_example(num, qa_examples_raw)

    def generate_prompt_question(self, query = None, *args, **kwargs):
        """
        生成包含 Motif 定义和图上下文的问题。(中文注释)
        kwargs 应该包含 'motif_definition' 和 'context'。
        """

        # 使用你提供的问题模板
        return f" When does each of the above temporal motifs first appear in the given undirected dynamic graph?"

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
        print("ground_truth: ", ground_truth)

        # 查找最后一个列表形式的答案
        # 首先尝试查找代码块中的列表
        code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_blocks:
            # 使用最后一个代码块
            answer_text = code_blocks[-1].strip()
        else:
            # 如果没有代码块，查找最后一个 "Answer:" 标记后的内容
            answer_markers = ["Answer:", "**Answer:**", "答案：", "### Final Answer"]
            answer_text = None
            for marker in answer_markers:
                answer_start_index = response.rfind(marker)
                if answer_start_index != -1:
                    answer_text = response[answer_start_index + len(marker):].strip()
                    break

        if answer_text:
            try:
                # 尝试将答案文本解析为列表
                extracted_answer = []
                
                # 清理文本，移除无关标记
                answer_text = re.sub(r'</?\w+>', '', answer_text)  # 移除XML标记
                answer_text = re.sub(r'```[^`]*```|```', '', answer_text)  # 移除代码块标记
                answer_text = answer_text.strip()
                print("Cleaned answer_text: ", answer_text)
                
                # 首先尝试解析 Python 列表格式
                if answer_text.strip().startswith('[') and answer_text.strip().endswith(']'):
                    try:
                        # 将字符串转换为有效的 Python 列表格式
                        # 替换单引号为双引号，确保字符串格式正确
                        formatted_text = answer_text.replace("'", '"')
                        print("Formatted text for eval: ", formatted_text)
                        # 尝试直接解析为 Python 对象
                        parsed_list = eval(formatted_text)
                        print("Successfully parsed list: ", parsed_list)
                        if isinstance(parsed_list, list):
                            for item in parsed_list:
                                # 修改判断条件，同时接受list和tuple类型
                                if (isinstance(item, (list, tuple)) and len(item) == 2):
                                    motif_name = str(item[0])
                                    time_value = int(item[1])
                                    extracted_answer.append([motif_name, time_value])
                            print("Extracted answer: ", extracted_answer)
                    except Exception as e:
                        print("Eval failed with error: ", str(e))
                        # 如果 eval 失败，回退到其他解析方法
                        pass

                # 如果 Python 列表解析失败，尝试其他格式
                if not extracted_answer:
                    print("Falling back to other parsing methods")
                    lines = answer_text.strip().split('\n')
                    # 处理其他格式
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # 处理 ("motif_name", time) 格式
                        if line.startswith('(') and line.endswith(')'):
                            parts = line.strip('()').split(',')
                            if len(parts) == 2:
                                motif_name = parts[0].strip().strip('"')
                                try:
                                    time_value = int(parts[1].strip())
                                    extracted_answer.append([motif_name, time_value])
                                except ValueError:
                                    continue
                                    
                        # 处理 - **motif_name**: time 格式
                        elif '**' in line and ':' in line:
                            line = line.lstrip('- ')
                            parts = line.split(':')
                            if len(parts) == 2:
                                motif_name = parts[0].strip().strip('*').strip()
                                try:
                                    time_value = int(parts[1].strip())
                                    extracted_answer.append([motif_name, time_value])
                                except ValueError:
                                    continue
                                    
                        # 处理 "motif_name", time 格式
                        elif '"' in line and ',' in line:
                            parts = line.split(',')
                            if len(parts) == 2:
                                motif_name = parts[0].strip().strip('"')
                                try:
                                    time_value = int(parts[1].strip())
                                    extracted_answer.append([motif_name, time_value])
                                except ValueError:
                                    continue
                                    
                        # 处理 "motif_name": time 格式
                        elif line.startswith('"') and ":" in line:
                            parts = line.split(':')
                            if len(parts) == 2:
                                motif_name = parts[0].strip().strip('"')
                                try:
                                    time_value = int(parts[1].strip())
                                    extracted_answer.append([motif_name, time_value])
                                except ValueError:
                                    continue
                                    
                        # 处理 - "motif_name": time 格式
                        elif line.startswith('-') and ":" in line:
                            line = line.lstrip('- ')
                            parts = line.split(':')
                            if len(parts) == 2:
                                motif_name = parts[0].strip().strip('"')
                                try:
                                    time_value = int(parts[1].strip())
                                    extracted_answer.append([motif_name, time_value])
                                except ValueError:
                                    continue
                
                # 将提取的答案和真实答案转换为集合进行比较(忽略顺序)
                if not extracted_answer:  # 如果没有提取到答案
                    print("No answer could be extracted")
                    metric = 0
                else:
                    # 将答案转换为字典，以motif名称为键
                    extracted_dict = {}
                    for motif_name, time_value in extracted_answer:
                        motif_name = motif_name.lower()
                        if motif_name not in extracted_dict:
                            extracted_dict[motif_name] = time_value

                    ground_truth_dict = {}
                    for motif_name, time_value in ground_truth:
                        motif_name = motif_name.lower()
                        if motif_name not in ground_truth_dict:
                            ground_truth_dict[motif_name] = time_value

                    print("Extracted dict: ", extracted_dict)
                    print("Ground truth dict: ", ground_truth_dict)

                    # 计算正确匹配的数量
                    correct_matches = 0
                    for motif_name, true_time in ground_truth_dict.items():
                        if motif_name in extracted_dict and extracted_dict[motif_name] == true_time:
                            correct_matches += 1

                    # 计算最终的metric
                    metric = float(correct_matches) / len(ground_truth_dict) if ground_truth_dict else 0
                    print("Metric: ", metric)
                
            except Exception as e:
                print(f"Error parsing answer: {e}")
                metric = -1
                extracted_answer = answer_text   
        return metric, extracted_answer


        