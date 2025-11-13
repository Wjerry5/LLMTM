from .base import DyGraphTask
import re
import json # 用于格式化列表以在 prompt 中显示
import networkx as nx
from collections import defaultdict # 确保导入

# --- 新的辅助函数，用于检查时序约束 ---
def check_temporal_constraints(mapping, sorted_motif_events, context_times_all, motif_time_window):
    """
    检查给定的同构映射是否存在一个时间戳序列满足时序和窗口约束。
    (使用迭代方法寻找有效序列)
    """
    possible_sequences = [[]]  # 存储部分有效的时间戳序列

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
                if t_c > last_t: # 检查顺序
                    if t_start is None: # 第一个事件
                        next_possible_sequences.append(sequence + [t_c])
                    else: # 后续事件，检查窗口
                        if motif_time_window <= 0 or (t_c - t_start) < motif_time_window:
                            next_possible_sequences.append(sequence + [t_c])

        possible_sequences = next_possible_sequences
        if not possible_sequences: return False # 没有有效的序列可以扩展了

    return True # 成功构建了至少一个完整序列

# --- 替换掉旧的 judge 函数 ---
def judge(context, motif_definition, motif_time_window):
    """
    精确判断 context 图中是否存在一个 motif_definition 的实例，
    同时满足拓扑、时间顺序和时间窗口约束 (考虑所有时间戳)。
    """
    context_a = [e for e in context if len(e) == 4 and e[3] == 'a']
    motif_a = [e for e in motif_definition if len(e) == 4 and e[3] == 'a']

    if not motif_a: return "No"

    # 1. 解析 Motif
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
        print(f"错误 (judge): 解析 motif_definition 失败: {e}")
        return "Error"

    # 2. 解析 Context (存储所有时间戳)
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
         print(f"错误 (judge): 解析 context 失败: {e}")
         return "Error"

    # 3. 检查同构
    if G_context.number_of_nodes() < G_motif_template.number_of_nodes() or \
       G_context.number_of_edges() < G_motif_template.number_of_edges():
        return "No"

    matcher = nx.isomorphism.GraphMatcher(G_context, G_motif_template)

    # 4. 迭代检查每个同构映射的时序约束
    for mapping in matcher.subgraph_isomorphisms_iter():
        reversed_mapping = {v: k for k, v in mapping.items()}
        if check_temporal_constraints(reversed_mapping, sorted_motif_events, context_times_all, motif_time_window):
            return "Yes" # 找到一个满足条件的实例

    # 5. 如果所有映射都不满足时序约束
    return "No"


class DyGraphTaskJudgeMultiMotif(DyGraphTask):
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
            if judge(context, predefined_motif, motif_definition["T"]) == "Yes":
                answer.append(motif_name)
        
        print(answer)

        qa = {
            "context": context,       # 待判断的图
            "query": None, # 对于判断任务，query 通常嵌入在问题模板中
            "answer": answer,               # 真实答案 "Yes" 或 "No"
            "task": self.task,
            }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        """生成任务指令 (中文注释)"""
        return "Your task is to identify What temporal motifs present in the given undirected dynamic graph?"

    def generate_instructor_answer(self, *args, **kwargs):
        """生成答案格式指令 (中文注释)"""

        return "Give the answer by listing the names of temporal motifs at the end of your response after 'Answer:'"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        """生成 Few-shot 示例 (中文注释)"""
        # 这里的 [ 之所以是红色的，是因为在下面这行代码中有一个括号配对错误，导致语法高亮器检测到括号不匹配：
        # (7, 19, 1, 'a'), 10, 13, 1, 'a'), (10, 18, 1, 'a'),
        # 其中 (7, 19, 1, 'a'), 后面应该是 (10, 13, 1, 'a')，而不是 10, 13, 1, 'a')
        # 正确写法如下，已修正括号配对问题：

        qa_examples_raw = [
            (
#             [(1, 3, 0, 'a'), (1, 7, 0, 'a'), (8, 12, 0, 'a'), (10, 16, 0, 'a'), (1, 9, 1, 'a'), (7, 19, 1, 'a'), (10, 13, 1, 'a'), (10, 18, 1, 'a'), (2, 5, 2, 'a'), (3, 10, 2, 'a'), (6, 11, 2, 'a'), (1, 7, 3, 'd'), (10, 16, 3, 'd'), (12, 19, 3, 'a'), (2, 8, 4, 'a'), (4, 6, 4, 'a'), (7, 8, 4, 'a'), (0, 1, 5, 'a'), (1, 2, 5, 'a'), (3, 7, 5, 'a'), (1, 17, 6, 'a'), (3, 7, 6, 'd'), (8, 13, 6, 'a'), (9, 18, 6, 'a'), (1, 13, 7, 'a'), (7, 9, 7, 'a'), (7, 10, 7, 'a'), (8, 9, 7, 'a'), (9, 10, 7, 'a'), (16, 18, 7, 'a'), (0, 10, 8, 'a'), (11, 12, 8, 'a'), (11, 16, 8, 'a'), (14, 19, 8, 'a'), (0, 14, 9, 'a'), (1, 18, 9, 'a'), (7, 8, 9, 'd'), (7, 16, 9, 'a'), (8, 17, 9, 'a'), (8, 19, 9, 'a'), (9, 19, 9, 'a'), (11, 16, 9, 'd'), (14, 18, 9, 'a'), (1, 18, 10, 'd'), (3, 10, 10, 'd'), (4, 18, 10, 'a'), (9, 19, 10, 'd'), (11, 12, 10, 'd'), (13, 14, 10, 'a'), (0, 1, 11, 'd'), (0, 4, 11, 'a'), (0, 17, 11, 'a'), (1, 13, 11, 'd'), (4, 7, 11, 'a'), (8, 13, 11, 'd'), (9, 10, 11, 'd'), (0, 4, 12, 'd'), (0, 9, 12, 'a'), (1, 10, 12, 'a'), (1, 15, 12, 'a'), (2, 8, 12, 'd'), (4, 7, 12, 'd'), (4, 12, 12, 'a'), (4, 18, 12, 'd'), (5, 9, 12, 'a'), (7, 16, 12, 'd'), (8, 9, 12, 'd'), (10, 17, 12, 'a'), (13, 14, 12, 'd'), (0, 17, 13, 'd'), (1, 10, 13, 'd'), (1, 14, 13, 'a'), (1, 15, 13, 'd'), (5, 12, 13, 'a'), (7, 9, 13, 'd'), (8, 16, 13, 'a'), (9, 15, 13, 'a'), (12, 13, 13, 'a'), (0, 9, 14, 'd'), (1, 14, 14, 'd'), (3, 6, 14, 'a'), (4, 12, 14, 'd'), (6, 8, 14, 'a'), (8, 10, 14, 'a'), (8, 16, 14, 'd'), (9, 18, 14, 'd'), (12, 13, 14, 'd'), (14, 19, 14, 'd'), (15, 17, 14, 'a')],
#             [],  # query
#             """**Chain of Thought:**  
# 1. My goal is to determine all temporal motifs present in the given undirected dynamic graph.
# 2. I'll start by identifying the nodes and added edges in the graph.
#     * Nodes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
#     * Edges: (1, 3), (1, 7), (8, 12), (10, 16), (1, 9), (7, 19), (10, 13), (10, 18), (2, 5), (3, 10), (6, 11), (12, 19), (2, 8), (4, 6), (7, 8), (0, 1), (1, 2), (3, 7), (1, 17), (8, 13), (9, 18), (1, 13), (7, 9), (7, 10), (8, 9), (9, 10), (16, 18), (0, 10), (11, 12), (11, 16), (14, 19), (0, 14), (1, 18), (7, 16), (8, 17), (8, 19), (9, 19), (14, 18), (4, 18), (13, 14), (0, 4), (0, 17), (4, 7), (0, 9), (1, 10), (1, 15), (5, 9), (10, 17), (1, 14), (5, 12), (8, 16), (9, 15), (12, 13), (3, 6), (6, 8), (8, 10), (15, 17)
# 3. Next, I'll iterate through all the possible existing motifs mentioned in the question. For each motif, we'll first consider the pattern matching, then check the temporal strictly increasing, finally check the time window constraint.
#     * triangle motif is a sequence of three edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. No motif is found.
#     * 3-star motif is a star structure where all edges are connected to a central node, and the timestamps are strictly increasing, the difference between the maximum and minimum timestamp is at most 3. Hence (0, 10, 8, 'a'), (0, 14, 9, 'a'), (0, 4, 11, 'a') forms a 3-star motif.
#     * 4-path motif is a sequence of three edges forming a path, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. Hence (0, 4, 11, 'a'), (4, 12, 12, 'a'), (12, 13, 13, 'a') forms a 4-path motif.
#     * 4-cycle motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (2, 8, 4, 'a'), (1, 2, 5, 'a'), (1, 17, 6, 'a'), (8, 17, 9, 'a') forms a 4-cycle motif.
#     * 4-tailedtriangle motif is a sequence of four edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u1'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (7, 10, 7, 'a'), (0, 10, 8, 'a'), (0, 17, 11, 'a'), (10, 17, 12, 'a') forms a 4-tailedtriangle motif.
#     * butterfly motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence  (2, 8, 4, 'a'), (1, 2, 5, 'a'), (1, 17, 6, 'a'), (8, 17, 9, 'a') forms a butterfly motif.
#     * 4-chordalcycle motif is a sequence of five edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3' ), ('u3', 'u1'), ('u3', 'u0'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14. Hence (10, 16, 0, 'a'), (3, 10, 2, 'a'), (3, 7, 5, 'a'), (7, 10, 7, 'a'), (7, 16, 9, 'a') forms a 4-chordalcycle motif.
#     * 4-clique motif is a sequence of six edges ('u0', 'u1'), ('u1', 'u2'), ('u1', 'u3'), ('u2', 'u3'), ('u3', 'u0'), ('u0', 'u2'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. No motif is found.
#     * bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. Hence (2, 5, 2, 'a'), (1, 2, 5, 'a'), (1, 18, 9, 'a') (4, 18, 10, 'a'), (4, 12, 12, 'a'), (5, 12, 13, 'a') forms a bitriangle motif.
# 4. Therefore, the graph contains the following temporal motifs: 3-star, 4-path, 4-cycle, 4-tailedtriangle, butterfly, 4-chordalcycle, bitriangle.""",
#             ["3-star", "4-path", "4-cycle", "4-tailedtriangle", "butterfly", "4-chordalcycle", "bitriangle"] # answer (会被 make_qa_example 格式化)

        [(0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 2, 3, 'a'), (2, 4, 4, 'a'), (2, 1, 5, 'a'), (2, 0, 6, 'a'), (4, 5, 7, 'a'), (0, 1, 8, 'd'), (5, 0, 9, 'a')],
            [],  # query
            """**Chain of Thought:**  
1. My goal is to determine all temporal motifs present in the given undirected dynamic graph.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5
    * Edges: (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 2, 3, 'a'), (2, 4, 4, 'a'), (2, 1, 5, 'a'), (2, 0, 6, 'a'), (4, 5, 7, 'a'), (5, 0, 9, 'a')
3. Next, I'll iterate through all the possible existing motifs mentioned in the question. For each motif, we'll first consider the pattern matching, then check the temporal strictly increasing, finally check the time window constraint.
    * triangle motif is a sequence of three edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 0, 2, 'a') forms a triangle motif.
    * 3-star motif is a star structure where all edges are connected to a central node, and the timestamps are strictly increasing, the difference between the maximum and minimum timestamp is at most 3. Hence (1, 3, 1, 'a'), (3, 0, 2, 'a'), (3, 2, 3, 'a') forms a 3-star motif.
    * 4-path motif is a sequence of three edges forming a path, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 3. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a') forms a 4-path motif.
    * 4-cycle motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 0, 6, 'a') forms a 4-cycle motif.
    * 4-tailedtriangle motif is a sequence of four edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u1'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 1, 5, 'a') forms a 4-tailedtriangle motif.
    * butterfly motif is a sequence of four edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 6. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 0, 6, 'a') forms a butterfly motif.
    * 4-chordalcycle motif is a sequence of five edges ('u0', 'u1'), ('u1', 'u2'), ('u2', 'u3' ), ('u3', 'u1'), ('u3', 'u0'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 14. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 1, 5, 'a'), (2, 0, 6, 'a') forms a 4-chordalcycle motif.
    * 4-clique motif is a sequence of six edges ('u0', 'u1'), ('u1', 'u2'), ('u1', 'u3'), ('u2', 'u3'), ('u3', 'u0'), ('u0', 'u2'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. No motif is found.
    * bitriangle motif is a sequence of six edges forming a closed cycle, where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 15. Hence (0, 1, 0, 'a'), (1, 3, 1, 'a'), (3, 2, 3, 'a'), (2, 4, 4, 'a'), (4, 5, 7, 'a'), (5, 0, 9, 'a') forms a bitriangle motif.
4. Therefore, the graph contains the following temporal motifs: triangle, 3-star, 4-path, 4-cycle, 4-tailedtriangle, butterfly, 4-chordalcycle, bitriangle.""",
            ["triangle", "3-star", "4-path", "4-cycle", "4-tailedtriangle", "butterfly", "4-chordalcycle", "bitriangle"] # answer (会被 make_qa_example 格式化)
        )
    ]
        return self.make_qa_example(num, qa_examples_raw)

    def generate_prompt_question(self, query = None, *args, **kwargs):
        """
        生成包含 Motif 定义和图上下文的问题。(中文注释)
        kwargs 应该包含 'motif_definition' 和 'context'。
        """

        # 使用你提供的问题模板
        return f" What temporal motifs present in the given undirected dynamic graph?"

    def evaluate(self, qa, response, use_agent):
        """
        评估模型响应是否为 "Yes" 或 "No"，并与真实答案比较。(中文注释)

        Args:
            qa (dict): 包含真实答案 'answer' ("Yes"或"No") 的字典。
            response (str): 模型生成的完整响应字符串。

        Returns:
            tuple: (matched_count, total_ground_truth_count)
                   matched_count: 与ground_truth匹配的数量
                   total_ground_truth_count: ground_truth的总数量
        """
        # 修复: 正确处理列表类型的answer
        ground_truth_raw = qa.get("answer", [])
        if isinstance(ground_truth_raw, list):
            # 如果answer是列表，转换为小写集合
            ground_truth_set = set(motif.strip().lower() for motif in ground_truth_raw if motif)
        elif isinstance(ground_truth_raw, str):
            # 如果answer是字符串，按逗号分割后转换为小写集合
            ground_truth_set = set(motif.strip().lower() for motif in ground_truth_raw.split(',') if motif.strip())
        else:
            # 处理其他类型，转为空集合
            ground_truth_set = set()
            
        extracted_answer = None
        matched_count = 0  # 匹配的数量
        total_ground_truth_count = len(ground_truth_set)  # 真实答案的总数量

        try:
            # 查找最后一个 "Answer:" 或 "**Answer:**" 标记后的内容
            # 首先替换所有的\n为空格，并处理可能的转义字符
            response = response.replace('\\n', '\n').replace('\n', ' ')
            
            answer_markers = ["Answer:", "**Answer:**"]
            answer_start_index = -1
            for marker in answer_markers:
                index = response.rfind(marker)
                if index != -1:
                    answer_start_index = index
                    answer_marker = marker
                    break
            
            if answer_start_index != -1 or use_agent == 1:
                # 提取Answer:后的内容并处理
                if use_agent == 1:
                    answer_text = response[0:].strip()
                else:
                    answer_text = response[answer_start_index + len(answer_marker):].strip()
                if len(answer_text) > 200:
                    return 0, total_ground_truth_count

                # 将答案文本按逗号或空格分割成列表,并去除空白
                extracted_motifs = set()
                for motif in re.split('[,\s]+', answer_text):
                    # 清理motif字符串：去掉括号、引号、点等不需要的字符
                    motif_clean = re.sub(r'[()\'\".,;:\[\]{}]', '', motif.strip())
                    if motif_clean:  # 忽略空字符串
                        extracted_motifs.add(motif_clean.lower())
                
                print(f"真实答案集合: {ground_truth_set}")
                print(f"提取的答案集合: {extracted_motifs}")
                
                # 计算交集数量（相同的数量）
                all_motifs = ['3-star', 'butterfly', '4-tailedtriangle', 'triangle', '4-chordalcycle', '4-path', 'bitriangle', '4-cycle', '4-clique']
                extracted_motifs = set(all_motifs).intersection(extracted_motifs)
                print(f"有效的motifs: {extracted_motifs}")
                
                intersection = extracted_motifs.intersection(ground_truth_set)
                
                matched_count = len(intersection) - (len(extracted_motifs) - len(intersection))
                
                print(f"匹配的motifs: {intersection}")
                print(f"匹配数量: {matched_count}, 总数量: {total_ground_truth_count}")
                
                extracted_answer = list(extracted_motifs)
            else:
                print("未找到Answer:或**Answer:**标记，返回0匹配")
                matched_count = 0
                extracted_answer = []
            
            if total_ground_truth_count == 0:
                return 0, extracted_answer
            return float(matched_count/total_ground_truth_count), extracted_answer
            
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            return -1, None

        