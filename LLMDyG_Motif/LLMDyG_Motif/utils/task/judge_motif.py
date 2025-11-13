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
        if set(mapping.keys()) != set(G_motif_template.nodes()):
             continue
        # 调用新的辅助函数进行精确时序检查
        reversed_mapping = {v: k for k, v in mapping.items()}
        if check_temporal_constraints(reversed_mapping, sorted_motif_events, context_times_all, motif_time_window):
            return "Yes" # 找到一个满足条件的实例

    # 5. 如果所有映射都不满足时序约束
    return "No"

class DyGraphTaskJudgeMotif(DyGraphTask):
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
        original_motif = info.get('original_motif', []) # Motif 定义
        # 确保 context 和 predefined_motif 是正确的类型列表
        context = [(int(u), int(v), int(t), str(op)) for u, v, t, op in context]
        original_motif = [(str(u), str(v), str(t), str(op)) for u, v, t, op in original_motif]
        
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
        # 计算 predefined_motif 的节点数和边数
        nodes = set()
        for u, v, _, _ in predefined_motif:
            nodes.add(u)
            nodes.add(v)
        N = len(nodes)  # 节点数
        M = len(predefined_motif)  # 边数
        T = info.get('motif_time_window', 0)
        answer = judge(context, predefined_motif, T)

        qa = {
            "context": context,       # 待判断的图
            "query": [original_motif, [N, M, T], [self.motif_name]], # 对于判断任务，query 通常嵌入在问题模板中
            "answer": answer,               # 真实答案 "Yes" 或 "No"
            "task": self.task,
            }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        """生成任务指令 (中文注释)"""
        return "Your task is to answer whether the given undirected dynamic graph is the given temporal motif?"

    def generate_instructor_answer(self, *args, **kwargs):
        """生成答案格式指令 (中文注释)"""

        return "Give the answer as 'Yes' or 'No' at the end of your response after 'Answer:'"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        """生成 Few-shot 示例 (中文注释)"""
        

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
                    ["Yes"] # answer (会被 make_qa_example 格式化)
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
                    ["Yes"] # answer (会被 make_qa_example 格式化)
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
                    ["Yes"] # answer (会被 make_qa_example 格式化)
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
4. Therefore, the graph is a 4-path motif.""", #cot(重跑)
                    ["Yes"] # answer (会被 make_qa_example 格式化)
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
                    ["Yes"] # answer (会被 make_qa_example 格式化)
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
                    ["Yes"] # answer (会被 make_qa_example 格式化)
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
                    ["Yes"] # answer (会被 make_qa_example 格式化)
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
                    ["Yes"] # answer (会被 make_qa_example 格式化)
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
                    ["Yes"] # answer (会被 make_qa_example 格式化)
                )
            ]
            

        # 转换为 make_qa_example 需要的格式
        qa_formatted = []
        for item in qa_examples_raw:
            try:
                if len(item) == 6:
                    graph_ctx, motif_def, NMT, motif_name, s, answer = item
                    qa_formatted.append([graph_ctx, [motif_def, NMT, motif_name], s, answer])
                else:
                    print(f"警告: 示例格式不正确，期望6个元素，实际有{len(item)}个")
            except Exception as e:
                print(f"警告: 处理示例时出错: {e}")
                continue

        return self.make_qa_example(num, qa_formatted)

    def generate_prompt_question(self, query = None, *args, **kwargs):
        """
        生成包含 Motif 定义和图上下文的问题。(中文注释)
        kwargs 应该包含 'motif_definition' 和 'context'。
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

        # 使用你提供的问题模板
        return f"Given a {motif_name} temporal motif which is a {N}-node, {M}-edge, {T}-temporal motif with the edges{motif_def_str}. Whether the given undirected dynamic graph is the given motif?"

    def evaluate(self, qa, response):
        """
        评估模型响应是否为 "Yes" 或 "No"，并与真实答案比较。(中文注释)

        Args:
            qa (dict): 包含真实答案 'answer' ("Yes"或"No") 的字典。
            response (str): 模型生成的完整响应字符串。

        Returns:
            tuple: (metric, extracted_answer)
                   metric: 1 (正确), 0 (错误), -3 (解析失败)
                   extracted_answer: 解析出的 "Yes" 或 "No" (小写)，或 None
        """
        ground_truth = qa.get("answer", "").strip().lower() # 获取真实答案并转小写
        extracted_answer = None
        metric = -3 # 默认为解析失败

        # # 查找最后一个 "Answer:" 标记后的内容
        # answer_marker = "Answer:"
        # answer_start_index = response.rfind(answer_marker)

        # if answer_start_index != -1:
        #     answer_part = response[answer_start_index + len(answer_marker):].strip()
        #     # 提取 "Yes" 或 "No"，忽略大小写和可能的标点
        #     # 使用更宽松的正则查找 "yes" 或 "no" (忽略大小写)
        #     match = re.search(r'\b(yes|no)\b', answer_part, re.IGNORECASE)
        #     if match:
        #         extracted_answer = match.group(1).lower()
        #         # print(f"  评估: GT='{ground_truth}', Extracted='{extracted_answer}'") # 调试
        #         if extracted_answer == ground_truth:
        #             metric = 1 # 正确匹配
        #         else:
        #             metric = 0 # 答案错误
        #     else:
        #         print(f"  评估: 解析失败 - 未能在 '{answer_part}' 中找到 'Yes' 或 'No'")
        #         metric = -3 # 解析失败
        # else:
        #     print("  评估: 解析失败 - 未能在响应中找到 'Answer:' 标记")
        #     metric = -3 # 解析失败
        answer_part = response[0:].strip()
        match = re.search(r'\b(yes|no)\b', answer_part, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).lower()
            # print(f"  评估: GT='{ground_truth}', Extracted='{extracted_answer}'") # 调试
            if extracted_answer == ground_truth:
                metric = 1 # 正确匹配
            else:
                metric = 0 # 答案错误
        else:
            print(f"  评估: 解析失败 - 未能在 '{answer_part}' 中找到 'Yes' 或 'No'")
            metric = -3 # 解析失败



        return metric, extracted_answer
        
        