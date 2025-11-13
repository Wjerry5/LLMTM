import networkx as nx
from collections import defaultdict

# PREDEFINED_MOTIFS = {
#     "triangle":     {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')], "T": 3},             # k=3, l=3
#     "3-star":       {"edge": [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')], "T": 3},             # k=4, l=3 (中心节点为 0)
#     "4-path":       {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')], "T": 3},             # k=4, l=3
#     "4-cycle":      {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')], "T": 6}, # k=4, l=4
#     "4-tailedtriangle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')], "T": 6}, # k=4, l=4
#     "butterfly":    {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')], "T": 6}, # k=4, l=4 (与 4-cycle 拓扑相同，时序不同？根据图像解析有歧义，暂不包含)
#     "4-chordalcycle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')], "T": 14}, # k=4, l=5
#     "4-clique":     {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')], "T": 15}, # k=4, l=6
#     "bitriangle":   {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')], "T": 15}, # k=6, l=6
# }

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
                        if motif_time_window <= 0 or (t_c - t_start) <= motif_time_window:
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
        for u, v, t, op in motif_a:
            if isinstance(u, str) and u.startswith('u'):
                u = int(u[1:])  # 从 'u0' 提取 0
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # 从 'u1' 提取 1
            else:
                v = int(v)
            if isinstance(t, str) and t.startswith('t'):
                t = int(t[1:])  # 从 't0' 提取 0
            else:
                t = int(t)
            u_node, v_node = int(u), int(v)
            motif_nodes_orig.add(u_node)
            motif_nodes_orig.add(v_node)
            G_motif_template.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t))
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

def multi_motif_judge(context, PREDEFINED_MOTIFS):
    motif_names = set()
    for motif_name, motif_definition in PREDEFINED_MOTIFS.items():
        if judge(context, motif_definition["edge_pattern"], motif_definition["time_window"]) == "Yes":
            motif_names.add(motif_name)
    return list(motif_names)
    


def count_temporal_constraints(mapping_motif_to_context, sorted_motif_events, context_times_all, motif_time_window):
    """
    对于给定的节点映射，找到所有满足时间约束的 motif 实例，
    并以规范化的 frozenset 形式返回。
    """
    found_instances = set() # 使用集合存储找到的唯一实例 (frozenset)
    num_events = len(sorted_motif_events)

    def find_instances_recursive(event_index, current_instance_edges):
        # current_instance_edges: list of (u_c, v_c, t_c, 'a') tuples for the instance being built

        if event_index == num_events:
            # 找到了一个完整的实例，将其规范化并添加到集合中
            normalized_instance = frozenset(
                (min(u, v), max(u, v), t, op) for u, v, t, op in current_instance_edges
            )
            found_instances.add(normalized_instance)
            return

        # 获取当前 motif 事件
        u_m, v_m, t_rel = sorted_motif_events[event_index]
        # 获取对应的 context 节点
        u_c = mapping_motif_to_context.get(u_m)
        v_c = mapping_motif_to_context.get(v_m)
        # (应该检查 u_c, v_c 是否为 None，虽然理论上不会)

        context_edge_topology = tuple(sorted((u_c, v_c)))
        candidate_times = context_times_all.get(context_edge_topology, [])

        last_t = current_instance_edges[-1][2] if current_instance_edges else -1
        t_start = current_instance_edges[0][2] if current_instance_edges else None

        for t_c in candidate_times:
            # 1. 检查时间顺序
            if t_c > last_t:
                # 2. 检查时间窗口
                is_within_window = True
                if t_start is not None and motif_time_window > 0:
                     # 注意：这里的时间窗口定义是 t_last - t_first < W
                     # 如果是 <= W，则条件是 (t_c - t_start) <= motif_time_window
                     # 假设是 < W (严格小于)
                    if (t_c - t_start) > motif_time_window:
                        is_within_window = False

                if is_within_window:
                    # 构建当前事件的 context 边元组
                    current_edge_event = (u_c, v_c, t_c, 'a')
                    # 加入当前实例并递归
                    current_instance_edges.append(current_edge_event)
                    find_instances_recursive(event_index + 1, current_instance_edges)
                    current_instance_edges.pop() # 回溯

    find_instances_recursive(0, [])
    # 返回找到的所有唯一实例 (frozenset 列表)
    # 注意：这里返回的是集合本身，外部调用者将其添加到更大的集合中
    # 或者可以直接返回集合，让外部 update
    return found_instances # 返回包含本次映射找到的所有实例的集合

# --- count 函数主体 ---
def count(context, motif_definition, motif_time_window):
    # 1. 解析 context 和 motif (复用 judge 的逻辑)
    context_a = [e for e in context if len(e) == 4 and e[3] == 'a']
    motif_a = [e for e in motif_definition if len(e) == 4 and e[3] == 'a']
    if not motif_a or not context_a: return 0

    try:
        # 解析 Motif
        G_motif_template = nx.Graph()
        motif_nodes_orig = set()
        temp_motif_events = []
        for u, v, t, op in motif_a:
            if isinstance(u, str) and u.startswith('u'):
                u = int(u[1:])  # 从 'u0' 提取 0
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # 从 'u1' 提取 1
            else:
                v = int(v)
            if isinstance(t, str) and t.startswith('t'):
                t = int(t[1:])  # 从 't0' 提取 0
            else:
                t = int(t)
            u_node, v_node = int(u), int(v)
            motif_nodes_orig.add(u_node)
            motif_nodes_orig.add(v_node)
            G_motif_template.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t))
        sorted_motif_events = sorted(temp_motif_events, key=lambda x: x[2])

        # 解析 Context
        G_context = nx.Graph()
        context_times_all = defaultdict(list)
        context_nodes = set()
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
        print(f"错误 (count): 解析 context 或 motif 失败: {e}")
        return -1 # 返回错误代码

    # 检查节点/边数量，提前退出
    if G_context.number_of_nodes() < G_motif_template.number_of_nodes() or \
       G_context.number_of_edges() < G_motif_template.number_of_edges():
        return 0

    # 2. 初始化同构匹配器和结果集合
    matcher = nx.isomorphism.GraphMatcher(G_context, G_motif_template)
    all_unique_instances = set()

    # 3. 迭代同构映射，调用 count_temporal_constraints 并收集结果
    for mapping_context_to_motif in matcher.subgraph_isomorphisms_iter():
        # 我们需要 motif -> context 的映射
        mapping_motif_to_context = {v: k for k, v in mapping_context_to_motif.items()}

        # 检查是否是完整映射 (以防万一，虽然 subgraph_isomorphisms_iter 应该只返回完整匹配)
        if set(mapping_motif_to_context.keys()) != motif_nodes_orig:
             continue

        # 为当前映射查找所有满足时间约束的实例
        instances_for_this_mapping = count_temporal_constraints(
            mapping_motif_to_context,
            sorted_motif_events,
            context_times_all,
            motif_time_window
        )
        # 将找到的实例（已经是 frozenset）添加到总集合中
        all_unique_instances.update(instances_for_this_mapping)
    print(all_unique_instances)
    # 4. 返回唯一实例的数量
    return len(all_unique_instances)

def multi_motif_counts(context, PREDEFINED_MOTIFS):
    motif_results = {}
    for motif_name, motif_definition in PREDEFINED_MOTIFS.items():
        counts = count(context, motif_definition["edge_pattern"], motif_definition["time_window"])
        if counts != 0:
            motif_results[motif_name] = counts
    return motif_results

# --- Motif And First Time ---
def motif_and_first_time(context, motif_definition, motif_time_window):
    # 1. 解析 context 和 motif (复用 judge 的逻辑)
    context_a = [e for e in context if len(e) == 4 and e[3] == 'a']
    motif_a = [e for e in motif_definition if len(e) == 4 and e[3] == 'a']
    if not motif_a or not context_a: return 0

    try:
        # 解析 Motif
        G_motif_template = nx.Graph()
        motif_nodes_orig = set()
        temp_motif_events = []
        for u, v, t, op in motif_a:
            if isinstance(u, str) and u.startswith('u'):
                u = int(u[1:])  # 从 'u0' 提取 0
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # 从 'u1' 提取 1
            else:
                v = int(v)
            if isinstance(t, str) and t.startswith('t'):
                t = int(t[1:])  # 从 't0' 提取 0
            else:
                t = int(t)
            u_node, v_node = int(u), int(v)
            motif_nodes_orig.add(u_node)
            motif_nodes_orig.add(v_node)
            G_motif_template.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t))
        sorted_motif_events = sorted(temp_motif_events, key=lambda x: x[2])

        # 解析 Context
        G_context = nx.Graph()
        context_times_all = defaultdict(list)
        context_nodes = set()
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
        print(f"错误 (count): 解析 context 或 motif 失败: {e}")
        return -1 # 返回错误代码

    # 检查节点/边数量，提前退出
    if G_context.number_of_nodes() < G_motif_template.number_of_nodes() or \
       G_context.number_of_edges() < G_motif_template.number_of_edges():
        return 0x3f3f3f3f

    # 2. 初始化同构匹配器和结果集合
    matcher = nx.isomorphism.GraphMatcher(G_context, G_motif_template)
    all_unique_instances = set()

    # 3. 迭代同构映射，调用 count_temporal_constraints 并收集结果
    for mapping_context_to_motif in matcher.subgraph_isomorphisms_iter():
        # 我们需要 motif -> context 的映射
        mapping_motif_to_context = {v: k for k, v in mapping_context_to_motif.items()}

        # 检查是否是完整映射 (以防万一，虽然 subgraph_isomorphisms_iter 应该只返回完整匹配)
        if set(mapping_motif_to_context.keys()) != motif_nodes_orig:
             continue

        # 为当前映射查找所有满足时间约束的实例
        instances_for_this_mapping = count_temporal_constraints(
            mapping_motif_to_context,
            sorted_motif_events,
            context_times_all,
            motif_time_window
        )
        # 将找到的实例（已经是 frozenset）添加到总集合中
        all_unique_instances.update(instances_for_this_mapping)
    if len(all_unique_instances) == 0:
        return 0x3f3f3f3f
    # 4. 返回唯一实例的数量
    else:
        # 找到所有实例中最后一条边的最小时间
        min_last_time = 0x3f3f3f3f
        for instance in all_unique_instances:
            # 将实例中的边按时间排序
            sorted_edges = sorted(instance, key=lambda x: x[2])  # x[2]是时间戳
            # 获取最后一条边的时间
            last_edge_time = sorted_edges[-1][2]
            # 更新最小值
            min_last_time = min(min_last_time, last_edge_time)
        return int(min_last_time)

def multi_motif_first_time(context, PREDEFINED_MOTIFS):
    """
    检测动态图中存在的所有motif及其首次完成时间
    
    参数:
        context: 动态图边列表 [(u, v, t, op), ...]
        
    返回:
        包含motif名称和首次完成时间的字典 {motif_name: first_time, ...}
        如果某个motif不存在，则不包含在结果中
    """
    motif_results = {}
    for motif_name, motif_definition in PREDEFINED_MOTIFS.items():
        first_time = motif_and_first_time(context, motif_definition["edge_pattern"], motif_definition["time_window"])
        # 0x3f3f3f3f 表示没有找到该motif的实例
        if first_time != 0x3f3f3f3f:
            motif_results[motif_name] = first_time
    return motif_results


# --- 动态图Motif修改功能：寻找可添加的边使动态图包含特定motif ---
# 
# 这两个函数配合实现：在动态图中找到一条可以添加的边，使得添加后的图包含指定的时序motif
# 
# 算法原理：
# 1. 对于每个motif中的边，假设它缺失，构建fragment（缺失一条边的motif）
# 2. 在context图中寻找fragment的所有匹配实例
# 3. 为每个实例计算缺失边的合适插入时间
# 4. 验证添加边后是否满足时间窗口约束和motif完整性

def find_all_fragment_temporal_instances(map_frag_to_ctx, frag_events, ctx_times, time_window):
    """
    在动态图中寻找所有匹配给定fragment（缺失一条边的motif）的时序实例
    
    参数:
        map_frag_to_ctx: fragment节点到context节点的映射
        frag_events: fragment中的事件序列 [(u_m, v_m, t_rel, orig_idx), ...]
        ctx_times: context图中每条边的时间戳信息 {(u_c, v_c): [t1, t2, ...]}
        time_window: 时间窗口约束
    
    返回:
        所有匹配的时序实例列表
    """
    all_instances = []
    num_frag_events = len(frag_events)

    def find_recursive(frag_idx, current_detail):
        # 递归终止：所有fragment事件都已匹配
        if frag_idx == num_frag_events:
            all_instances.append(list(current_detail))
            return

        # 获取当前fragment事件
        u_m, v_m, t_rel, orig_idx = frag_events[frag_idx]
        # 通过映射获取对应的context节点
        u_c = map_frag_to_ctx.get(u_m)
        v_c = map_frag_to_ctx.get(v_m)
        if u_c is None or v_c is None: 
            return  # 映射无效，不应该发生

        # 获取context中对应边的所有时间戳
        edge_topo = tuple(sorted((u_c, v_c)))
        candidate_times = ctx_times.get(edge_topo, [])
        
        # 获取时间约束：当前序列的最后时间和开始时间
        last_t = current_detail[-1][3] if current_detail else -1
        t_start = current_detail[0][3] if current_detail else None

        # 尝试所有候选时间戳
        for t_c in candidate_times:
            # 检查时间顺序：当前时间必须晚于之前的时间
            if t_c > last_t:
                # 检查时间窗口约束
                if t_start is None or time_window <= 0 or (t_c - t_start) <= time_window:
                    current_detail.append((orig_idx, u_c, v_c, t_c))
                    find_recursive(frag_idx + 1, current_detail)
                    current_detail.pop()  # 回溯

    find_recursive(0, [])
    return all_instances


# --- modify 函数：寻找可添加的边使动态图包含特定motif ---

def modify(context, motif_def, time_window):
    """
    找到一条可以添加的边，使得动态图包含特定的motif
    
    算法思路：
    1. 遍历motif中的每条边，假设缺失这条边
    2. 构建去掉一条边的fragment图
    3. 在context图中寻找fragment的子图同构映射
    4. 对每个映射，寻找fragment的时序实例
    5. 计算应该添加的边的时间戳
    6. 检查添加后是否满足时间窗口约束
    7. 验证添加边后motif是否存在
    
    参数:
        context: 动态图边列表 [(u, v, t, op), ...]
        motif_def: motif定义 [(u, v, t, op), ...]  
        time_window: 时间窗口约束
        
    返回:
        可添加的边 (u, v, t, 'a') 或 None
    """
    # 1. 预处理：只考虑添加操作的边
    ctx_edges = [e for e in context if len(e) == 4 and e[3].lower() == 'a']
    motif_edges = [e for e in motif_def if len(e) == 4 and e[3].lower() == 'a']

    if not motif_edges:
        print("Error (modify): Motif definition has no 'add' edges.")
        return None

    try:
        # 2. 构建motif图和排序事件
        motif_graph = nx.Graph()
        motif_nodes = set()
        temp_motif_events = []
        
        for u, v, t, op in motif_edges:
            if isinstance(u, str) and u.startswith('u'):
                u = int(u[1:])  # 从 'u0' 提取 0
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # 从 'u1' 提取 1
            else:
                v = int(v)
            if isinstance(t, str) and t.startswith('t'):
                t = int(t[1:])  # 从 't0' 提取 0
            else:
                t = int(t)
            u_node, v_node = int(u), int(v)
            motif_nodes.add(u_node)
            motif_nodes.add(v_node)
            motif_graph.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t))
            
        sorted_events = sorted(temp_motif_events, key=lambda x: x[2])
        num_events = len(sorted_events)
        
        # 3. 构建context图和时间信息
        ctx_graph = nx.Graph()
        ctx_times = defaultdict(list)
        
        for u, v, t, op in ctx_edges:
            u_node, v_node = int(u), int(v)
            edge = tuple(sorted((u_node, v_node)))
            ctx_graph.add_edge(u_node, v_node)
            ctx_times[edge].append(int(t))
            
        # 对每条边的时间戳排序
        for edge in ctx_times:
            ctx_times[edge].sort()
            
    except Exception as e:
        print(f"Error (modify): 解析数据失败: {e}")
        return None

    # 4. 遍历每个可能缺失的motif边
    for miss_idx in range(num_events):
        u_m_miss, v_m_miss, t_miss = sorted_events[miss_idx]

        # 5. 构建去掉一条边的fragment图
        frag_graph = motif_graph.copy()
        frag_graph.remove_edge(u_m_miss, v_m_miss)
        print(f"frag_graph.edges(): {frag_graph.edges()}")
        # 构建fragment事件序列（不包含缺失的边）
        frag_events = []
        for idx, (u, v, t_rel) in enumerate(sorted_events):
            if idx != miss_idx:
                frag_events.append((u, v, t_rel, idx))

        # 6. 检查context图是否足够大来容纳fragment
        if ctx_graph.number_of_nodes() < frag_graph.number_of_nodes() or \
           ctx_graph.number_of_edges() < frag_graph.number_of_edges():
            continue

        # 7. 寻找fragment在context中的子图同构映射
        # print("ctx_graph.edges(): ", ctx_graph.edges())
        # print("frag_graph.edges(): ", frag_graph.edges())
        # print("ctx_graph: ", ctx_graph)
        # print("frag_graph: ", frag_graph)
        matcher = nx.isomorphism.GraphMatcher(ctx_graph, frag_graph, edge_match=lambda e1, e2: True)
        
        for ctx_map in matcher.subgraph_isomorphisms_iter():
            # 构建fragment到context的映射
            frag_map = {v: k for k, v in ctx_map.items()}
            print(f"frag_map: {frag_map}")
            if set(frag_map.keys()) != set(frag_graph.nodes()):
                continue

            # 8. 寻找fragment的所有时序实例
            found_instances = find_all_fragment_temporal_instances(
                frag_map,
                frag_events,
                ctx_times,
                time_window
            )
            # print(found_instances)
            # 9. 对每个fragment实例，计算缺失边的插入时间
            for instance_detail in found_instances:
                # 找到缺失边应该插入的时间范围
                t_prev = -1  # 缺失边之前的最晚时间
                t_next = float('inf')  # 缺失边之后的最早时间

                for orig_idx, _, _, t_c in instance_detail:
                    if orig_idx < miss_idx:
                        t_prev = max(t_prev, t_c)
                    elif orig_idx > miss_idx:
                        t_next = min(t_next, t_c)

                # 计算添加边的时间：需要灵活处理各种情况
                possible_times = []
                
                if miss_idx == 0:
                    # 第一条边缺失：尝试多个候选时间
                    if t_next != float('inf'):
                        # 在第一个后续边之前尝试多个时间点
                        for candidate_t in range(max(0, int(t_next) - 10), int(t_next)):
                            if candidate_t >= 0:
                                possible_times.append(candidate_t)
                    else:
                        # 没有后续边，使用默认时间
                        possible_times.append(0)
                else:
                    # 非第一条边缺失：在前后边之间寻找合适时间
                    if t_prev >= 0 and t_next != float('inf'):
                        # 在前后边之间寻找时间
                        for candidate_t in range(t_prev + 1, min(int(t_next), t_prev + 10)):
                            possible_times.append(candidate_t)
                    elif t_prev >= 0:
                        # 只有前边，在其后插入
                        possible_times.append(t_prev + 1)
                    elif t_next != float('inf'):
                        # 只有后边，在其前插入
                        possible_times.append(max(0, t_next - 1))
                
                # 尝试每个候选时间
                for t_add in possible_times:
                    # 检查时间范围有效性
                    if miss_idx > 0 and t_add >= t_next:
                        continue  # 无效时间，跳过
                    
                    # 10. 获取缺失边在context中对应的节点
                    ctx_u = frag_map.get(u_m_miss)
                    ctx_v = frag_map.get(v_m_miss)

                    if ctx_u is None or ctx_v is None:
                        continue

                    # 11. 检查时间窗口约束
                    full_times = [t for _, _, _, t in instance_detail] + [t_add]
                    min_time = min(full_times)
                    max_time = max(full_times)

                    is_within_window = (time_window <= 0) or ((max_time - min_time) <= time_window)

                    if is_within_window:
                        # 构建要添加的边
                        add_edge = tuple(sorted((ctx_u, ctx_v))) + (t_add, 'a')
                        
                        # 12. 验证添加边后motif是否真的存在
                        if judge(context + [add_edge], motif_def, time_window) == "Yes":
                            return add_edge
                                                 # 如果这个时间不行，继续尝试下一个时间

    print("Error (modify): Could not find edge to add. Guarantee violated?")
    return None


def sort_edges(temporal_edges):
    """
    按时间戳对动态图边进行排序
    
    参数:
        temporal_edges: 动态图边列表 [(u, v, t, op), ...]
        
    返回:
        按时间戳排序的边列表
    """
    return sorted(temporal_edges, key=lambda x: x[2])

def when_direct_link(temporal_edges, u, v):
    """
    查找两个节点之间首次连接和首次断开的时间
    
    参数:
        temporal_edges: 动态图边列表 [(u, v, t, op), ...]
        u: 第一个节点
        v: 第二个节点
        
    返回:
        (first_link, first_delete): 首次连接时间和首次断开时间
    """
    edges = sort_edges(temporal_edges)
    first_link = None
    first_delete = None
    
    for edge in edges:
        if edge[0] == u and edge[1] == v and edge[3] == 'a':
            if first_link is None:
                first_link = edge[2]
        if edge[0] == u and edge[1] == v and edge[3] == 'd':
            if first_delete is None:
                first_delete = edge[2]
                
    return first_link, first_delete

def what_edge_at_time(temporal_edges, selected_time):
    """
    查找在指定时间点存在的所有边
    
    参数:
        temporal_edges: 动态图边列表 [(u, v, t, op), ...]
        selected_time: 目标时间戳
        
    返回:
        在selected_time时刻存在的边列表 [(u, v, t_add, 'a'), ...]
    """
    if not temporal_edges:
        return []

    current_edges = {}  # {(u, v): original_add_quadruple}
    temporal_edges = sort_edges(temporal_edges)
    
    # 遍历所有时间 <= selected_time 的事件
    for u, v, t_event, op in temporal_edges:
        if t_event > selected_time:
            break  # 因为已排序，后续事件时间更大，无需处理

        edge = tuple(sorted((u, v)))  # 规范化边表示

        if op == 'a':
            # 记录添加操作，存储原始四元组
            current_edges[edge] = (u, v, t_event, op)
        elif op == 'd':
            # 如果边存在，则删除
            if edge in current_edges:
                del current_edges[edge]

    # 提取在selected_time结束时存在的边的原始四元组
    existing_edges_at_t = list(current_edges.values())
    return existing_edges_at_t

def reverse_graph(temporal_edges):
    """
    反转动态图的操作：将添加操作变为删除操作，删除操作变为添加操作
    
    参数:
        temporal_edges: 动态图边列表 [(u, v, t, op), ...]
        
    返回:
        反转后的边列表
    """
    reverse_edges = []
    for u, v, t, op in temporal_edges:
        if op == 'a':
            reverse_edges.append((u, v, t, 'd'))
        elif op == 'd':
            reverse_edges.append((u, v, t, 'a'))
    # 对反转后的边列表进行整体逆序
    reverse_edges.reverse()
    return reverse_edges
    