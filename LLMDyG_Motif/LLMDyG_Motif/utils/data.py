import numpy as np
import pandas as pd
import os
import torch
from libwon.utils import setup_seed
from collections import Counter, defaultdict
from LLMDyG_Motif.utils.modif_judge_count import judge, check_temporal_constraints
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,"../../data")

import networkx as nx    
import random
from .motif_utils import assign_timestamps
from typing import List, Set, Dict, Tuple, Optional # 添加类型提示
import itertools
import math

def get_sbm_graph(N, pin, pout, C, directed):
    """
    生成一个基于 SBM 模型的图。

    Args:
        N (int): 每个社区的节点数。
        pin (float): 社区内部节点间的连接概率。
        pout (float): 社区之间节点的连接概率。
        C (int): 社区数量。
        directed (bool): 是否生成有向图。

    Returns:
        np.array: 图的边列表，形状为 [E, 2]。
    """
    sizes = [N] * C
    in_prob = pin
    out_prob = pout
    probs = np.zeros((len(sizes), len(sizes)))
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            probs[i, j] = in_prob if i == j else out_prob
    G = nx.stochastic_block_model(sizes, probs, directed = directed)
    edges = [e for e in G.edges()]
    return np.array(edges) # [E, 2]

import numpy as np
class DyGraphGenERCon:
    """
    Generates dynamic graphs based on Erdos-Renyi model with edge additions and deletions.
    Uses np.random.randint for initial timestamp assignment and includes deletion logic
    similar to data_generate.py.
    """
    def sample_dynamic_graph(self, T=5, N=10, p=0.3, directed=False, seed=0):
        """
        Generates a dynamic graph using np.random.randint for addition times
        and includes deletion events.

    Args:
            T (int): Maximum timestamp + 1. Timestamps will be in [0, T-1].
            N (int): Number of nodes.
            p (float): Probability for edge creation in the initial ER graph.
            directed (bool): If True, generates a directed graph. (Currently ignored).
            seed (int): Random seed.

    Returns:
            dict: A dictionary containing graph information.
        """
        random.seed(seed)
        np.random.seed(seed) # Seed numpy as well

        # 1. 使用 ER 模型生成基础图
        G_base = nx.erdos_renyi_graph(N, p, seed=seed)
        initial_edges = list(G_base.edges())
        num_initial_edges = len(initial_edges)

        if num_initial_edges == 0: # Handle case with no initial edges - skip this seed
             return None  # 返回None，让调用方跳过这个种子

        temporal_edges = []

        # 2. 使用 np.random.randint 为所有初始边生成添加时间 t_a
        #    时间戳范围是 [0, T-1]
        addition_times = np.random.randint(0, T, size=num_initial_edges)

        # 3. 处理每条边，生成添加事件，并概率性生成删除事件
        for i, edge in enumerate(initial_edges):
            u, v = edge
            if u > v: u, v = v, u # 规范化

            t_a = addition_times[i] # 获取该边的添加时间
            temporal_edges.append((u, v, t_a, 'a'))

            # 概率性生成删除事件
            # 确保 t_a < T-1 才能在之后删除
            if np.random.random() < 0.5 and t_a < T - 1:
                # 删除时间 t_d > t_a 且 t_d <= T-1
                t_d = np.random.randint(t_a + 1, T)
                temporal_edges.append((u, v, t_d, 'd'))
        # 按时间排序
        temporal_edges.sort(key=lambda x: x[2])
        # 转换为所需格式
        edge_index_list = [(int(u), int(v), int(t), str(op)) for u, v, t, op in temporal_edges]
        edges_array = np.array([(e[0], e[1]) for e in edge_index_list])
        node_set = set(edges_array.flatten()) if edges_array.size > 0 else set()
        num_nodes = len(node_set) if node_set else N 
        num_time = len(set([e[2] for e in edge_index_list]))

        
        info = {
            'T': T, 
            'N': N, 
            'p': p, 
            'seed': seed, 
            'num_nodes': num_nodes,
            'num_time': num_time, 
            'edge_index': edge_index_list,
            'num_edges': len(edge_index_list) # 总事件数
                }
        return info


class DyGraphGenEdge:
    """
    Generates dynamic graphs with fixed number of edges.
    Uses np.random.randint for initial timestamp assignment and includes deletion logic.
    """
    def sample_dynamic_graph(self, T=5, N=10, M=15, directed=False, seed=0):
        """
        Generates a dynamic graph with fixed number of edges.

        Args:
            T (int): Maximum timestamp + 1. Timestamps will be in [0, T-1].
            N (int): Number of nodes.
            M (int): Number of edges to generate.
            directed (bool): If True, generates a directed graph. (Currently ignored).
            seed (int): Random seed.

        Returns:
            dict: A dictionary containing graph information.
        """
        random.seed(seed)
        np.random.seed(seed)

        # 检查边数是否合法
        max_edges = N * (N-1) // 2  # 最大可能的边数
        if M > max_edges:
            print(f"警告：请求的边数{M}超过了最大可能的边数{max_edges}，将使用最大边数")
            M = max_edges

        # 1. 生成固定边数的随机图
        G_base = nx.gnm_random_graph(N, M, seed=seed)
        initial_edges = list(G_base.edges())
        num_initial_edges = len(initial_edges)

        if num_initial_edges == 0:  # Handle case with no edges
            return None

        temporal_edges = []

        # 2. 为所有边生成随机添加时间
        addition_times = np.random.randint(0, T, size=num_initial_edges)

        # 3. 处理每条边，生成添加事件，并概率性生成删除事件
        for i, edge in enumerate(initial_edges):
            u, v = edge
            if u > v: u, v = v, u  # 规范化

            t_a = addition_times[i]  # 获取该边的添加时间
            temporal_edges.append((u, v, t_a, 'a'))

            # 概率性生成删除事件
            if np.random.random() < 0.5 and t_a < T - 1:
                t_d = np.random.randint(t_a + 1, T)
                temporal_edges.append((u, v, t_d, 'd'))

        # 按时间排序
        temporal_edges.sort(key=lambda x: x[2])
        temporal_edges = temporal_edges[:M]
        # 转换为所需格式
        edge_index_list = [(int(u), int(v), int(t), str(op)) for u, v, t, op in temporal_edges]
        edges_array = np.array([(e[0], e[1]) for e in edge_index_list])
        node_set = set(edges_array.flatten()) if edges_array.size > 0 else set()
        num_nodes = len(node_set) if node_set else N 
        num_time = len(set([e[2] for e in edge_index_list]))

        info = {
            'T': T, 
            'N': N, 
            'M': M,  # 记录初始边数而不是概率p
            'seed': seed, 
            'num_nodes': num_nodes,
            'num_time': num_time, 
            'edge_index': edge_index_list,
            'num_edges': len(edge_index_list)  # 总事件数（包括添加和删除）
        }
        return info


class DyGraphGenSBMCon:
    """
    生成一个动态图，边来自初始 SBM 图，随机分配时间戳 t。
    记录为 [u, v, t, 'a'] 和可能的 [u, v, t_delete, 'd']。
    """
    def __init__(self, deletion_prob=0.1):
        self.deletion_prob = deletion_prob

    def sample_dynamic_graph(self, T = 5, N = 10 , p = 0.3, C = 2, directed = False, seed = 0):
        """
        生成动态图。
        Args:
            T (int): 最大时间戳。默认为 5。
            N (int): *总*节点数。默认为 10。
            p (float): SBM 社区内连接概率。默认为 0.3。
            C (int): 社区数量。默认为 2。
            directed (bool): 是否有向图。默认为 False。
            seed (int): 随机种子。默认为 0。
        Returns:
            dict: 包含图信息的字典。
        """
        setup_seed(seed)
        # 1. 生成静态 SBM 图
        initial_edges = get_sbm_graph(N//C, p, p/2, C, directed)
        if initial_edges.shape[0] == 0:
             return None  # 跳过没有边的种子

        # 2. 分配时间戳并生成添加/删除记录
        final_edge_list = []
        timestamps = np.random.randint(0, T, initial_edges.shape[0])
        for i, edge in enumerate(initial_edges):
            u, v = edge
            t_add = timestamps[i]
            final_edge_list.append([int(u), int(v), int(t_add), 'a'])
            if random.random() < self.deletion_prob and t_add < T - 1:
                t_delete = np.random.randint(t_add + 1, T)
                final_edge_list.append([int(u), int(v), int(t_delete), 'd'])

        # 3. 排序
        final_edge_list = sorted(final_edge_list, key=lambda x: (x[2], x[0]))
        
        # 4. 计算统计信息
        if not final_edge_list:
             num_nodes, num_edges, num_time = N, 0, 0
        else:
            edges_array = np.array([(e[0], e[1]) for e in final_edge_list])
            times_array = np.array([e[2] for e in final_edge_list])
            node_set = set(edges_array.flatten()) if edges_array.size > 0 else set()
            num_nodes = len(node_set) if node_set else N
            num_edges = len(final_edge_list)
            time_set = set(times_array.flatten()) if times_array.size > 0 else set()
            num_time = len(time_set) if time_set else 0

        # 5. 准备返回字典 (添加 "C")
        info = {"edge_index": final_edge_list,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_time": num_time,
                "T": T,
                "N": N,
                "p": p,
                "C": C,
                "directed": directed,
                "seed": seed
                }
        return info

from igraph import Graph


class DyGraphGenFFCon:
    """
    生成一个动态图，边来自初始 Forest Fire 图，随机分配时间戳 t。
    记录为 [u, v, t, 'a'] 和可能的 [u, v, t_delete, 'd']。
    """
    def __init__(self, deletion_prob=0.1):
        self.deletion_prob = deletion_prob

    def sample_dynamic_graph(self, T = 5, N = 10 , p = 0.3, directed = False, seed = 0):
        """
        生成动态图。
        Args:
            T (int): 最大时间戳。默认为 5。
            N (int): 节点数。默认为 10。
            p (float): Forest Fire 前向燃烧概率 (fw_prob)。默认为 0.3。
            directed (bool): 是否有向图 (igraph Forest_Fire 默认无向)。默认为 False。
            seed (int): 随机种子。默认为 0。
        Returns:
            dict: 包含图信息的字典。
        """
        setup_seed(seed)
        # 1. 生成 Forest Fire 图
        g = Graph.Forest_Fire(N, fw_prob=p, directed=directed) # 使用 directed 参数
        initial_edges = g.get_edgelist() # [(u, v)]
        if not initial_edges:
            return None  # 跳过没有边的种子
        initial_edges = np.array(initial_edges)

        # 2. 分配时间戳并生成添加/删除记录
        final_edge_list = []
        timestamps = np.random.randint(0, T, initial_edges.shape[0])
        for i, edge in enumerate(initial_edges):
            u, v = edge
            t_add = timestamps[i]
            final_edge_list.append([int(u), int(v), int(t_add), 'a'])
            if random.random() < self.deletion_prob and t_add < T - 1:
                t_delete = np.random.randint(t_add + 1, T)
                final_edge_list.append([int(u), int(v), int(t_delete), 'd'])

        # 3. 排序
        final_edge_list = sorted(final_edge_list, key=lambda x: (x[2], x[0]))

        # 4. 计算统计信息
        if not final_edge_list:
             num_nodes, num_edges, num_time = N, 0, 0
        else:
            edges_array = np.array([(e[0], e[1]) for e in final_edge_list])
            times_array = np.array([e[2] for e in final_edge_list])
            node_set = set(edges_array.flatten()) if edges_array.size > 0 else set()
            num_nodes = len(node_set) if node_set else N
            num_edges = len(final_edge_list)
            time_set = set(times_array.flatten()) if times_array.size > 0 else set()
            num_time = len(time_set) if time_set else 0

        # 5. 准备返回字典
        info = {"edge_index": final_edge_list,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_time": num_time,
                "T": T,
                "N": N,
                "p": p,
                "directed": directed,
                "seed": seed
                }
        return info



class DyGraphGenMotifCon:
    """
    生成用于 Motif 判断任务的动态图。(最终简化版 v5) (中文注释)
    直接生成 Motif 拓扑并随机置换节点，或生成随机 GNM 图。
    然后根据情况分配时间戳 (Case 1/2/3/4)。
    """
    def __init__(self, p_use_random_gnm: float = 0.25, p_permute_motif_nodes: float = 0.5, deletion_prob: float = 0.5):
        """
        初始化 Motif 图生成器。(中文注释)

        Args:
            p_use_random_gnm (float): 使用随机 GNM 图（而非 Motif 拓扑）的概率。
            p_permute_motif_nodes (float): 当生成 Motif 拓扑时，对其节点进行随机置换的概率。
            deletion_prob (float): 对每个 'a' 事件，添加 'd' 事件的概率。
        """
        self.p_use_random_gnm = p_use_random_gnm
        self.p_permute_motif_nodes = p_permute_motif_nodes
        self.deletion_prob = deletion_prob

    def sample_dynamic_graph(self, T: int, predefined_motif: list, motif_time_window: int = 5, seed: int = 0, deletion_prob: float = None, **kwargs) -> dict:
        """
        生成用于 Motif 判断的动态图样本。

        Args:
            T (int): 最大时间戳 + 1。
            predefined_motif (list): 预定义的 Motif 结构 [(u, v, t_rel, 'a'), ...]，事件需要按时间顺序。
            motif_time_window (int): Motif 的最大时间窗口 delta。
            deletion_prob (float, optional): 添加删除事件的概率。如果为 None, 使用类默认值。
            seed (int): 随机种子。
            **kwargs: 允许传入 motif_name 等额外参数 (N, M 被忽略)。

        Returns:
            dict: 包含生成的图信息、Motif 元数据等的字典。
        """
        setup_seed(seed)
        current_deletion_prob = deletion_prob if deletion_prob is not None else self.deletion_prob
        motif_name_from_arg = kwargs.get("motif_name", "custom")
        target_edge_order: List[Tuple[int, int]] = [] # 有序的静态边列表
        is_motif_topology = False
        N_motif = 0
        M_motif = 0 # 拓扑边数
        original_motif = predefined_motif
        # --- Step 1: 解析 Motif 定义 ---
        # 预处理 Motif 定义，将字符串节点标识符转换为整数
        processed_motif = []
        for item in predefined_motif:
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
                processed_motif.append((u, v, t_rel, event_type))
            else:
                processed_motif.append(item)  # 保持原样
        # 更新 predefined_motif 为处理后的版本
        predefined_motif = processed_motif
        motif_nodes_orig = set()
        ordered_motif_events_orig: List[Tuple[int, int]] = [] # 按时间顺序的原始 'a' 事件边列表
        motif_topology_edges = set() # 存储唯一拓扑边
        predefined_motif = sorted(predefined_motif, key=lambda x: x[2])
        print(f"predefined_motif: {predefined_motif}")
        try:
            # 假设 predefined_motif 已经按 t_rel 排序，或者我们按出现顺序处理 'a' 事件
            for item in predefined_motif:
                if len(item) == 4 and item[3] == 'a':
                    u, v = int(item[0]), int(item[1])
                    motif_nodes_orig.add(u)
                    motif_nodes_orig.add(v)
                    edge_orig = tuple(sorted((u, v)))
                    ordered_motif_events_orig.append(edge_orig) # 按事件顺序添加
                    motif_topology_edges.add(edge_orig) # 记录拓扑边
            N_motif = len(motif_nodes_orig)
            M_motif = len(motif_topology_edges) # 唯一拓扑边数
            if N_motif == 0 or M_motif == 0: raise ValueError("Motif definition invalid")
            # 创建从原始节点到 0..N-1 的映射
            node_map_orig_to_new = {orig_node: i for i, orig_node in enumerate(sorted(list(motif_nodes_orig)))}

        except Exception as e:
            print(f"错误: 解析 predefined_motif 失败: {e}")
            return None  # 跳过解析失败的种子

        # --- Step 2 & 3: 概率决策并生成有序静态边列表 ---
        use_random_gnm = (random.random() < self.p_use_random_gnm)
        print(f"use_random_gnm: {use_random_gnm}")
        if use_random_gnm:
            # --- 生成随机 GNM 图 ---
            is_motif_topology = False
            # print("生成随机 GNM 图...") # 中文调试
            try:
                G_gnm = nx.gnm_random_graph(N_motif, M_motif, seed=seed + 1)
                if set(G_gnm.nodes()) != set(range(N_motif)):
                     mapping_relabel = {old_node: new_node for new_node, old_node in enumerate(sorted(G_gnm.nodes()))}
                     G_gnm = nx.relabel_nodes(G_gnm, mapping_relabel)
                target_edge_order = [tuple(sorted(e)) for e in G_gnm.edges()]
            except nx.NetworkXError as e:
                 print(f"错误: 生成 GNM 图失败: {e}。将改为生成 Motif 实例。")
                 use_random_gnm = False # 回退
            if not target_edge_order and not use_random_gnm: # GNM 失败的回退
                 use_random_gnm = False

        if not use_random_gnm:
            # --- 生成 Motif 拓扑实例 (可能带节点置换) ---
            is_motif_topology = True
            # print("生成 Motif 拓扑实例...") # 中文调试

            # 决定是否置换节点
            permute_nodes = (random.random() < self.p_permute_motif_nodes)
            node_map_new_to_final = {} # 从 0..N-1 到最终节点 ID 的映射
            if permute_nodes:
                nodes_list = list(range(N_motif))
                permuted_nodes = list(np.random.permutation(nodes_list))
                node_map_new_to_final = {original: permuted for original, permuted in zip(nodes_list, permuted_nodes)}
            else:
                node_map_new_to_final = {i: i for i in range(N_motif)} # 恒等映射

            # 按 ordered_motif_events_orig 的顺序生成 target_edge_order
            target_edge_order = []
            valid_mapping = True
            for u_orig, v_orig in ordered_motif_events_orig:
                # 映射到 0..N-1
                u_new = node_map_orig_to_new.get(u_orig)
                v_new = node_map_orig_to_new.get(v_orig)
                if u_new is None or v_new is None:
                    print(f"警告: 无法映射原始 Motif 边 ({u_orig}, {v_orig}) 到新节点。")
                    valid_mapping = False; break
                # 应用最终映射 (置换或恒等)
                u_final = node_map_new_to_final.get(u_new)
                v_final = node_map_new_to_final.get(v_new)
                if u_final is None or v_final is None:
                     print(f"警告: 无法应用最终映射到新节点 ({u_new}, {v_new})。")
                     valid_mapping = False; break

                final_edge = tuple(sorted((u_final, v_final)))
                target_edge_order.append(final_edge) # 按原始事件顺序添加

            if not valid_mapping:
                 print("警告: 生成有序边列表时映射失败。返回空。")
                 target_edge_order = [] # 置为空列表


        # --- 健壮性检查: 确保 target_edge_order (作为列表) 存在 ---
        if target_edge_order is None: # 理论上不应发生，除非 GNM 彻底失败
             print("严重警告: target_edge_order 为 None。")
             target_edge_order = []

        # --- Step 4: 分配时间戳 ---
        edge_timestamps, assignment_case = assign_timestamps(
            T, is_motif_topology, target_edge_order, # 传递有序列表
            motif_time_window, seed
        )

        # --- Step 5: 生成带时间戳的 'a' 事件列表 ---
        temporal_add_events = []
        # 直接使用 edge_timestamps 字典生成，它包含了正确的边和时间戳
        for edge_tuple, t_add in edge_timestamps.items():
            u, v = edge_tuple
            temporal_add_events.append([int(u), int(v), int(t_add), 'a'])

        # --- Step 6: 添加删除事件 ---
        final_event_list = list(temporal_add_events)
        for u, v, t_add, op in temporal_add_events:
            if random.random() < current_deletion_prob and t_add < T - 1:
                 if t_add + 1 < T:
                     t_delete = np.random.randint(t_add + 1, T)
                     final_event_list.append([int(u), int(v), int(t_delete), 'd'])

        # --- Step 7: 排序 ---
        final_event_list.sort(key=lambda x: (x[2], x[0], x[1]))

        # --- Step 8: 计算统计信息并返回 ---
        num_events_final = len(final_event_list)
        num_nodes_actual = N_motif
        num_time = 0
        if final_event_list:
             time_set = set(e[2] for e in final_event_list)
             num_time = len(time_set)
        print(f"final_event_list: {final_event_list}")
        info = {
            "N": num_nodes_actual,
            "M": num_events_final,
            "edge_index": final_event_list,
            "num_nodes": num_nodes_actual,
            "num_edges": num_events_final,
            "num_time": num_time,
            "T": T,
            "seed": seed,
            "motif_name": motif_name_from_arg,
            "motif_time_window": motif_time_window,
            "original_motif": original_motif, # 保留用于调试或后续分析
            "is_motif_topology": is_motif_topology,
            "assignment_case": assignment_case
        }

        return info

class DyGraphGenMorMotifCon:

    def sample_dynamic_graph(
        self,
        T_total_time: int,      # 参数 T: 总时间跨度 (0 to T-1)
        N_total_nodes: int,     # 参数 N: 图中总节点数 (0 to N-1)
        M_target_edges: int,    # 参数 M: 最终图中目标总事件数
        W_motif_window: int,    # 参数 W: target_motif的时间窗口长度
        target_motif_definition: List[Tuple[str, str, str, str]], # 例如 ('u0','u1','t0','a')
        seed: 0,
        p_remap_node: float = 0.5, # 概率：对motif的节点进行重映射
        p_delete_bg: float = 0.5,  # 背景事件中'd'操作的概率
        max_overall_attempts: int = 5 # 生成整个有效图的最大尝试次数
    ) -> Optional[Dict]:

        for attempt_idx in range(max_overall_attempts):
            current_run_seed = seed + attempt_idx
            random.seed(current_run_seed)
            np.random.seed(current_run_seed)
            unique_nodes: Set[int] = set()
            processed_motif = [] 

            for item_tuple in target_motif_definition: # item_tuple is like ('u0', 'u1', 't0', 'a')
                if len(item_tuple) != 4: continue   
                u_orig, v_orig, t_placeholder, op_str = item_tuple
                if op_str.lower() != 'a': continue
                u_int_from_def = int(u_orig[1:])
                v_int_from_def = int(v_orig[1:])
                t_int_from_def = int(t_placeholder[1:])
                processed_motif.append((u_int_from_def, v_int_from_def, t_int_from_def, op_str.lower()))
                unique_nodes.add(u_int_from_def)
                unique_nodes.add(v_int_from_def)
            
            if not processed_motif: continue 
            num_nodes_in_motif = len(unique_nodes)
            if N_total_nodes < num_nodes_in_motif: continue

            # 2b. 节点ID重映射: 
            #     以 p_remap_node 的概率将 unique_nodes_from_def_set 中的节点映射到图中的随机节点。
            #     否则，直接使用它们自身作为图节点ID (前提是它们在 N_total_nodes 范围内)。
            # def_node_to_graph_node_map: Key是定义中的整数ID, Value是图中最终的节点ID
            node_map: Dict[int, int] = {}
            available_graph_nodes = list(range(N_total_nodes))
            random.shuffle(available_graph_nodes)
            
            sorted_unique = sorted(list(unique_nodes))

            if random.random() < p_remap_node:
                # 随机映射到图中可用节点的前 N_motif 个
                for i, def_node_id in enumerate(sorted_unique):
                    if i < len(available_graph_nodes): # Ensure we don't go out of bounds for available_graph_nodes
                        node_map[def_node_id] = available_graph_nodes[i]
                    else: 
                        break 
            else:
                # 直接使用定义中的整数ID作为图节点ID
                for def_node_id in sorted_unique:
                    if def_node_id >= N_total_nodes: 
                        break 
                    node_map[def_node_id] = def_node_id
            
            if len(node_map) != num_nodes_in_motif: 
                continue # Mapping was incomplete

            # 2c. 时间戳生成: 为 motif 实例的边在 W_motif_window 内生成时间戳
            t_segment_start = random.randint(0, T_total_time - W_motif_window - 1)
 
            num_motif_edges = len(processed_motif)
            motif_instance_timestamps = sorted(random.sample(range(t_segment_start, t_segment_start + W_motif_window), num_motif_edges))
            print(f"motif_instance_timestamps: {motif_instance_timestamps}")
            motif_instance_timestamps.sort()
            # 构建 motif 实例的边列表 (带真实节点ID和新时间戳)
            # 格式: (u_graph, v_graph, abs_time, op)
            generated_motif_instance_events: List[Tuple[int, int, int, str]] = []
            for i, (u_d, v_d, t, op) in enumerate(processed_motif): # u_d, v_d are nodes from definition
                # Lookup these def_nodes in our map to get their graph IDs
                if u_d not in node_map or v_d not in node_map:
                    continue # Skip this edge if its nodes weren't mapped

                u_g = node_map[u_d]
                v_g = node_map[v_d]
                # 统一节点顺序 u < v
                generated_motif_instance_events.append(tuple(sorted((u_g, v_g))) + (motif_instance_timestamps[i], op))
            
            # --- 去除由 motif 定义、节点映射和时间戳分配共同导致的重复事件 ---
            generated_motif_instance_events = list(set(generated_motif_instance_events))
            # ----------------------------------------------------------------

            if not generated_motif_instance_events : continue
            print(f"generated_motif_instance_events (unique): {generated_motif_instance_events}") # Adjusted print statement
            # --- 步骤 3: 随机删除实例的一条边作为答案 ---
            generated_motif_instance_events.sort(key=lambda x: x[2])
            answer_edge = generated_motif_instance_events[-1]
            current_event_list = [e for e in generated_motif_instance_events if e != answer_edge]
            print(f"current_event_list: {current_event_list}")
            print(f"answer_edge: {answer_edge}")
            # --- 步骤 4: 添加背景边，目标总事件数 M_target_edges ---
            current_event_set: Set[Tuple[int, int, int, str]] = set(current_event_list)
            
            # 循环直到达到 M_target_edges 或尝试次数上限
            # 为了简洁，我们循环固定次数，这个次数基于 M_target_edges
            # 需要添加 M_target_edges - len(current_event_list) 条边
            # 每次有效添加的概率受 judge 和 p_delete_bg 影响
            num_edges_to_add_approx = M_target_edges - len(current_event_list)
            if num_edges_to_add_approx < 0: num_edges_to_add_approx = 0 # 可能初始motif边已很多

            bg_addition_attempts = max(num_edges_to_add_approx * 3, M_target_edges) # 保证一些尝试次数
            active_edges = [e for e in current_event_list if e[3] == 'a']
            for _ in range(bg_addition_attempts):
                if len(current_event_list) >= M_target_edges: break
                if random.random() < p_delete_bg:
                    # 从current_event_list中随机选择一条边
                    # 筛选出第四个元素为'a'且没有被删除过的边
                    if active_edges:
                        random_edge = random.choice(active_edges)
                        bg_u_node, bg_v_node = random_edge[0], random_edge[1]
                        # 从(t, T_total_time-1)范围内随机选择时间
                        bg_time = random.randint(random_edge[2], T_total_time - 1)
                    else:
                        continue
                    current_event_list.append(tuple(sorted((bg_u_node, bg_v_node))) + (bg_time, 'd'))
                    current_event_set.add(tuple(sorted((bg_u_node, bg_v_node))) + (bg_time, 'd'))
                    active_edges.remove(random_edge)
                    print(f"current_event_list: {current_event_list}")
                else:
                    # 随机选择两个节点和时间
                    bg_u_node, bg_v_node = random.sample(range(N_total_nodes), 2)
                    bg_time = random.randint(0, T_total_time - 1)
                    bg_op_rand = 'a'
                    potential_bg_event_tuple = tuple(sorted((bg_u_node, bg_v_node))) + (bg_time, bg_op_rand)
                
                    if potential_bg_event_tuple in current_event_set: continue
                
                    if judge(current_event_list + [potential_bg_event_tuple], processed_motif, W_motif_window) == "Yes":
                            continue
                
                    current_event_list.append(potential_bg_event_tuple)
                    current_event_set.add(potential_bg_event_tuple)
                    active_edges.append(potential_bg_event_tuple)
                    print(f"current_event_list: {current_event_list}")

                if len(current_event_list) >= M_target_edges:
                    break
            
            current_event_list.sort(key=lambda x: (x[2], x[0], x[1], x[3])) # 最终排序
            print(f"current_event_list: {current_event_list}")

            # --- 最后验证 ---
            if judge(current_event_list, processed_motif, W_motif_window) == "Yes":
                print("Yes")
                continue
            if judge(current_event_list + [answer_edge], processed_motif, W_motif_window) != "Yes":
                print("No")
                continue
    
            print(judge(current_event_list, processed_motif, W_motif_window))
            print(judge(current_event_list + [answer_edge], processed_motif, W_motif_window))
            # --- 成功 ---
            final_nodes = set()
            for u,v,t,op in current_event_list: final_nodes.add(u); final_nodes.add(v)
            return {
                "edge_index": current_event_list,  
                "N": N_total_nodes,
                "M": M_target_edges,
                "T": T_total_time, 
                "W": W_motif_window, # For reference
                "num_nodes": len(final_nodes), 
                "num_edges": len(current_event_list),
                "seed": current_run_seed,
                "answer_edge": answer_edge,
                "target_motif": target_motif_definition, # For reference
                "motif_edges": num_motif_edges,
                "motif_nodes": num_nodes_in_motif
                }
            
        return None # 所有尝试失败


PREDEFINED_MOTIFS = {
    "triangle":       {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')], "T": 3},
    "3-star":         {"edge": [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')], "T": 3},
    "4-path":         {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')], "T": 3},
    "4-cycle":        {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')], "T": 6},
    "4-tailedtriangle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')], "T": 6},
    "butterfly":      {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')], "T": 6},
    "4-chordalcycle": {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')], "T": 14},
    "4-clique":       {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')], "T": 15},
    "bitriangle":     {"edge": [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')], "T": 15},
}
from .modif_judge_count import judge

class DyGraphGenControlMotif:
    
    
    def generate_graph_with_motif_control(self,
    M: int,
    motif_name: str,
    seed: int,
    T: int,
) -> List[Tuple[int, int, int, str]]:
    # """
    # 根据 ER 模型生成一个具有 M 条边的时序图。

    # Args:
    #     M (int): 图的总边数。
    #     motif_name (str): 要控制的 Motif 的名称。
    #     seed (int): 随机种子。
    #     T (int): 时间戳的范围 [0, T-1]。

    # Returns:
    #     一个包含 M 个四元组的边列表。
    # """
        random.seed(seed)
        motif_def = PREDEFINED_MOTIFS[motif_name]
        motif_edges_def = motif_def['edge']
        num_motif_edges = len(motif_edges_def)
        
        # 假设节点数 N = M，这通常足够大以避免过于密集的图
        N = M

        # 根据 seed 的奇偶性决定是正例还是负例
        is_positive_example = (seed % 2 == 0)

        if is_positive_example:
            # --- 生成正例 (包含 Motif) ---
            if M < num_motif_edges:
                raise ValueError(f"边数 M={M} 太小，无法容纳一个有 {num_motif_edges} 条边的 '{motif_name}' Motif。")

            final_edges = []
            
            # 1. 植入 Motif
            motif_nodes_str = sorted(list(set(n for e in motif_edges_def for n in e[:2])))
            num_motif_nodes = len(motif_nodes_str)

            if N < num_motif_nodes:
                raise ValueError(f"节点数 N={N} 太小，无法容纳一个有 {num_motif_nodes} 个节点的 '{motif_name}' Motif。")

            # 从图中随机选择节点并映射到 Motif 的抽象节点
            graph_nodes_for_motif = random.sample(range(N), num_motif_nodes)
            node_map = {str_node: real_node for str_node, real_node in zip(motif_nodes_str, graph_nodes_for_motif)}

            # 为 Motif 的边生成符合时序的时间戳
            mt = PREDEFINED_MOTIFS[motif_name]['T']
            t_start = random.randint(0, max(0, T - mt-2))
            motif_timestamps = sorted(random.sample(range(t_start, min(T, t_start + mt)), num_motif_edges))
            planted_edges_set = set()
            for i, (u_str, v_str, _, event_type) in enumerate(motif_edges_def):
                u, v = node_map[u_str], node_map[v_str]
                t = motif_timestamps[i]
                final_edges.append((u, v, t, event_type))
                # 将边标准化存储 (较小的节点在前)
                planted_edges_set.add(tuple(sorted((u, v))))

            # 2. 添加剩余的随机边
            edges_to_add = M - num_motif_edges
            # 创建所有可能的边池，并移除已植入的边和已用过的节点
            # 1. 找出已用过的节点
            used_nodes = set(graph_nodes_for_motif)
            # 2. 构建边池时排除这些节点
            all_possible_edges = set(
                (u, v) for u in range(N) for v in range(u + 1, N)
                if u not in used_nodes and v not in used_nodes
            )
            # 3. 再移除已植入的边
            available_edges = list(all_possible_edges - planted_edges_set)

            if len(available_edges) < edges_to_add:
                raise ValueError(f"图的节点数 N={N} 太小，无法在植入 Motif 后再添加 {edges_to_add} 条不重复的边。")
            
            random_new_edges_a = random.sample(available_edges, int(edges_to_add * 2/3))

            for u, v in random_new_edges_a:
                # 为新边随机分配时间戳
                t = random.randint(0, T - 1)
                final_edges.append((u, v, t, 'a'))
            
            random_new_edges_d = random.sample(final_edges, M - len(final_edges))
            max_time = max(edge[2] for edge in final_edges)
            print(f"max_time: {max_time}")
            for u, v, t0, s0 in random_new_edges_d:
                t = random.randint(max_time, T - 1)
                final_edges.append((u, v, t, 'd'))

            # 最后排序所有边，隐藏植入的 Motif
            final_edges = sorted(final_edges, key=lambda x: (x[2], 0 if x[3] == 'a' else 1))
            info = {
            'T': M, 
            'N': M, 
            'seed': seed, 
            'edge_index': final_edges,
            'num_edges': len(final_edges) # 总事件数
                }
            print("包含Motif")
            print(final_edges)
            print(judge(final_edges,PREDEFINED_MOTIFS[motif_name]['edge'],PREDEFINED_MOTIFS[motif_name]['T']))
            return info

        else:
            # --- 生成负例 (不包含 Motif) ---
            # "生成-测试" 循环
            while True:
                # 1. 利用ER模型生成一个候选图，直接指定边数为M*2/3
                target_edges = math.ceil(M * 2/3)  # 目标边数
                G_base = nx.gnm_random_graph(M, target_edges, seed=seed)
                initial_edges = list(G_base.edges())
                num_initial_edges = len(initial_edges)

                if num_initial_edges == 0: # Handle case with no initial edges - skip this seed
                    return None  # 返回None，让调用方跳过这个种子

                temporal_edges = []
                for u, v in initial_edges:
                    t = random.randint(0, T - 1)
                    temporal_edges.append((u, v, t, 'a'))
                random_new_edges_d = random.sample(temporal_edges, M - num_initial_edges)
                max_time = max(edge[2] for edge in temporal_edges)
                for u, v, t0, s0 in random_new_edges_d:
                    t = random.randint(max_time, T - 1)
                    temporal_edges.append((u, v, t, 'd'))

                final_edges = sorted(temporal_edges, key=lambda x: (x[2], 0 if x[3] == 'a' else 1))
                
                
                # 2. 检查生成的图是否意外包含了 Motif
                if not judge(final_edges,PREDEFINED_MOTIFS[motif_name]['edge'],PREDEFINED_MOTIFS[motif_name]['T']) == "Yes":
                    # 如果不包含，这就是一个合格的负例
                    info = {
                    'T': M, 
                    'N': M, 
                    'seed': seed, 
                    'edge_index': final_edges,
                    'num_edges': len(final_edges) # 总事件数
                    }
                    print("不包含Motif")
                    return info
                else:
                    seed += 1
                    continue
                # 否则，循环将继续，生成并测试一个新的图
                # print(f"  [Debug] Seed {seed}: 生成的图包含 '{motif_name}', 正在重新生成...")




class DyGraphGen:
    def __init__(self, dataset = "enron"):
        if dataset == "enron":
            datafile = os.path.join(dataroot, "enron10/adj_time_list.npy")
            data = np.load(datafile, allow_pickle=True)
            edge_index = [torch.LongTensor(np.array(g.nonzero())) for g in data]
        elif dataset == "dblp":
            datafile = os.path.join(dataroot, "dblp/adj_time_list.npy")
            data = np.load(datafile, allow_pickle=True)
            edge_index = [torch.LongTensor(np.array(g.nonzero())) for g in data]
            
        elif dataset == "flights":
            datafile = os.path.join(dataroot, "Flights/adj_time_list.npy")
            data = np.load(datafile, allow_pickle=True)
            edge_index = [torch.LongTensor(np.array(g.nonzero())) for g in data]
        else:
            raise NotImplementedError(f"{dataset} not implemented")
        self.edge_index = edge_index
    
    def sample_dynamic_graph(self, T = 3, N = 3, seed = 0, undirect = True, **kwargs):
        edge_index = self.edge_index
        setup_seed(seed)
        
        # select time 
        allt = len(edge_index)
        t_start = np.random.choice(np.arange(allt - T - 1))
        t_end = t_start + T
        print(f"sampling time interval [{t_start},{t_end}]")

        # select nodes
        edge3d = turn3dm(edge_index[t_start:t_end])
        if undirect: edge3d = turn_undirect(edge3d)
        edge3d = edge3d.numpy()
        node_set = list(set(edge3d[:,:2].flatten()))
        nodes = set(np.random.choice(node_set, N, replace=False))
        
        # select subgraph
        df = pd.DataFrame(edge3d, columns = "n1 n2 t".split())
        df = df.query("n1 in @nodes or n2 in @nodes").copy()
        org_node_set = set(list(df["n1 n2".split()].values.flatten()))
        node_map = {n:i for i,n in enumerate(list(org_node_set))}
        df['n1'] = df["n1"].apply(lambda x :node_map[x])
        df['n2'] = df["n2"].apply(lambda x :node_map[x])
        edges = df.to_numpy()
        
        # get subgraph info
        num_nodes = len(set(edges[:, :2].flatten()))
        num_edges = len(edges)
        num_time = len(set(edges[:, 2].flatten()))
        ego_nodes = [ node_map[x] for x in list(nodes)]
        info = {"edge_index": edges.tolist(), "num_nodes":num_nodes, "num_edges":num_edges, "num_time": num_time, "ego_nodes":ego_nodes, "T": T, "N":N, "seed":seed,'p':None}
        return info


def turn3dm(edges):
    """
    将一个时间步列表的边（每个时间步是一个 [2, num_edges] 的张量）
    转换为一个包含所有边的 [total_num_edges, 3] 张量，其中第三列是时间步索引。
    """
    es = []
    for i,e in enumerate(edges):
        if e.numel() > 0: # 检查张量是否为空
             # 确保 e 是 LongTensor
             if not isinstance(e, torch.LongTensor):
                  e = e.long()
             # 添加时间戳列
             time_col = torch.full((e.shape[1], 1), i, dtype=torch.long, device=e.device)
             e_with_time = torch.cat([e.t(), time_col], dim=1)
             es.append(e_with_time)
    if not es: # 如果所有时间步都为空
        return torch.empty((0, 3), dtype=torch.long)
    es = torch.cat(es, dim = 0)
    return es

def turn_undirect(edges_tensor):
    """
    处理一个包含 (u, v, t) 元组的边张量，去除重复的无向边。
    只保留 u <= v 的表示。
    """
    if edges_tensor.numel() == 0:
        return edges_tensor

    # 确保 u <= v
    u = torch.min(edges_tensor[:, 0], edges_tensor[:, 1])
    v = torch.max(edges_tensor[:, 0], edges_tensor[:, 1])
    t = edges_tensor[:, 2]

    # 组合成新的张量并去重
    unique_edges = torch.unique(torch.stack([u, v, t], dim=1), dim=0)
    return unique_edges