import random
import numpy as np
from libwon.utils import setup_seed
from typing import List, Dict, Tuple, Optional # 添加类型提示

def assign_timestamps(T: int,
                      is_motif_topology: bool,
                      target_edge_order: List[Tuple[int, int]], # 现在接收的是有序列表
                      motif_time_window: int,
                      seed: int) -> Tuple[Dict[Tuple[int, int], int], int]:
    """
    为图中的边分配时间戳。(最终简化版) (中文注释)
    如果 is_motif_topology=True, 概率性生成 Case 1/2/3 时间戳, 使用 target_edge_order 顺序。
    如果 is_motif_topology=False, 随机分配时间戳 (Case 4)。

    Args:
        T (int): 最大时间戳 + 1 (时间范围 [0, T-1])。
        is_motif_topology (bool): 输入图的拓扑是否与 Motif 同构。
        target_edge_order (list): 图中边的有序列表。对于 Motif 拓扑，
                                  此顺序反映了 Motif 的时间顺序；对于随机图，
                                  此顺序无关紧要（但仍用于迭代）。
        motif_time_window (int): Motif 定义的最大时间窗口 delta。
        seed (int): 随机种子。

    Returns:
        tuple: (edge_timestamps (dict), assignment_case (int))
    """
    setup_seed(seed + 2)
    edge_timestamps = {}
    assignment_case = 4 # 默认 Case 4

    if not target_edge_order: # 列表为空
        return edge_timestamps, assignment_case

    num_events_to_assign = len(target_edge_order) # 事件数/边数

    # --- 情况 A: 拓扑与 Motif 不同构 (或者随机 GNM 图) ---
    if not is_motif_topology:
        # print("Case 4: 拓扑不同构或随机 GNM，随机分配时间戳。") # 中文调试
        timestamps = np.random.randint(0, T, num_events_to_assign)
        # 分配给 target_edge_order 中的所有边 (顺序无关紧要)
        for i, edge in enumerate(target_edge_order):
            edge_timestamps[edge] = timestamps[i]
        return edge_timestamps, 4

    # --- 情况 B: 拓扑与 Motif 同构 ---
    # 概率性决定目标 Case 并尝试生成时间戳 (逻辑不变)
    assign_prob = random.random()
    target_case = 4
    prob_case1 = 0.33
    prob_case2 = 0.33
    prob_case3 = 0.34
    if assign_prob < prob_case1: target_case = 1
    elif assign_prob < prob_case1 + prob_case2: target_case = 2
    else: target_case = 3
    print(f"target_case: {target_case}")
    final_times = []

    candidate_times = []
    is_ordered = False
    in_window = False

    generated_successfully = False
    base_times = [] # 存储 Case 1 的基础时间序列
    try: 
        t_start = random.randint(0, T - motif_time_window - 2)
        print(T) 
        print(motif_time_window)
        print(T - motif_time_window - 2)
        possible_times = list(range(t_start+1, t_start + motif_time_window))
        sampled_times = random.sample(possible_times, num_events_to_assign-1)
        sampled_times.append(t_start)
        base_times = sorted(sampled_times)
    except ValueError as e:
        print(f"生成 Case 1 基础序列时发生错误: {e}")
        base_times = [] # 生成失败
    print(f"base_times: {base_times}")
# 如果基础序列生成失败，则无法生成 Case 1, 2, 3
    if not base_times:
        target_case = 4 # 强制 Fallback
    else:
        if target_case == 1:
            final_times = base_times
            assignment_case = 1
            generated_successfully = True
        elif target_case == 3:
            shuffled_times = list(base_times) # 复制
            original_sorted = list(base_times) # 保留原始排序
            attempts = 0
            max_shuffle_attempts = 5
            is_actually_shuffled = False
            while attempts < max_shuffle_attempts:
                attempts += 1
                random.shuffle(shuffled_times)
                print(f"shuffled_times: {shuffled_times}")
                # 检查是否真的不再严格有序
                is_shuffled_strictly_ordered = all(shuffled_times[i] < shuffled_times[i+1] for i in range(num_events_to_assign-1))
                if not is_shuffled_strictly_ordered:
                    final_times = shuffled_times
                    assignment_case = 3
                    generated_successfully = True
                    # print(f"成功打乱 (尝试 {attempts} 次)。") # 中文调试
                    break # 成功打乱，退出循环
            if not generated_successfully:
                print("多次打乱仍无法破坏严格顺序，Fallback 到 Case 4。")
                target_case = 4 
        elif target_case == 2:
            attempts = 0
            max_construct_attempts = 5 # 尝试几次构造
            while attempts < max_construct_attempts and not generated_successfully:
                attempts += 1
                print("attempt: ", attempts)
                try:
                    t_min = base_times[0]
                    lower_bound_tmax = max(t_min + motif_time_window+1, base_times[-2]+1)
                    upper_bound_tmax = T - 1
                
                    t_max = random.randint(lower_bound_tmax, upper_bound_tmax)

                    # 3. 确定中间点
                    intermediate_times = []
                    possible_intermediate = list(range(t_min + 1, t_max))
                    intermediate_times = random.sample(possible_intermediate, num_events_to_assign - 2)
                    # 4. 组合并排序
                    candidate_final_times = sorted([t_min] + intermediate_times + [t_max])

                    # 5. 验证 (理论上应该满足)
                    final_span = candidate_final_times[-1] - candidate_final_times[0]
                    print(f"candidate_final_times: {candidate_final_times}")
                    print(f"final_span: {final_span}")
                    if final_span > motif_time_window:
                        final_times = candidate_final_times
                        assignment_case = 2
                        generated_successfully = True
                        break 
                    else: 
                        print(f"Case 2 构造验证失败: ordered={final_is_ordered}, span={final_span}>{motif_time_window}")
                    if not generated_successfully:
                        print(f"Case 2 构造验证失败: ordered={final_is_ordered}, span={final_span}>{motif_time_window}")
                except ValueError as e: 
                    continue # 重试

    if not generated_successfully:
        # print(f"Fallback 到 Case 4 (目标: {target_case}, 成功: {generated_successfully})") # 中文调试
        assignment_case = 4
        timestamps = np.random.randint(0, T, num_events_to_assign)
        final_times = list(timestamps) 
    # --- 分配最终的时间戳 ---
    # 使用 target_edge_order 分配
    for i, edge in enumerate(target_edge_order):
        # 确保 final_times 列表长度正确
        if i < len(final_times):
             edge_timestamps[edge] = final_times[i]
        else:
             # 这理论上不应发生，但作为保险
             print(f"警告: final_times 列表长度 ({len(final_times)}) 短于 target_edge_order ({len(target_edge_order)})。")
             # 可以分配一个默认值或最后一个值
             edge_timestamps[edge] = final_times[-1] if final_times else random.randint(0, T-1)


    return edge_timestamps, assignment_case
