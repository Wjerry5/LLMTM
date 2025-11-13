import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import math
import numpy as np
import os

def visualize_graph(n, temporal_edges, graph_path):
    """
    Generates and saves a grid of static graph snapshots for each timestamp.
    移植自 data_generate.py。

    Args:
        n (int): Number of nodes.
        temporal_edges (list): List of temporal edges [(u, v, t, op), ...].
                               Assumes u, v, t are integers, op is string.
        graph_path (str): Full path to save the output PNG file.
    """
    # 获取所有唯一的时间戳并排序
    timestamps = sorted(list(set(int(edge[2]) for edge in temporal_edges)))
    if not timestamps:
        print(f"警告 (visualize_graph): 在 {graph_path} 中没有找到有效的时间戳，无法生成快照。")
        return
    num_snapshots = len(timestamps)

    # 自动计算子图网格大小
    cols = math.ceil(math.sqrt(num_snapshots))
    rows = math.ceil(num_snapshots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows * cols == 1:  # 处理只有一个快照的情况
        axes = np.array([axes])
    # Ensure axes is always iterable, even if it's a single Axes object initially
    axes = axes.flatten()


    # 创建统一的布局位置 - 使用完全图确保布局一致性
    pos = nx.spring_layout(nx.complete_graph(n), seed=42)  # 使用固定种子

    # 当前存在的边集合 (规范化: tuple(sorted((u, v))))
    current_edges_normalized = set()
    # 将 temporal_edges 按时间排序，如果尚未排序
    temporal_edges_sorted = sorted(temporal_edges, key=lambda x: int(x[2]))
    event_idx = 0
    total_events = len(temporal_edges_sorted)

    processed_timestamps = set() # Track processed timestamps to handle multiple events at the same time

    current_t_index = -1
    for t in range(timestamps[-1] + 1): # Iterate through all possible times up to max
        # 处理当前时间戳的所有事件
        while event_idx < total_events and int(temporal_edges_sorted[event_idx][2]) == t:
            u, v, ts_event, op = temporal_edges_sorted[event_idx]
            u_int, v_int = int(u), int(v)
            edge_normalized = tuple(sorted((u_int, v_int)))

            if op == 'a':
                current_edges_normalized.add(edge_normalized)
            elif op == 'd':
                current_edges_normalized.discard(edge_normalized)
            event_idx += 1

        # 如果当前时间 t 是我们需要绘制快照的时间点
        if t in timestamps and t not in processed_timestamps:
            current_t_index += 1
            processed_timestamps.add(t)

            if current_t_index >= len(axes):
                print(f"警告 (visualize_graph): 时间戳数量 {num_snapshots} 超出子图数量 {len(axes)}。")
                break

            # 创建图快照
            G_snapshot = nx.Graph()
            G_snapshot.add_nodes_from(range(n))  # 确保所有节点都存在
            G_snapshot.add_edges_from(current_edges_normalized)

            # 绘制子图
            ax = axes[current_t_index]
            nx.draw(G_snapshot, pos=pos, ax=ax, with_labels=True,
                    node_color='lightblue', node_size=300, font_size=10, edge_color='gray') # Added edge_color
            ax.set_title(f"t = {t}")
            ax.axis('on') # Ensure axis is on to show title properly

            # 添加边框线 (可选)
            # for spine in ax.spines.values():
            #     spine.set_visible(True)
            #     spine.set_color('gray')
            #     spine.set_linewidth(0.5)

    # 隐藏多余子图
    for j in range(current_t_index + 1, len(axes)):
         # Check if the axes object exists and has 'axis' attribute
         if j < len(axes) and hasattr(axes[j], 'axis'):
              axes[j].axis('off')


    try:
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        plt.savefig(graph_path, dpi=150) # Lower dpi for faster saving
        print(f"  静态图快照已保存至: {os.path.basename(graph_path)}")
    except Exception as e:
        print(f"错误 (visualize_graph): 保存图像时出错 {graph_path}: {e}")
    finally:
        plt.close(fig) # 确保关闭图形以释放内存


def create_colored_animation(n, temporal_edges, gif_path):
    """
    Generates and saves a GIF animation of the dynamic graph evolution.
    移植自 data_generate.py。

    Args:
        n (int): Number of nodes.
        temporal_edges (list): List of temporal edges [(u, v, t, op), ...].
        gif_path (str): Full path to save the output GIF file.
    """
    timestamps = sorted(list(set(int(edge[2]) for edge in temporal_edges)))
    if not timestamps:
        print(f"警告 (create_colored_animation): 在 {gif_path} 中没有找到有效的时间戳，无法生成动画。")
        return

    # 按时间戳将事件分组
    events_by_timestamp = {}
    for edge in temporal_edges:
        u, v, t, op = edge
        t_int = int(t)
        if t_int not in events_by_timestamp:
            events_by_timestamp[t_int] = []
        events_by_timestamp[t_int].append((int(u), int(v), str(op)))

    # 初始化
    current_edges_normalized = set() # 存储规范化边 (u, v)
    all_edges_normalized = set()     # 记录所有出现过的规范化边
    edge_last_op = {}            # 记录边的最后操作 ('a' or 'd')
    edge_first_add_time = {}     # 记录边首次添加的时间戳

    fig, ax = plt.subplots(figsize=(7, 6))

    # 使用与静态图相同的布局方法
    pos = nx.spring_layout(nx.complete_graph(n), seed=42)

    # 准备节点绘制
    G_nodes_only = nx.Graph()
    G_nodes_only.add_nodes_from(range(n))
    labels = {i: str(i) for i in range(n)}

    # 存储边绘制对象以便更新 alpha
    edge_lines = {} # {(u, v): line_object}

    def update(frame_idx):
        nonlocal current_edges_normalized, all_edges_normalized, edge_last_op, edge_first_add_time
        t = timestamps[frame_idx]

        ax.clear() # 清除上一帧

        # 处理当前时间戳的事件
        if t in events_by_timestamp:
            for u, v, op in events_by_timestamp[t]:
                edge_normalized = tuple(sorted((u, v)))
                all_edges_normalized.add(edge_normalized)
                edge_last_op[edge_normalized] = op
                if op == 'a':
                    current_edges_normalized.add(edge_normalized)
                    if edge_normalized not in edge_first_add_time:
                        edge_first_add_time[edge_normalized] = t
                elif op == 'd':
                    current_edges_normalized.discard(edge_normalized)

        # 绘制节点
        nx.draw_networkx_nodes(G_nodes_only, pos, node_color='skyblue', node_size=300, ax=ax)
        nx.draw_networkx_labels(G_nodes_only, pos, labels=labels, font_size=12, ax=ax)

        # 绘制所有出现过的边
        for edge in all_edges_normalized:
            last_op = edge_last_op.get(edge, 'a') # 默认为 'a' 如果只有添加
            first_add = edge_first_add_time.get(edge, t) # 首次添加时间

            color = 'gray' # 默认颜色
            width = 1.5    # 默认宽度
            alpha = 0.2    # 默认透明度 (历史轨迹)

            if edge in current_edges_normalized: # 如果边当前存在
                color = 'green'
                width = 2.5
                alpha = 1.0
            elif last_op == 'd': # 如果边最后是删除状态
                color = 'red'
                width = 2.0
                # 透明度随时间衰减
                dt = t - first_add # 距离首次添加的时间
                # (可以调整衰减逻辑，例如基于最后删除时间衰减)
                alpha = max(0.15, 1.0 - 0.1 * dt) # 稍微不同的衰减

            # 绘制边
            nx.draw_networkx_edges(G_nodes_only, pos, edgelist=[edge],
                                   edge_color=color,
                                   width=width, alpha=alpha, ax=ax)

        ax.set_title(f"t = {t}", fontsize=14)

        # 添加图例
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Active Edge'),
            Line2D([0], [0], color='red', lw=2, alpha=0.7, label='Deleted Edge Trace'),
            Line2D([0], [0], color='gray', lw=2, alpha=0.3, label='Historical Edge Trace')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.axis('off')

    # 创建动画
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        # Interval in milliseconds (e.g., 1000ms = 1 second per frame)
        # FPS determines how fast the GIF plays (e.g., fps=1 means 1 frame per second)
        anim = FuncAnimation(fig, update, frames=len(timestamps),
                             interval=800, repeat=False) # Slower interval
        anim.save(gif_path, writer='pillow', fps=1.25) # Match interval roughly
        print(f"  GIF 动画已保存至: {os.path.basename(gif_path)}")
    except Exception as e:
        print(f"错误 (create_colored_animation): 保存 GIF 时出错 {gif_path}: {e}")
        # Try saving with a different writer if pillow fails
        try:
            print("尝试使用 'imagemagick' 保存 GIF...")
            anim.save(gif_path, writer='imagemagick', fps=1.25)
            print(f"  GIF 动画已使用 'imagemagick' 保存至: {os.path.basename(gif_path)}")
        except Exception as e2:
            print(f"错误 (create_colored_animation): 使用 'imagemagick' 保存 GIF 时也出错: {e2}")
    finally:
        plt.close(fig) # 确保关闭图形

# New file: LLMDyG_Motif/utils/visualization.py
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import math
import numpy as np
import os

def visualize_graph(n, temporal_edges, graph_path):
    """
    Generates and saves a grid of static graph snapshots for each timestamp.
    移植自 data_generate.py。

    Args:
        n (int): Number of nodes.
        temporal_edges (list): List of temporal edges [(u, v, t, op), ...].
                               Assumes u, v, t are integers, op is string.
        graph_path (str): Full path to save the output PNG file.
    """
    # 获取所有唯一的时间戳并排序
    timestamps = sorted(list(set(int(edge[2]) for edge in temporal_edges)))
    if not timestamps:
        print(f"警告 (visualize_graph): 在 {graph_path} 中没有找到有效的时间戳，无法生成快照。")
        return
    num_snapshots = len(timestamps)

    # 自动计算子图网格大小
    cols = math.ceil(math.sqrt(num_snapshots))
    rows = math.ceil(num_snapshots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows * cols == 1:  # 处理只有一个快照的情况
        axes = np.array([axes])
    # Ensure axes is always iterable, even if it's a single Axes object initially
    axes = axes.flatten()


    # 创建统一的布局位置 - 使用完全图确保布局一致性
    pos = nx.spring_layout(nx.complete_graph(n), seed=42)  # 使用固定种子

    # 当前存在的边集合 (规范化: tuple(sorted((u, v))))
    current_edges_normalized = set()
    # 将 temporal_edges 按时间排序，如果尚未排序
    temporal_edges_sorted = sorted(temporal_edges, key=lambda x: int(x[2]))
    event_idx = 0
    total_events = len(temporal_edges_sorted)

    processed_timestamps = set() # Track processed timestamps to handle multiple events at the same time

    current_t_index = -1
    for t in range(timestamps[-1] + 1): # Iterate through all possible times up to max
        # 处理当前时间戳的所有事件
        while event_idx < total_events and int(temporal_edges_sorted[event_idx][2]) == t:
            u, v, ts_event, op = temporal_edges_sorted[event_idx]
            u_int, v_int = int(u), int(v)
            edge_normalized = tuple(sorted((u_int, v_int)))

            if op == 'a':
                current_edges_normalized.add(edge_normalized)
            elif op == 'd':
                current_edges_normalized.discard(edge_normalized)
            event_idx += 1

        # 如果当前时间 t 是我们需要绘制快照的时间点
        if t in timestamps and t not in processed_timestamps:
            current_t_index += 1
            processed_timestamps.add(t)

            if current_t_index >= len(axes):
                print(f"警告 (visualize_graph): 时间戳数量 {num_snapshots} 超出子图数量 {len(axes)}。")
                break

            # 创建图快照
            G_snapshot = nx.Graph()
            G_snapshot.add_nodes_from(range(n))  # 确保所有节点都存在
            G_snapshot.add_edges_from(current_edges_normalized)

            # 绘制子图
            ax = axes[current_t_index]
            nx.draw(G_snapshot, pos=pos, ax=ax, with_labels=True,
                    node_color='lightblue', node_size=300, font_size=10, edge_color='gray') # Added edge_color
            ax.set_title(f"t = {t}")
            ax.axis('on') # Ensure axis is on to show title properly

            # 添加边框线 (可选)
            # for spine in ax.spines.values():
            #     spine.set_visible(True)
            #     spine.set_color('gray')
            #     spine.set_linewidth(0.5)

    # 隐藏多余子图
    for j in range(current_t_index + 1, len(axes)):
         # Check if the axes object exists and has 'axis' attribute
         if j < len(axes) and hasattr(axes[j], 'axis'):
              axes[j].axis('off')


    try:
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        plt.savefig(graph_path, dpi=150) # Lower dpi for faster saving
        print(f"  静态图快照已保存至: {os.path.basename(graph_path)}")
    except Exception as e:
        print(f"错误 (visualize_graph): 保存图像时出错 {graph_path}: {e}")
    finally:
        plt.close(fig) # 确保关闭图形以释放内存


def create_colored_animation(n, temporal_edges, gif_path):
    """
    Generates and saves a GIF animation of the dynamic graph evolution.
    移植自 data_generate.py。

    Args:
        n (int): Number of nodes.
        temporal_edges (list): List of temporal edges [(u, v, t, op), ...].
        gif_path (str): Full path to save the output GIF file.
    """
    timestamps = sorted(list(set(int(edge[2]) for edge in temporal_edges)))
    if not timestamps:
        print(f"警告 (create_colored_animation): 在 {gif_path} 中没有找到有效的时间戳，无法生成动画。")
        return

    # 按时间戳将事件分组
    events_by_timestamp = {}
    for edge in temporal_edges:
        u, v, t, op = edge
        t_int = int(t)
        if t_int not in events_by_timestamp:
            events_by_timestamp[t_int] = []
        events_by_timestamp[t_int].append((int(u), int(v), str(op)))

    # 初始化
    current_edges_normalized = set() # 存储规范化边 (u, v)
    all_edges_normalized = set()     # 记录所有出现过的规范化边
    edge_last_op = {}            # 记录边的最后操作 ('a' or 'd')
    edge_first_add_time = {}     # 记录边首次添加的时间戳

    fig, ax = plt.subplots(figsize=(7, 6))

    # 使用与静态图相同的布局方法
    pos = nx.spring_layout(nx.complete_graph(n), seed=42)

    # 准备节点绘制
    G_nodes_only = nx.Graph()
    G_nodes_only.add_nodes_from(range(n))
    labels = {i: str(i) for i in range(n)}

    # 存储边绘制对象以便更新 alpha
    edge_lines = {} # {(u, v): line_object}

    def update(frame_idx):
        nonlocal current_edges_normalized, all_edges_normalized, edge_last_op, edge_first_add_time
        t = timestamps[frame_idx]

        ax.clear() # 清除上一帧

        # 处理当前时间戳的事件
        if t in events_by_timestamp:
            for u, v, op in events_by_timestamp[t]:
                edge_normalized = tuple(sorted((u, v)))
                all_edges_normalized.add(edge_normalized)
                edge_last_op[edge_normalized] = op
                if op == 'a':
                    current_edges_normalized.add(edge_normalized)
                    if edge_normalized not in edge_first_add_time:
                        edge_first_add_time[edge_normalized] = t
                elif op == 'd':
                    current_edges_normalized.discard(edge_normalized)

        # 绘制节点
        nx.draw_networkx_nodes(G_nodes_only, pos, node_color='skyblue', node_size=300, ax=ax)
        nx.draw_networkx_labels(G_nodes_only, pos, labels=labels, font_size=12, ax=ax)

        # 绘制所有出现过的边
        for edge in all_edges_normalized:
            last_op = edge_last_op.get(edge, 'a') # 默认为 'a' 如果只有添加
            first_add = edge_first_add_time.get(edge, t) # 首次添加时间

            color = 'gray' # 默认颜色
            width = 1.5    # 默认宽度
            alpha = 0.2    # 默认透明度 (历史轨迹)

            if edge in current_edges_normalized: # 如果边当前存在
                color = 'green'
                width = 2.5
                alpha = 1.0
            elif last_op == 'd': # 如果边最后是删除状态
                color = 'red'
                width = 2.0
                # 透明度随时间衰减
                dt = t - first_add # 距离首次添加的时间
                # (可以调整衰减逻辑，例如基于最后删除时间衰减)
                alpha = max(0.15, 1.0 - 0.1 * dt) # 稍微不同的衰减

            # 绘制边
            nx.draw_networkx_edges(G_nodes_only, pos, edgelist=[edge],
                                   edge_color=color,
                                   width=width, alpha=alpha, ax=ax)

        ax.set_title(f"t = {t}", fontsize=14)

        # 添加图例
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Active Edge'),
            Line2D([0], [0], color='red', lw=2, alpha=0.7, label='Deleted Edge Trace'),
            Line2D([0], [0], color='gray', lw=2, alpha=0.3, label='Historical Edge Trace')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.axis('off')

    # 创建动画
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        # Interval in milliseconds (e.g., 1000ms = 1 second per frame)
        # FPS determines how fast the GIF plays (e.g., fps=1 means 1 frame per second)
        anim = FuncAnimation(fig, update, frames=len(timestamps),
                             interval=800, repeat=False) # Slower interval
        anim.save(gif_path, writer='pillow', fps=1.25) # Match interval roughly
        print(f"  GIF 动画已保存至: {os.path.basename(gif_path)}")
    except Exception as e:
        print(f"错误 (create_colored_animation): 保存 GIF 时出错 {gif_path}: {e}")
        # Try saving with a different writer if pillow fails
        try:
            print("尝试使用 'imagemagick' 保存 GIF...")
            anim.save(gif_path, writer='imagemagick', fps=1.25)
            print(f"  GIF 动画已使用 'imagemagick' 保存至: {os.path.basename(gif_path)}")
        except Exception as e2:
            print(f"错误 (create_colored_animation): 使用 'imagemagick' 保存 GIF 时也出错: {e2}")
    finally:
        plt.close(fig) # 确保关闭图形 