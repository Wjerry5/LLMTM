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
    Ported from data_generate.py.

    Args:
        n (int): Number of nodes.
        temporal_edges (list): List of temporal edges [(u, v, t, op), ...].
                               Assumes u, v, t are integers, op is string.
        graph_path (str): Full path to save the output PNG file.
    """
    # Get all unique timestamps and sort them
    timestamps = sorted(list(set(int(edge[2]) for edge in temporal_edges)))
    if not timestamps:
        print(f"Warning (visualize_graph): No valid timestamps found in {graph_path}, cannot generate snapshots.")
        return
    num_snapshots = len(timestamps)

    # Automatically calculate the subplot grid size
    cols = math.ceil(math.sqrt(num_snapshots))
    rows = math.ceil(num_snapshots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows * cols == 1:  # Handle the case of only one snapshot
        axes = np.array([axes])
    # Ensure axes is always iterable, even if it's a single Axes object initially
    axes = axes.flatten()


    # Create a unified layout position - use a complete graph to ensure consistent layout
    pos = nx.spring_layout(nx.complete_graph(n), seed=42)  # Use a fixed seed

    # Set of currently existing edges (normalized: tuple(sorted((u, v))))
    current_edges_normalized = set()
    # Sort temporal_edges by time, if not already sorted
    temporal_edges_sorted = sorted(temporal_edges, key=lambda x: int(x[2]))
    event_idx = 0
    total_events = len(temporal_edges_sorted)

    processed_timestamps = set() # Track processed timestamps to handle multiple events at the same time

    current_t_index = -1
    for t in range(timestamps[-1] + 1): # Iterate through all possible times up to max
        # Process all events at the current timestamp
        while event_idx < total_events and int(temporal_edges_sorted[event_idx][2]) == t:
            u, v, ts_event, op = temporal_edges_sorted[event_idx]
            u_int, v_int = int(u), int(v)
            edge_normalized = tuple(sorted((u_int, v_int)))

            if op == 'a':
                current_edges_normalized.add(edge_normalized)
            elif op == 'd':
                current_edges_normalized.discard(edge_normalized)
            event_idx += 1

        # If the current time t is a timestamp we need to draw a snapshot for
        if t in timestamps and t not in processed_timestamps:
            current_t_index += 1
            processed_timestamps.add(t)

            if current_t_index >= len(axes):
                print(f"Warning (visualize_graph): Number of timestamps {num_snapshots} exceeds number of subplots {len(axes)}.")
                break

            # Create graph snapshot
            G_snapshot = nx.Graph()
            G_snapshot.add_nodes_from(range(n))  # Ensure all nodes exist
            G_snapshot.add_edges_from(current_edges_normalized)

            # Draw subplot
            ax = axes[current_t_index]
            nx.draw(G_snapshot, pos=pos, ax=ax, with_labels=True,
                    node_color='lightblue', node_size=300, font_size=10, edge_color='gray') # Added edge_color
            ax.set_title(f"t = {t}")
            ax.axis('on') # Ensure axis is on to show title properly

            # Add border lines (optional)
            # for spine in ax.spines.values():
            #     spine.set_visible(True)
            #     spine.set_color('gray')
            #     spine.set_linewidth(0.5)

    # Hide extra subplots
    for j in range(current_t_index + 1, len(axes)):
       # Check if the axes object exists and has 'axis' attribute
       if j < len(axes) and hasattr(axes[j], 'axis'):
            axes[j].axis('off')


    try:
        plt.tight_layout()
        # Ensure directory exists
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        plt.savefig(graph_path, dpi=150) # Lower dpi for faster saving
        print(f"  Static graph snapshots saved to: {os.path.basename(graph_path)}")
    except Exception as e:
        print(f"Error (visualize_graph): Error saving image {graph_path}: {e}")
    finally:
        plt.close(fig) # Ensure the figure is closed to free memory


def create_colored_animation(n, temporal_edges, gif_path):
    """
    Generates and saves a GIF animation of the dynamic graph evolution.
    Ported from data_generate.py.

    Args:
        n (int): Number of nodes.
        temporal_edges (list): List of temporal edges [(u, v, t, op), ...].
        gif_path (str): Full path to save the output GIF file.
    """
    timestamps = sorted(list(set(int(edge[2]) for edge in temporal_edges)))
    if not timestamps:
        print(f"Warning (create_colored_animation): No valid timestamps found in {gif_path}, cannot generate animation.")
        return

    # Group events by timestamp
    events_by_timestamp = {}
    for edge in temporal_edges:
        u, v, t, op = edge
        t_int = int(t)
        if t_int not in events_by_timestamp:
            events_by_timestamp[t_int] = []
        events_by_timestamp[t_int].append((int(u), int(v), str(op)))

    # Initialization
    current_edges_normalized = set() # Store normalized edges (u, v)
    all_edges_normalized = set()     # Record all normalized edges that have appeared
    edge_last_op = {}            # Record the last operation for an edge ('a' or 'd')
    edge_first_add_time = {}     # Record the first timestamp an edge was added

    fig, ax = plt.subplots(figsize=(7, 6))

    # Use the same layout method as the static graph
    pos = nx.spring_layout(nx.complete_graph(n), seed=42)

    # Prepare node drawing
    G_nodes_only = nx.Graph()
    G_nodes_only.add_nodes_from(range(n))
    labels = {i: str(i) for i in range(n)}

    # Store edge drawing objects to update alpha
    edge_lines = {} # {(u, v): line_object}

    def update(frame_idx):
        nonlocal current_edges_normalized, all_edges_normalized, edge_last_op, edge_first_add_time
        t = timestamps[frame_idx]

        ax.clear() # Clear the previous frame

        # Process events at the current timestamp
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

        # Draw nodes
        nx.draw_networkx_nodes(G_nodes_only, pos, node_color='skyblue', node_size=300, ax=ax)
        nx.draw_networkx_labels(G_nodes_only, pos, labels=labels, font_size=12, ax=ax)

        # Draw all edges that have appeared
        for edge in all_edges_normalized:
            last_op = edge_last_op.get(edge, 'a') # Default to 'a' if only added
            first_add = edge_first_add_time.get(edge, t) # First add time

            color = 'gray' # Default color
            width = 1.5    # Default width
            alpha = 0.2    # Default transparency (historical trace)

            if edge in current_edges_normalized: # If the edge currently exists
                color = 'green'
                width = 2.5
                alpha = 1.0
            elif last_op == 'd': # If the edge's last state was 'deleted'
                color = 'red'
                width = 2.0
                # Alpha decays over time
                dt = t - first_add # Time since first add
                # (Can adjust decay logic, e.g., decay based on last delete time)
                alpha = max(0.15, 1.0 - 0.1 * dt) # Slightly different decay

            # Draw edge
            nx.draw_networkx_edges(G_nodes_only, pos, edgelist=[edge],
                                  edge_color=color,
                                  width=width, alpha=alpha, ax=ax)

        ax.set_title(f"t = {t}", fontsize=14)

        # Add legend
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Active Edge'),
            Line2D([0], [0], color='red', lw=2, alpha=0.7, label='Deleted Edge Trace'),
            Line2D([0], [0], color='gray', lw=2, alpha=0.3, label='Historical Edge Trace')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.axis('off')

    # Create animation
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        # Interval in milliseconds (e.g., 1000ms = 1 second per frame)
        # FPS determines how fast the GIF plays (e.g., fps=1 means 1 frame per second)
        anim = FuncAnimation(fig, update, frames=len(timestamps),
                             interval=800, repeat=False) # Slower interval
        anim.save(gif_path, writer='pillow', fps=1.25) # Match interval roughly
        print(f"  GIF animation saved to: {os.path.basename(gif_path)}")
    except Exception as e:
        print(f"Error (create_colored_animation): Error saving GIF {gif_path}: {e}")
        # Try saving with a different writer if pillow fails
        try:
            print("Trying to save GIF using 'imagemagick'...")
            anim.save(gif_path, writer='imagemagick', fps=1.25)
            print(f"  GIF animation saved using 'imagemagick' to: {os.path.basename(gif_path)}")
        except Exception as e2:
            print(f"Error (create_colored_animation): Error saving GIF using 'imagemagick' as well: {e2}")
    finally:
        plt.close(fig) # Ensure the figure is closed