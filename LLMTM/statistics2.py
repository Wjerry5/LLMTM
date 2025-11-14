import networkx as nx
import os 
import json 
from tqdm import tqdm 

from scipy.stats import entropy
from collections import Counter, defaultdict
import pandas as pd

import numpy as np
from collections import Counter, defaultdict

def analyze_graph_features(edges):
    
    G = nx.Graph()
    active_edges = [edge for edge in edges if edge[3] == 'a']
    
    if not active_edges:
        return {
            "num_nodes": 0, "num_edges": 0, "edge_density": 0,
            "num_nodes_ge_3": 0, "num_nodes_eq_2": 0, "cyclomatic_complexity": 1,
            "edge_locality_score": 0, "temporal_clash_count": 0, "ratio_nodes_ge_3": 0, "ratio_nodes_eq_2": 0
        }
    
    edge_tuples = [(u, v) for u, v, t, op in active_edges]
    G.add_edges_from(edge_tuples)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()


    if num_nodes > 1:
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        edge_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    else:
        edge_density = 0
        
    # (E - N + P)
    num_connected_components = nx.number_connected_components(G)
    cyclomatic_complexity = num_edges - num_nodes + num_connected_components

    degrees = dict(G.degree())
    num_nodes_ge_3 = sum(1 for degree in degrees.values() if degree >= 3)
    num_nodes_eq_2 = sum(1 for degree in degrees.values() if degree == 2)
    ratio_nodes_ge_3 = num_nodes_ge_3/num_nodes
    ratio_nodes_eq_2 = num_nodes_eq_2/num_nodes
    
    core_nodes = [node for node, degree in degrees.items() if degree >= 3]

    node_to_edge_info = defaultdict(list)
    for i, (u, v, t, op) in enumerate(edges):
        if op == 'a':
            info = {'t': t, 'idx': i}
            node_to_edge_info[u].append(info)
            if u != v:
                node_to_edge_info[v].append(info)

    center_locality_scores = []
    for node in core_nodes:
        indices = [info['idx'] for info in node_to_edge_info[node]]
        if len(indices) >= 2:
            center_locality_scores.append(np.std(indices))
    
    edge_locality_score = np.mean(center_locality_scores) if center_locality_scores else 0

    total_clashes = 0
    for node in core_nodes:
        timestamps = [info['t'] for info in node_to_edge_info[node]]
        if not timestamps:
            continue
        time_counts = Counter(timestamps)
        for count in time_counts.values():
            if count > 1:
                total_clashes += count * (count - 1) / 2
    temporal_clash_count = total_clashes

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "edge_density": edge_density,
        "num_nodes_ge_3": num_nodes_ge_3,
        "num_nodes_eq_2": num_nodes_eq_2,
        "cyclomatic_complexity": cyclomatic_complexity,
        "edge_locality_score": edge_locality_score,
        "temporal_clash_count": int(total_clashes),
        "ratio_nodes_ge_3": ratio_nodes_ge_3,
        "ratio_nodes_eq_2": ratio_nodes_eq_2,
    }



def statistic():
    result_dict = []
    motif_name = ["3-star", "4-cycle", "4-clique", "bitriangle","4-chordalcycle"]
    for motif in motif_name:
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/example/run_one_task/judge_contain_motif/api/", motif)
        M_folders = ["M[10.0]","M[20.0]","M[30.0]","M[40.0]","M[50.0]","M[60.0]","M[70.0]","M[80.0]","M[90.0]","M[100.0]"]
        for M_folder in M_folders:
            M_folder_path = os.path.join(base_path, M_folder)
            instance_folders = os.path.join(M_folder_path, "instance_list.json")
            flag_folder = os.path.join(M_folder_path, f"results_{motif}_pangu_auto_cotFalse_roleFalse_k0_dyg1_edge0_imp0_short0_temperature0.0_maxtokens8192_motifname{motif}_motif1_change0_use_agent0_api1.json")
            with open(flag_folder, "r") as f:
                data = json.load(f) 
                flag_data = data["error_folders"]
                flag_data += data["failure_folders"]
            with open(instance_folders, "r") as f:
                instance_list = json.load(f)["instance_list"]
            count = -1
            for instance in instance_list:
                count += 1
                if count >= 30:
                    break
                instance_path = os.path.join(M_folder_path, instance)
                graph_path = os.path.join(instance_path, "graph.json")
                with open(graph_path, "r") as f:
                    graph_data = json.load(f)
                edges = graph_data["edge_index"]
                features = {}
                features["instance"] = instance
                if instance in flag_data:
                    features["flag"] = 0
                else:
                    features["flag"] = 1
                features.update(analyze_graph_features(edges))
                result_dict.append(features)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/example/run_one_task/judge_contain_motif/api")
    json_path = os.path.join(output_dir, f"result_dict_five.json")
    with open(json_path, "w") as f:
        for item in result_dict:
            json.dump(item, f)
            f.write("\n")
    df = pd.DataFrame(result_dict)
    csv_path = os.path.join(output_dir, f"result_dict_five.csv")
    df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    statistic()


    
    