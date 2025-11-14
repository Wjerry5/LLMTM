import numpy as np
import pandas as pd
import os
import torch
from libwon.utils import setup_seed
from collections import Counter, defaultdict
from LLMTM.utils.modif_judge_count import judge, check_temporal_constraints
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,"../../data")

import networkx as nx    
import random
from .motif_utils import assign_timestamps
from typing import List, Set, Dict, Tuple, Optional # Added type hinting
import itertools
import math

def get_sbm_graph(N, pin, pout, C, directed):
    """
    Generate a graph based on the SBM model.

    Args:
        N (int): Number of nodes per community.
        pin (float): Connection probability within communities.
        pout (float): Connection probability between communities.
        C (int): Number of communities.
        directed (bool): Whether to generate a directed graph.

    Returns:
        np.array: List of graph edges, shape [E, 2].
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

        # 1. Generate base graph using ER model
        G_base = nx.erdos_renyi_graph(N, p, seed=seed)
        initial_edges = list(G_base.edges())
        num_initial_edges = len(initial_edges)

        if num_initial_edges == 0: # Handle case with no initial edges - skip this seed
             return None  # Return None to let the caller skip this seed

        temporal_edges = []

        # 2. Use np.random.randint to generate addition times t_a for all initial edges
        #    Timestamp range is [0, T-1]
        addition_times = np.random.randint(0, T, size=num_initial_edges)

        # 3. Process each edge, generate addition event, and probabilistically generate deletion event
        for i, edge in enumerate(initial_edges):
            u, v = edge
            if u > v: u, v = v, u # Normalize

            t_a = addition_times[i] # Get the addition time for this edge
            temporal_edges.append((u, v, t_a, 'a'))

            # Probabilistically generate deletion event
            # Ensure t_a < T-1 to allow deletion afterward
            if np.random.random() < 0.5 and t_a < T - 1:
                # Deletion time t_d > t_a and t_d <= T-1
                t_d = np.random.randint(t_a + 1, T)
                temporal_edges.append((u, v, t_d, 'd'))
        # Sort by time
        temporal_edges.sort(key=lambda x: x[2])
        # Convert to required format
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
            'num_edges': len(edge_index_list) # Total number of events
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

        # Check if the number of edges is valid
        max_edges = N * (N-1) // 2  # Maximum possible number of edges
        if M > max_edges:
            print(f"Warning: Requested number of edges {M} exceeds maximum possible {max_edges}, using max_edges instead")
            M = max_edges

        # 1. Generate a random graph with a fixed number of edges
        G_base = nx.gnm_random_graph(N, M, seed=seed)
        initial_edges = list(G_base.edges())
        num_initial_edges = len(initial_edges)

        if num_initial_edges == 0:  # Handle case with no edges
            return None

        temporal_edges = []

        # 2. Generate random addition times for all edges
        addition_times = np.random.randint(0, T, size=num_initial_edges)

        # 3. Process each edge, generate addition event, and probabilistically generate deletion event
        for i, edge in enumerate(initial_edges):
            u, v = edge
            if u > v: u, v = v, u  # Normalize

            t_a = addition_times[i]  # Get the addition time for this edge
            temporal_edges.append((u, v, t_a, 'a'))

            # Probabilistically generate deletion event
            if np.random.random() < 0.5 and t_a < T - 1:
                t_d = np.random.randint(t_a + 1, T)
                temporal_edges.append((u, v, t_d, 'd'))

        # Sort by time
        temporal_edges.sort(key=lambda x: x[2])
        temporal_edges = temporal_edges[:M]
        # Convert to required format
        edge_index_list = [(int(u), int(v), int(t), str(op)) for u, v, t, op in temporal_edges]
        edges_array = np.array([(e[0], e[1]) for e in edge_index_list])
        node_set = set(edges_array.flatten()) if edges_array.size > 0 else set()
        num_nodes = len(node_set) if node_set else N 
        num_time = len(set([e[2] for e in edge_index_list]))

        info = {
            'T': T, 
            'N': N, 
            'M': M,  # Record initial number of edges instead of probability p
            'seed': seed, 
            'num_nodes': num_nodes,
            'num_time': num_time, 
            'edge_index': edge_index_list,
            'num_edges': len(edge_index_list)  # Total number of events (including add and delete)
        }
        return info


class DyGraphGenSBMCon:
    """
    Generates a dynamic graph, edges from an initial SBM graph, assigns random timestamp t.
    Records as [u, v, t, 'a'] and possibly [u, v, t_delete, 'd'].
    """
    def __init__(self, deletion_prob=0.1):
        self.deletion_prob = deletion_prob

    def sample_dynamic_graph(self, T = 5, N = 10 , p = 0.3, C = 2, directed = False, seed = 0):
        """
        Generates a dynamic graph.
        Args:
            T (int): Maximum timestamp. Default is 5.
            N (int): *Total* number of nodes. Default is 10.
            p (float): SBM intra-community connection probability. Default is 0.3.
            C (int): Number of communities. Default is 2.
            directed (bool): Whether it is a directed graph. Default is False.
            seed (int): Random seed. Default is 0.
        Returns:
            dict: A dictionary containing graph information.
        """
        setup_seed(seed)
        # 1. Generate static SBM graph
        initial_edges = get_sbm_graph(N//C, p, p/2, C, directed)
        if initial_edges.shape[0] == 0:
             return None  # Skip seed with no edges

        # 2. Assign timestamps and generate add/delete records
        final_edge_list = []
        timestamps = np.random.randint(0, T, initial_edges.shape[0])
        for i, edge in enumerate(initial_edges):
            u, v = edge
            t_add = timestamps[i]
            final_edge_list.append([int(u), int(v), int(t_add), 'a'])
            if random.random() < self.deletion_prob and t_add < T - 1:
                t_delete = np.random.randint(t_add + 1, T)
                final_edge_list.append([int(u), int(v), int(t_delete), 'd'])

        # 3. Sort
        final_edge_list = sorted(final_edge_list, key=lambda x: (x[2], x[0]))
        
        # 4. Calculate statistics
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

        # 5. Prepare return dictionary (add "C")
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
    Generates a dynamic graph, edges from an initial Forest Fire graph, assigns random timestamp t.
    Records as [u, v, t, 'a'] and possibly [u, v, t_delete, 'd'].
    """
    def __init__(self, deletion_prob=0.1):
        self.deletion_prob = deletion_prob

    def sample_dynamic_graph(self, T = 5, N = 10 , p = 0.3, directed = False, seed = 0):
        """
        Generates a dynamic graph.
        Args:
            T (int): Maximum timestamp. Default is 5.
            N (int): Number of nodes. Default is 10.
            p (float): Forest Fire forward burning probability (fw_prob). Default is 0.3.
            directed (bool): Whether it is a directed graph (igraph Forest_Fire is undirected by default). Default is False.
            seed (int): Random seed. Default is 0.
        Returns:
            dict: A dictionary containing graph information.
        """
        setup_seed(seed)
        # 1. Generate Forest Fire graph
        g = Graph.Forest_Fire(N, fw_prob=p, directed=directed) # Use directed parameter
        initial_edges = g.get_edgelist() # [(u, v)]
        if not initial_edges:
            return None  # Skip seed with no edges
        initial_edges = np.array(initial_edges)

        # 2. Assign timestamps and generate add/delete records
        final_edge_list = []
        timestamps = np.random.randint(0, T, initial_edges.shape[0])
        for i, edge in enumerate(initial_edges):
            u, v = edge
            t_add = timestamps[i]
            final_edge_list.append([int(u), int(v), int(t_add), 'a'])
            if random.random() < self.deletion_prob and t_add < T - 1:
                t_delete = np.random.randint(t_add + 1, T)
                final_edge_list.append([int(u), int(v), int(t_delete), 'd'])

        # 3. Sort
        final_edge_list = sorted(final_edge_list, key=lambda x: (x[2], x[0]))

        # 4. Calculate statistics
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

        # 5. Prepare return dictionary
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
    Generates dynamic graphs for Motif judgment tasks. (Final simplified v5) (English comments)
    Directly generates Motif topology and randomly permutes nodes, or generates a random GNM graph.
    Then assigns timestamps according to the situation (Case 1/2/3/4).
    """
    def __init__(self, p_use_random_gnm: float = 0.25, p_permute_motif_nodes: float = 0.5, deletion_prob: float = 0.5):
        """
        Initializes the Motif graph generator. (English comments)

        Args:
            p_use_random_gnm (float): Probability of using a random GNM graph (instead of Motif topology).
            p_permute_motif_nodes (float): Probability of randomly permuting its nodes when generating Motif topology.
            deletion_prob (float): Probability of adding a 'd' event for each 'a' event.
        """
        self.p_use_random_gnm = p_use_random_gnm
        self.p_permute_motif_nodes = p_permute_motif_nodes
        self.deletion_prob = deletion_prob

    def sample_dynamic_graph(self, T: int, predefined_motif: list, motif_time_window: int = 5, seed: int = 0, deletion_prob: float = None, **kwargs) -> dict:
        """
        Generates a dynamic graph sample for Motif judgment.

        Args:
            T (int): Maximum timestamp + 1.
            predefined_motif (list): Predefined Motif structure [(u, v, t_rel, 'a'), ...], events need to be in temporal order.
            motif_time_window (int): Maximum time window delta for the Motif.
            deletion_prob (float, optional): Probability of adding deletion events. If None, uses class default.
            seed (int): Random seed.
            **kwargs: Allows passing extra parameters like motif_name (N, M are ignored).

        Returns:
            dict: A dictionary containing the generated graph info, Motif metadata, etc.
        """
        setup_seed(seed)
        current_deletion_prob = deletion_prob if deletion_prob is not None else self.deletion_prob
        motif_name_from_arg = kwargs.get("motif_name", "custom")
        target_edge_order: List[Tuple[int, int]] = [] # Ordered static edge list
        is_motif_topology = False
        N_motif = 0
        M_motif = 0 # Number of topology edges
        original_motif = predefined_motif
        # --- Step 1: Parse Motif definition ---
        # Preprocess Motif definition, convert string node identifiers to integers
        processed_motif = []
        for item in predefined_motif:
            if len(item) == 4:
                u, v, t_rel, event_type = item
                # Check if node identifiers need conversion
                if isinstance(u, str) and u.startswith('u'):
                    try:
                        u = int(u[1:])  # Extract 0 from 'u0'
                    except ValueError:
                        pass  # Keep as is
                if isinstance(v, str) and v.startswith('u'):
                    try:
                        v = int(v[1:])  # Extract 1 from 'u1'
                    except ValueError:
                        pass  # Keep as is
                processed_motif.append((u, v, t_rel, event_type))
            else:
                processed_motif.append(item)  # Keep as is
        # Update predefined_motif to the processed version
        predefined_motif = processed_motif
        motif_nodes_orig = set()
        ordered_motif_events_orig: List[Tuple[int, int]] = [] # Original 'a' event edge list in temporal order
        motif_topology_edges = set() # Store unique topology edges
        predefined_motif = sorted(predefined_motif, key=lambda x: x[2])
        print(f"predefined_motif: {predefined_motif}")
        try:
            # Assume predefined_motif is already sorted by t_rel, or process 'a' events in order of appearance
            for item in predefined_motif:
                if len(item) == 4 and item[3] == 'a':
                    u, v = int(item[0]), int(item[1])
                    motif_nodes_orig.add(u)
                    motif_nodes_orig.add(v)
                    edge_orig = tuple(sorted((u, v)))
                    ordered_motif_events_orig.append(edge_orig) # Add in event order
                    motif_topology_edges.add(edge_orig) # Record topology edge
            N_motif = len(motif_nodes_orig)
            M_motif = len(motif_topology_edges) # Number of unique topology edges
            if N_motif == 0 or M_motif == 0: raise ValueError("Motif definition invalid")
            # Create mapping from original nodes to 0..N-1
            node_map_orig_to_new = {orig_node: i for i, orig_node in enumerate(sorted(list(motif_nodes_orig)))}

        except Exception as e:
            print(f"Error: Failed to parse predefined_motif: {e}")
            return None  # Skip seed that failed to parse

        # --- Step 2 & 3: Probabilistic decision and generation of ordered static edge list ---
        use_random_gnm = (random.random() < self.p_use_random_gnm)
        print(f"use_random_gnm: {use_random_gnm}")
        if use_random_gnm:
            # --- Generate random GNM graph ---
            is_motif_topology = False
            # print("Generating random GNM graph...") # Chinese debug
            try:
                G_gnm = nx.gnm_random_graph(N_motif, M_motif, seed=seed + 1)
                if set(G_gnm.nodes()) != set(range(N_motif)):
                     mapping_relabel = {old_node: new_node for new_node, old_node in enumerate(sorted(G_gnm.nodes()))}
                     G_gnm = nx.relabel_nodes(G_gnm, mapping_relabel)
                target_edge_order = [tuple(sorted(e)) for e in G_gnm.edges()]
            except nx.NetworkXError as e:
                 print(f"Error: Failed to generate GNM graph: {e}. Will generate Motif instance instead.")
                 use_random_gnm = False # Fallback
            if not target_edge_order and not use_random_gnm: # Fallback if GNM failed
                 use_random_gnm = False

        if not use_random_gnm:
            # --- Generate Motif topology instance (possibly with node permutation) ---
            is_motif_topology = True
            # print("Generating Motif topology instance...") # Chinese debug

            # Decide whether to permute nodes
            permute_nodes = (random.random() < self.p_permute_motif_nodes)
            node_map_new_to_final = {} # Mapping from 0..N-1 to final node ID
            if permute_nodes:
                nodes_list = list(range(N_motif))
                permuted_nodes = list(np.random.permutation(nodes_list))
                node_map_new_to_final = {original: permuted for original, permuted in zip(nodes_list, permuted_nodes)}
            else:
                node_map_new_to_final = {i: i for i in range(N_motif)} # Identity mapping

            # Generate target_edge_order in the order of ordered_motif_events_orig
            target_edge_order = []
            valid_mapping = True
            for u_orig, v_orig in ordered_motif_events_orig:
                # Map to 0..N-1
                u_new = node_map_orig_to_new.get(u_orig)
                v_new = node_map_orig_to_new.get(v_orig)
                if u_new is None or v_new is None:
                    print(f"Warning: Cannot map original Motif edge ({u_orig}, {v_orig}) to new nodes.")
                    valid_mapping = False; break
                # Apply final mapping (permutation or identity)
                u_final = node_map_new_to_final.get(u_new)
                v_final = node_map_new_to_final.get(v_new)
                if u_final is None or v_final is None:
                     print(f"Warning: Cannot apply final mapping to new nodes ({u_new}, {v_new}).")
                     valid_mapping = False; break

                final_edge = tuple(sorted((u_final, v_final)))
                target_edge_order.append(final_edge) # Add in original event order

            if not valid_mapping:
                 print("Warning: Mapping failed when generating ordered edge list. Returning empty.")
                 target_edge_order = [] # Set to empty list


        # --- Robustness check: Ensure target_edge_order (as a list) exists ---
        if target_edge_order is None: # Should theoretically not happen, unless GNM completely failed
             print("Critical Warning: target_edge_order is None.")
             target_edge_order = []

        # --- Step 4: Assign timestamps ---
        edge_timestamps, assignment_case = assign_timestamps(
            T, is_motif_topology, target_edge_order, # Pass the ordered list
            motif_time_window, seed
        )

        # --- Step 5: Generate timestamped 'a' event list ---
        temporal_add_events = []
        # Generate directly using the edge_timestamps dictionary, which contains the correct edges and timestamps
        for edge_tuple, t_add in edge_timestamps.items():
            u, v = edge_tuple
            temporal_add_events.append([int(u), int(v), int(t_add), 'a'])

        # --- Step 6: Add deletion events ---
        final_event_list = list(temporal_add_events)
        for u, v, t_add, op in temporal_add_events:
            if random.random() < current_deletion_prob and t_add < T - 1:
                 if t_add + 1 < T:
                     t_delete = np.random.randint(t_add + 1, T)
                     final_event_list.append([int(u), int(v), int(t_delete), 'd'])

        # --- Step 7: Sort ---
        final_event_list.sort(key=lambda x: (x[2], x[0], x[1]))

        # --- Step 8: Calculate statistics and return ---
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
            "original_motif": original_motif, # Keep for debugging or subsequent analysis
            "is_motif_topology": is_motif_topology,
            "assignment_case": assignment_case
        }

        return info

class DyGraphGenMorMotifCon:

    def sample_dynamic_graph(
        self,
        T_total_time: int,     # Parameter T: Total time span (0 to T-1)
        N_total_nodes: int,     # Parameter N: Total number of nodes in the graph (0 to N-1)
        M_target_edges: int,    # Parameter M: Target total number of events in the final graph
        W_motif_window: int,    # Parameter W: Time window length for target_motif
        target_motif_definition: List[Tuple[str, str, str, str]], # e.g., ('u0','u1','t0','a')
        seed: 0,
        p_remap_node: float = 0.5, # Probability: remap motif nodes
        p_delete_bg: float = 0.5,  # Probability of 'd' operation for background events
        max_overall_attempts: int = 5 # Max attempts to generate the entire valid graph
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

            # 2b. Node ID Remapping: 
            #     With probability p_remap_node, map nodes from unique_nodes_from_def_set to random nodes in the graph.
            #     Otherwise, use their own values as graph node IDs (provided they are within the N_total_nodes range).
            # def_node_to_graph_node_map: Key is the integer ID from definition, Value is the final node ID in the graph
            node_map: Dict[int, int] = {}
            available_graph_nodes = list(range(N_total_nodes))
            random.shuffle(available_graph_nodes)
            
            sorted_unique = sorted(list(unique_nodes))

            if random.random() < p_remap_node:
                # Randomly map to the first N_motif available nodes in the graph
                for i, def_node_id in enumerate(sorted_unique):
                    if i < len(available_graph_nodes): # Ensure we don't go out of bounds for available_graph_nodes
                        node_map[def_node_id] = available_graph_nodes[i]
                    else: 
                        break 
            else:
                # Use the integer ID from definition directly as the graph node ID
                for def_node_id in sorted_unique:
                    if def_node_id >= N_total_nodes: 
                        break 
                    node_map[def_node_id] = def_node_id
            
            if len(node_map) != num_nodes_in_motif: 
                continue # Mapping was incomplete

            # 2c. Timestamp Generation: Generate timestamps for motif instance edges within W_motif_window
            t_segment_start = random.randint(0, T_total_time - W_motif_window - 1)

            num_motif_edges = len(processed_motif)
            motif_instance_timestamps = sorted(random.sample(range(t_segment_start, t_segment_start + W_motif_window), num_motif_edges))
            print(f"motif_instance_timestamps: {motif_instance_timestamps}")
            motif_instance_timestamps.sort()
            # Build the edge list for the motif instance (with real node IDs and new timestamps)
            # Format: (u_graph, v_graph, abs_time, op)
            generated_motif_instance_events: List[Tuple[int, int, int, str]] = []
            for i, (u_d, v_d, t, op) in enumerate(processed_motif): # u_d, v_d are nodes from definition
                # Lookup these def_nodes in our map to get their graph IDs
                if u_d not in node_map or v_d not in node_map:
                    continue # Skip this edge if its nodes weren't mapped

                u_g = node_map[u_d]
                v_g = node_map[v_d]
                # Unify node order u < v
                generated_motif_instance_events.append(tuple(sorted((u_g, v_g))) + (motif_instance_timestamps[i], op))
            
            # --- Remove duplicate events caused by motif definition, node mapping, and timestamp assignment ---
            generated_motif_instance_events = list(set(generated_motif_instance_events))
            # ----------------------------------------------------------------

            if not generated_motif_instance_events : continue
            print(f"generated_motif_instance_events (unique): {generated_motif_instance_events}") # Adjusted print statement
            # --- Step 3: Randomly remove one edge from the instance as the answer ---
            generated_motif_instance_events.sort(key=lambda x: x[2])
            answer_edge = generated_motif_instance_events[-1]
            current_event_list = [e for e in generated_motif_instance_events if e != answer_edge]
            print(f"current_event_list: {current_event_list}")
            print(f"answer_edge: {answer_edge}")
            # --- Step 4: Add background edges, target total events M_target_edges ---
            current_event_set: Set[Tuple[int, int, int, str]] = set(current_event_list)
            
            # Loop until M_target_edges is reached or max attempts
            # For simplicity, we loop a fixed number of times based on M_target_edges
            # Need to add M_target_edges - len(current_event_list) edges
            # The probability of each valid addition is affected by judge and p_delete_bg
            num_edges_to_add_approx = M_target_edges - len(current_event_list)
            if num_edges_to_add_approx < 0: num_edges_to_add_approx = 0 # Initial motif might already have many edges

            bg_addition_attempts = max(num_edges_to_add_approx * 3, M_target_edges) # Ensure some attempts
            active_edges = [e for e in current_event_list if e[3] == 'a']
            for _ in range(bg_addition_attempts):
                if len(current_event_list) >= M_target_edges: break
                if random.random() < p_delete_bg:
                    # Randomly select an edge from current_event_list
                    # Filter for edges with 'a' and not already deleted
                    if active_edges:
                        random_edge = random.choice(active_edges)
                        bg_u_node, bg_v_node = random_edge[0], random_edge[1]
                        # Randomly select time from (t, T_total_time-1)
                        bg_time = random.randint(random_edge[2], T_total_time - 1)
                    else:
                        continue
                    current_event_list.append(tuple(sorted((bg_u_node, bg_v_node))) + (bg_time, 'd'))
                    current_event_set.add(tuple(sorted((bg_u_node, bg_v_node))) + (bg_time, 'd'))
                    active_edges.remove(random_edge)
                    print(f"current_event_list: {current_event_list}")
                else:
                    # Randomly select two nodes and a time
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
            
            current_event_list.sort(key=lambda x: (x[2], x[0], x[1], x[3])) # Final sort
            print(f"current_event_list: {current_event_list}")

            # --- Final Validation ---
            if judge(current_event_list, processed_motif, W_motif_window) == "Yes":
                print("Yes")
                continue
            if judge(current_event_list + [answer_edge], processed_motif, W_motif_window) != "Yes":
                print("No")
                continue
    
            print(judge(current_event_list, processed_motif, W_motif_window))
            print(judge(current_event_list + [answer_edge], processed_motif, W_motif_window))
            # --- Success ---
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
            
        return None # All attempts failed


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
    # Generate a temporal graph with M edges based on the ER model.

    # Args:
    #     M (int): Total number of edges in the graph.
    #     motif_name (str): Name of the Motif to control.
    #     seed (int): Random seed.
    #     T (int): Timestamp range [0, T-1].

    # Returns:
    #     A list of M quadruples (edges).
    # """
        random.seed(seed)
        motif_def = PREDEFINED_MOTIFS[motif_name]
        motif_edges_def = motif_def['edge']
        num_motif_edges = len(motif_edges_def)
        
        # Assume number of nodes N = M, which is usually large enough to avoid overly dense graphs
        N = M

        # Decide whether it's a positive or negative example based on seed parity
        is_positive_example = (seed % 2 == 0)

        if is_positive_example:
            # --- Generate positive example (contains Motif) ---
            if M < num_motif_edges:
                raise ValueError(f"Number of edges M={M} is too small to accommodate a '{motif_name}' Motif with {num_motif_edges} edges.")

            final_edges = []
            
            # 1. Plant the Motif
            motif_nodes_str = sorted(list(set(n for e in motif_edges_def for n in e[:2])))
            num_motif_nodes = len(motif_nodes_str)

            if N < num_motif_nodes:
                raise ValueError(f"Number of nodes N={N} is too small to accommodate a '{motif_name}' Motif with {num_motif_nodes} nodes.")

            # Randomly select nodes from the graph and map them to the Motif's abstract nodes
            graph_nodes_for_motif = random.sample(range(N), num_motif_nodes)
            node_map = {str_node: real_node for str_node, real_node in zip(motif_nodes_str, graph_nodes_for_motif)}

            # Generate temporally compliant timestamps for the Motif's edges
            mt = PREDEFINED_MOTIFS[motif_name]['T']
            t_start = random.randint(0, max(0, T - mt-2))
            motif_timestamps = sorted(random.sample(range(t_start, min(T, t_start + mt)), num_motif_edges))
            planted_edges_set = set()
            for i, (u_str, v_str, _, event_type) in enumerate(motif_edges_def):
                u, v = node_map[u_str], node_map[v_str]
                t = motif_timestamps[i]
                final_edges.append((u, v, t, event_type))
                # Store normalized edge (smaller node first)
                planted_edges_set.add(tuple(sorted((u, v))))

            # 2. Add remaining random edges
            edges_to_add = M - num_motif_edges
            # Create a pool of all possible edges, removing planted edges and used nodes
            # 1. Find used nodes
            used_nodes = set(graph_nodes_for_motif)
            # 2. Exclude these nodes when building the edge pool
            all_possible_edges = set(
                (u, v) for u in range(N) for v in range(u + 1, N)
                if u not in used_nodes and v not in used_nodes
            )
            # 3. Then remove planted edges
            available_edges = list(all_possible_edges - planted_edges_set)

            if len(available_edges) < edges_to_add:
                raise ValueError(f"Graph node count N={N} is too small to add {edges_to_add} non-repeating edges after planting the Motif.")
            
            random_new_edges_a = random.sample(available_edges, int(edges_to_add * 2/3))

            for u, v in random_new_edges_a:
                # Assign random timestamps to new edges
                t = random.randint(0, T - 1)
                final_edges.append((u, v, t, 'a'))
            
            random_new_edges_d = random.sample(final_edges, M - len(final_edges))
            max_time = max(edge[2] for edge in final_edges)
            print(f"max_time: {max_time}")
            for u, v, t0, s0 in random_new_edges_d:
                t = random.randint(max_time, T - 1)
                final_edges.append((u, v, t, 'd'))

            # Finally, sort all edges to hide the planted Motif
            final_edges = sorted(final_edges, key=lambda x: (x[2], 0 if x[3] == 'a' else 1))
            info = {
            'T': M, 
            'N': M, 
            'seed': seed, 
            'edge_index': final_edges,
            'num_edges': len(final_edges) # Total number of events
                }
            print("Contains Motif")
            print(final_edges)
            print(judge(final_edges,PREDEFINED_MOTIFS[motif_name]['edge'],PREDEFINED_MOTIFS[motif_name]['T']))
            return info

        else:
            # --- Generate negative example (does not contain Motif) ---
            # "Generate-Test" loop
            while True:
                # 1. Use ER model to generate a candidate graph, directly specifying edge count as M*2/3
                target_edges = math.ceil(M * 2/3)  # Target number of edges
                G_base = nx.gnm_random_graph(M, target_edges, seed=seed)
                initial_edges = list(G_base.edges())
                num_initial_edges = len(initial_edges)

                if num_initial_edges == 0: # Handle case with no initial edges - skip this seed
                    return None  # Return None to let the caller skip this seed

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
                
                
                # 2. Check if the generated graph accidentally contains the Motif
                if not judge(final_edges,PREDEFINED_MOTIFS[motif_name]['edge'],PREDEFINED_MOTIFS[motif_name]['T']) == "Yes":
                    # If it doesn't, this is a valid negative example
                    info = {
                    'T': M, 
                    'N': M, 
                    'seed': seed, 
                    'edge_index': final_edges,
                    'num_edges': len(final_edges) # Total number of events
                    }
                    print("Does not contain Motif")
                    return info
                else:
                    seed += 1
                    continue
                # Otherwise, the loop will continue, generating and testing a new graph
                # print(f"  [Debug] Seed {seed}: Generated graph contains '{motif_name}', regenerating...")




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
    Converts a list of edges per timestep (each timestep is a [2, num_edges] tensor)
    into a single [total_num_edges, 3] tensor, where the third column is the timestep index.
    """
    es = []
    for i,e in enumerate(edges):
        if e.numel() > 0: # Check if tensor is empty
             # Ensure e is LongTensor
             if not isinstance(e, torch.LongTensor):
                 e = e.long()
             # Add timestamp column
             time_col = torch.full((e.shape[1], 1), i, dtype=torch.long, device=e.device)
             e_with_time = torch.cat([e.t(), time_col], dim=1)
             es.append(e_with_time)
    if not es: # If all timesteps were empty
        return torch.empty((0, 3), dtype=torch.long)
    es = torch.cat(es, dim = 0)
    return es

def turn_undirect(edges_tensor):
    """
    Processes an edge tensor containing (u, v, t) tuples, removing duplicate undirected edges.
    Only keeps the representation where u <= v.
    """
    if edges_tensor.numel() == 0:
        return edges_tensor

    # Ensure u <= v
    u = torch.min(edges_tensor[:, 0], edges_tensor[:, 1])
    v = torch.max(edges_tensor[:, 0], edges_tensor[:, 1])
    t = edges_tensor[:, 2]

    # Combine into a new tensor and remove duplicates
    unique_edges = torch.unique(torch.stack([u, v, t], dim=1), dim=0)
    return unique_edges