import networkx as nx
from collections import defaultdict


def check_temporal_constraints(mapping, sorted_motif_events, context_times_all, motif_time_window):
    """
    Check if there exists a timestamp sequence satisfying temporal and window constraints
    for the given isomorphic mapping.
    (Uses iterative method to find valid sequences)
    """
    possible_sequences = [[]]  # Store partially valid timestamp sequences

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
                if t_c > last_t: # Check order
                    if t_start is None: # First event
                        next_possible_sequences.append(sequence + [t_c])
                    else: # Subsequent events, check window
                        if motif_time_window <= 0 or (t_c - t_start) <= motif_time_window:
                            next_possible_sequences.append(sequence + [t_c])

        possible_sequences = next_possible_sequences
        if not possible_sequences: return False # No valid sequences can be extended
    return True # Successfully built at least one complete sequence

def judge(context, motif_definition, motif_time_window):
    """
    Precisely determine if there exists an instance of motif_definition in the context graph,
    satisfying both topological, temporal order, and time window constraints (considering all timestamps).
    """
    context_a = [e for e in context if len(e) == 4 and e[3] == 'a']
    motif_a = [e for e in motif_definition if len(e) == 4 and e[3] == 'a']

    if not motif_a: return "No"

    # 1. Parse Motif
    G_motif_template = nx.Graph()
    motif_nodes_orig = set()
    sorted_motif_events = []
    try:
        temp_motif_events = []
        for u, v, t, op in motif_a:
            if isinstance(u, str) and u.startswith('u'):
                u = int(u[1:])  # Extract 0 from 'u0'
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # Extract 1 from 'u1'
            else:
                v = int(v)
            if isinstance(t, str) and t.startswith('t'):
                t = int(t[1:])  # Extract 0 from 't0'
            else:
                t = int(t)
            u_node, v_node = int(u), int(v)
            motif_nodes_orig.add(u_node)
            motif_nodes_orig.add(v_node)
            G_motif_template.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t))
        sorted_motif_events = sorted(temp_motif_events, key=lambda x: x[2])
    except Exception as e:
        print(f"Error (judge): Failed to parse motif_definition: {e}")
        return "Error"

    # 2. Parse Context (store all timestamps)
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
         print(f"Error (judge): Failed to parse context: {e}")
         return "Error"

    # 3. Check isomorphism
    if G_context.number_of_nodes() < G_motif_template.number_of_nodes() or \
       G_context.number_of_edges() < G_motif_template.number_of_edges():
        return "No"

    matcher = nx.isomorphism.GraphMatcher(G_context, G_motif_template)

    # 4. Iterate through each isomorphic mapping to check temporal constraints
    for mapping in matcher.subgraph_isomorphisms_iter():
        reversed_mapping = {v: k for k, v in mapping.items()}
        if check_temporal_constraints(reversed_mapping, sorted_motif_events, context_times_all, motif_time_window):
            return "Yes" # Found an instance that satisfies conditions

    # 5. If no mapping satisfies temporal constraints
    return "No"

def multi_motif_judge(context, PREDEFINED_MOTIFS):
    motif_names = set()
    for motif_name, motif_definition in PREDEFINED_MOTIFS.items():
        if judge(context, motif_definition["edge_pattern"], motif_definition["time_window"]) == "Yes":
            motif_names.add(motif_name)
    return list(motif_names)
    


def count_temporal_constraints(mapping_motif_to_context, sorted_motif_events, context_times_all, motif_time_window):
    """
    For a given node mapping, find all motif instances that satisfy temporal constraints,
    and return them in normalized frozenset form.
    """
    found_instances = set() # Use set to store unique instances (frozenset)
    num_events = len(sorted_motif_events)

    def find_instances_recursive(event_index, current_instance_edges):
        # current_instance_edges: list of (u_c, v_c, t_c, 'a') tuples for the instance being built

        if event_index == num_events:
            # Found a complete instance, normalize it and add to set
            normalized_instance = frozenset(
                (min(u, v), max(u, v), t, op) for u, v, t, op in current_instance_edges
            )
            found_instances.add(normalized_instance)
            return

        # Get current motif event
        u_m, v_m, t_rel = sorted_motif_events[event_index]
        # Get corresponding context nodes
        u_c = mapping_motif_to_context.get(u_m)
        v_c = mapping_motif_to_context.get(v_m)
        # (Should check if u_c, v_c are None, though theoretically impossible)

        context_edge_topology = tuple(sorted((u_c, v_c)))
        candidate_times = context_times_all.get(context_edge_topology, [])

        last_t = current_instance_edges[-1][2] if current_instance_edges else -1
        t_start = current_instance_edges[0][2] if current_instance_edges else None

        for t_c in candidate_times:
            # 1. Check temporal order
            if t_c > last_t:
                # 2. Check time window
                is_within_window = True
                if t_start is not None and motif_time_window > 0:
                     # Note: Time window definition here is t_last - t_first < W
                     # If it's <= W, then condition is (t_c - t_start) <= motif_time_window
                     # Assume < W (strictly less than)
                    if (t_c - t_start) > motif_time_window:
                        is_within_window = False

                if is_within_window:
                    # Build context edge tuple for current event
                    current_edge_event = (u_c, v_c, t_c, 'a')
                    # Add to current instance and recurse
                    current_instance_edges.append(current_edge_event)
                    find_instances_recursive(event_index + 1, current_instance_edges)
                    current_instance_edges.pop() # Backtrack

    find_instances_recursive(0, [])

    return found_instances # Return set containing all instances found for this mapping

def count(context, motif_definition, motif_time_window):
    # 1. Parse context and motif (reuse judge logic)
    context_a = [e for e in context if len(e) == 4 and e[3] == 'a']
    motif_a = [e for e in motif_definition if len(e) == 4 and e[3] == 'a']
    if not motif_a or not context_a: return 0

    try:
        # Parse Motif
        G_motif_template = nx.Graph()
        motif_nodes_orig = set()
        temp_motif_events = []
        for u, v, t, op in motif_a:
            if isinstance(u, str) and u.startswith('u'):
                u = int(u[1:])  # Extract 0 from 'u0'
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # Extract 1 from 'u1'
            else:
                v = int(v)
            if isinstance(t, str) and t.startswith('t'):
                t = int(t[1:])  # Extract 0 from 't0'
            else:
                t = int(t)
            u_node, v_node = int(u), int(v)
            motif_nodes_orig.add(u_node)
            motif_nodes_orig.add(v_node)
            G_motif_template.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t))
        sorted_motif_events = sorted(temp_motif_events, key=lambda x: x[2])

        # Parse Context
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
        print(f"Error (count): Failed to parse context or motif: {e}")
        return -1 # Return error code

    # Check node/edge counts for early exit
    if G_context.number_of_nodes() < G_motif_template.number_of_nodes() or \
       G_context.number_of_edges() < G_motif_template.number_of_edges():
        return 0

    # 2. Initialize isomorphism matcher and result set
    matcher = nx.isomorphism.GraphMatcher(G_context, G_motif_template)
    all_unique_instances = set()

    # 3. Iterate through isomorphic mappings, call count_temporal_constraints and collect results
    for mapping_context_to_motif in matcher.subgraph_isomorphisms_iter():
        # We need motif -> context mapping
        mapping_motif_to_context = {v: k for k, v in mapping_context_to_motif.items()}

        # Check if it's a complete mapping (just in case, though subgraph_isomorphisms_iter should only return complete matches)
        if set(mapping_motif_to_context.keys()) != motif_nodes_orig:
             continue

        # Find all instances satisfying temporal constraints for current mapping
        instances_for_this_mapping = count_temporal_constraints(
            mapping_motif_to_context,
            sorted_motif_events,
            context_times_all,
            motif_time_window
        )
        # Add found instances (already frozensets) to total set
        all_unique_instances.update(instances_for_this_mapping)
    print(all_unique_instances)
    # 4. Return count of unique instances
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
    # 1. Parse context and motif (reuse judge logic)
    context_a = [e for e in context if len(e) == 4 and e[3] == 'a']
    motif_a = [e for e in motif_definition if len(e) == 4 and e[3] == 'a']
    if not motif_a or not context_a: return 0

    try:
        # Parse Motif
        G_motif_template = nx.Graph()
        motif_nodes_orig = set()
        temp_motif_events = []
        for u, v, t, op in motif_a:
            if isinstance(u, str) and u.startswith('u'):
                u = int(u[1:])  # Extract 0 from 'u0'
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # Extract 1 from 'u1'
            else:
                v = int(v)
            if isinstance(t, str) and t.startswith('t'):
                t = int(t[1:])  # Extract 0 from 't0'
            else:
                t = int(t)
            u_node, v_node = int(u), int(v)
            motif_nodes_orig.add(u_node)
            motif_nodes_orig.add(v_node)
            G_motif_template.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t))
        sorted_motif_events = sorted(temp_motif_events, key=lambda x: x[2])

        # Parse Context
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
        print(f"Error (count): Failed to parse context or motif: {e}")
        return -1 # Return error code

    # Check node/edge counts for early exit
    if G_context.number_of_nodes() < G_motif_template.number_of_nodes() or \
       G_context.number_of_edges() < G_motif_template.number_of_edges():
        return 0x3f3f3f3f

    # 2. Initialize isomorphism matcher and result set
    matcher = nx.isomorphism.GraphMatcher(G_context, G_motif_template)
    all_unique_instances = set()

    # 3. Iterate through isomorphic mappings, call count_temporal_constraints and collect results
    for mapping_context_to_motif in matcher.subgraph_isomorphisms_iter():
        # We need motif -> context mapping
        mapping_motif_to_context = {v: k for k, v in mapping_context_to_motif.items()}

        # Check if it's a complete mapping (just in case, though subgraph_isomorphisms_iter should only return complete matches)
        if set(mapping_motif_to_context.keys()) != motif_nodes_orig:
             continue

        # Find all instances satisfying temporal constraints for current mapping
        instances_for_this_mapping = count_temporal_constraints(
            mapping_motif_to_context,
            sorted_motif_events,
            context_times_all,
            motif_time_window
        )
        # Add found instances (already frozensets) to total set
        all_unique_instances.update(instances_for_this_mapping)
    if len(all_unique_instances) == 0:
        return 0x3f3f3f3f
    # 4. Return count of unique instances
    else:
        # Find minimum time of last edge among all instances
        min_last_time = 0x3f3f3f3f
        for instance in all_unique_instances:
            # Sort edges in instance by time
            sorted_edges = sorted(instance, key=lambda x: x[2])  # x[2] is timestamp
            # Get time of last edge
            last_edge_time = sorted_edges[-1][2]
            # Update minimum
            min_last_time = min(min_last_time, last_edge_time)
        return int(min_last_time)

def multi_motif_first_time(context, PREDEFINED_MOTIFS):
    """
    Detect all motifs present in the dynamic graph and their first completion times
    
    Args:
        context: Dynamic graph edge list [(u, v, t, op), ...]
        
    Returns:
        Dictionary containing motif names and their first completion times {motif_name: first_time, ...}
        If a motif is not present, it will not be included in the results
    """
    motif_results = {}
    for motif_name, motif_definition in PREDEFINED_MOTIFS.items():
        first_time = motif_and_first_time(context, motif_definition["edge_pattern"], motif_definition["time_window"])
        # 0x3f3f3f3f indicates no instance of this motif was found
        if first_time != 0x3f3f3f3f:
            motif_results[motif_name] = first_time
    return motif_results

def find_all_fragment_temporal_instances(map_frag_to_ctx, frag_events, ctx_times, time_window):
    """
    Find all temporal instances in the dynamic graph that match a given fragment (motif missing one edge)
    
    Args:
        map_frag_to_ctx: Mapping from fragment nodes to context nodes
        frag_events: Event sequence in the fragment [(u_m, v_m, t_rel, orig_idx), ...]
        ctx_times: Timestamp information for each edge in context graph {(u_c, v_c): [t1, t2, ...]}
        time_window: Time window constraint
    
    Returns:
        List of all matching temporal instances
    """
    all_instances = []
    num_frag_events = len(frag_events)

    def find_recursive(frag_idx, current_detail):
        # Recursion termination: all fragment events have been matched
        if frag_idx == num_frag_events:
            all_instances.append(list(current_detail))
            return

        # Get current fragment event
        u_m, v_m, t_rel, orig_idx = frag_events[frag_idx]
        # Get corresponding context nodes through mapping
        u_c = map_frag_to_ctx.get(u_m)
        v_c = map_frag_to_ctx.get(v_m)
        if u_c is None or v_c is None: 
            return  # Invalid mapping, should not happen

        # Get all timestamps for corresponding edge in context
        edge_topo = tuple(sorted((u_c, v_c)))
        candidate_times = ctx_times.get(edge_topo, [])
        
        # Get time constraints: last time and start time of current sequence
        last_t = current_detail[-1][3] if current_detail else -1
        t_start = current_detail[0][3] if current_detail else None

        # Try all candidate timestamps
        for t_c in candidate_times:
            # Check time order: current time must be later than previous time
            if t_c > last_t:
                # Check time window constraint
                if t_start is None or time_window <= 0 or (t_c - t_start) <= time_window:
                    current_detail.append((orig_idx, u_c, v_c, t_c))
                    find_recursive(frag_idx + 1, current_detail)
                    current_detail.pop()  # Backtrack

    find_recursive(0, [])
    return all_instances

def modify(context, motif_def, time_window):
    """
    Find an edge that can be added to make the dynamic graph contain a specific motif
    
    Algorithm approach:
    1. Iterate through each edge in the motif, assuming it's missing
    2. Construct fragment graph with one edge removed
    3. Find subgraph isomorphism mappings of fragment in context graph
    4. For each mapping, find temporal instances of fragment
    5. Calculate timestamp for edge to be added
    6. Check if addition satisfies time window constraints
    7. Verify if motif exists after adding edge
    
    Args:
        context: Dynamic graph edge list [(u, v, t, op), ...]
        motif_def: Motif definition [(u, v, t, op), ...]  
        time_window: Time window constraint
        
    Returns:
        Edge that can be added (u, v, t, 'a') or None
    """
    # 1. Preprocessing: only consider add operations
    ctx_edges = [e for e in context if len(e) == 4 and e[3].lower() == 'a']
    motif_edges = [e for e in motif_def if len(e) == 4 and e[3].lower() == 'a']

    if not motif_edges:
        print("Error (modify): Motif definition has no 'add' edges.")
        return None

    try:
        # 2. Build motif graph and sort events
        motif_graph = nx.Graph()
        motif_nodes = set()
        temp_motif_events = []
        
        for u, v, t, op in motif_edges:
            if isinstance(u, str) and u.startswith('u'):
                u = int(u[1:])  # Extract 0 from 'u0'
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # Extract 1 from 'u1'
            else:
                v = int(v)
            if isinstance(t, str) and t.startswith('t'):
                t = int(t[1:])  # Extract 0 from 't0'
            else:
                t = int(t)
            u_node, v_node = int(u), int(v)
            motif_nodes.add(u_node)
            motif_nodes.add(v_node)
            motif_graph.add_edge(u_node, v_node)
            temp_motif_events.append((u_node, v_node, t))
            
        sorted_events = sorted(temp_motif_events, key=lambda x: x[2])
        num_events = len(sorted_events)
            
        # 3. Build context graph and time information
        ctx_graph = nx.Graph()
        ctx_times = defaultdict(list)
        
        for u, v, t, op in ctx_edges:
            u_node, v_node = int(u), int(v)
            edge = tuple(sorted((u_node, v_node)))
            ctx_graph.add_edge(u_node, v_node)
            ctx_times[edge].append(int(t))
            
        # Sort timestamps for each edge
        for edge in ctx_times:
            ctx_times[edge].sort()
                
    except Exception as e:
        print(f"Error (modify): Failed to parse data: {e}")
        return None

    # 4. Iterate through each potentially missing motif edge
    for miss_idx in range(num_events):
        u_m_miss, v_m_miss, t_miss = sorted_events[miss_idx]

        # 5. Build fragment graph with one edge removed
        frag_graph = motif_graph.copy()
        frag_graph.remove_edge(u_m_miss, v_m_miss)
        print(f"frag_graph.edges(): {frag_graph.edges()}")
        # Build fragment event sequence (excluding missing edge)
        frag_events = []
        for idx, (u, v, t_rel) in enumerate(sorted_events):
            if idx != miss_idx:
                frag_events.append((u, v, t_rel, idx))

        # 6. Check if context graph is large enough to contain fragment
        if ctx_graph.number_of_nodes() < frag_graph.number_of_nodes() or \
            ctx_graph.number_of_edges() < frag_graph.number_of_edges():
            continue

        # 7. Find subgraph isomorphism mappings of fragment in context
        matcher = nx.isomorphism.GraphMatcher(ctx_graph, frag_graph, edge_match=lambda e1, e2: True)
        
        for ctx_map in matcher.subgraph_isomorphisms_iter():
            # Build mapping from fragment to context
            frag_map = {v: k for k, v in ctx_map.items()}
            print(f"frag_map: {frag_map}")
            if set(frag_map.keys()) != set(frag_graph.nodes()):
                continue

            # 8. Find all temporal instances of fragment
            found_instances = find_all_fragment_temporal_instances(
                frag_map,
                frag_events,
                ctx_times,
                time_window
            )
            # print(found_instances)
            # 9. For each fragment instance, calculate insertion time for missing edge
            for instance_detail in found_instances:
                # Find time range where missing edge should be inserted
                t_prev = -1  # Latest time before missing edge
                t_next = float('inf')  # Earliest time after missing edge

                for orig_idx, _, _, t_c in instance_detail:
                    if orig_idx < miss_idx:
                        t_prev = max(t_prev, t_c)
                    elif orig_idx > miss_idx:
                        t_next = min(t_next, t_c)

                # Calculate time for adding edge: need to handle various cases flexibly
                possible_times = []
                
                if miss_idx == 0:
                    # First edge missing: try multiple candidate times
                    if t_next != float('inf'):
                        # Try multiple time points before first subsequent edge
                        for candidate_t in range(max(0, int(t_next) - 10), int(t_next)):
                            if candidate_t >= 0:
                                possible_times.append(candidate_t)
                    else:
                        # No subsequent edges, use default time
                        possible_times.append(0)
                else:
                    # Non-first edge missing: find suitable time between before and after edges
                    if t_prev >= 0 and t_next != float('inf'):
                        # Find time between before and after edges
                        for candidate_t in range(t_prev + 1, min(int(t_next), t_prev + 10)):
                            possible_times.append(candidate_t)
                    elif t_prev >= 0:
                        # Only have previous edge, insert after it
                        possible_times.append(t_prev + 1)
                    elif t_next != float('inf'):
                        # Only have next edge, insert before it
                        possible_times.append(max(0, t_next - 1))
            
                # Try each candidate time
                for t_add in possible_times:
                    # Check time range validity
                    if miss_idx > 0 and t_add >= t_next:
                        continue  # Invalid time, skip
                    
                    # 10. Get corresponding nodes in context for missing edge
                    ctx_u = frag_map.get(u_m_miss)
                    ctx_v = frag_map.get(v_m_miss)

                    if ctx_u is None or ctx_v is None:
                        continue

                    # 11. Check time window constraint
                    full_times = [t for _, _, _, t in instance_detail] + [t_add]
                    min_time = min(full_times)
                    max_time = max(full_times)

                    is_within_window = (time_window <= 0) or ((max_time - min_time) <= time_window)

                    if is_within_window:
                        # Build edge to add
                        add_edge = tuple(sorted((ctx_u, ctx_v))) + (t_add, 'a')
                        
                        # 12. Verify if motif exists after adding edge
                        if judge(context + [add_edge], motif_def, time_window) == "Yes":
                            return add_edge
                                                    # If this time doesn't work, try next time

    print("Error (modify): Could not find edge to add. Guarantee violated?")
    return None


def sort_edges(temporal_edges):
    """
    Sort dynamic graph edges by timestamp
    
    Args:
        temporal_edges: Dynamic graph edge list [(u, v, t, op), ...]
        
    Returns:
        Edge list sorted by timestamp
    """
    return sorted(temporal_edges, key=lambda x: x[2])

def when_direct_link(temporal_edges, u, v):
    """
    Find first connection and first disconnection time between two nodes
    
    Args:
        temporal_edges: Dynamic graph edge list [(u, v, t, op), ...]
        u: First node
        v: Second node
        
    Returns:
        (first_link, first_delete): First connection time and first disconnection time
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
    Find all edges that exist at a specified time point
    
    Args:
        temporal_edges: Dynamic graph edge list [(u, v, t, op), ...]
        selected_time: Target timestamp
        
    Returns:
        List of edges existing at selected_time [(u, v, t_add, 'a'), ...]
    """
    if not temporal_edges:
        return []

    current_edges = {}  # {(u, v): original_add_quadruple}
    temporal_edges = sort_edges(temporal_edges)
    
    # Iterate through all events with time <= selected_time
    for u, v, t_event, op in temporal_edges:
        if t_event > selected_time:
            break  # Since sorted, subsequent events have larger times, no need to process

        edge = tuple(sorted((u, v)))  # Normalize edge representation

        if op == 'a':
            # Record add operation, store original quadruple
            current_edges[edge] = (u, v, t_event, op)
        elif op == 'd':
            # If edge exists, delete it
            if edge in current_edges:
                del current_edges[edge]

    # Extract original quadruples of edges that exist at selected_time
    existing_edges_at_t = list(current_edges.values())
    return existing_edges_at_t

def reverse_graph(temporal_edges):
    """
    Reverse operations in dynamic graph: change add operations to delete operations,
    and delete operations to add operations
    
    Args:
        temporal_edges: Dynamic graph edge list [(u, v, t, op), ...]
        
    Returns:
        Reversed edge list
    """
    reverse_edges = []
    for u, v, t, op in temporal_edges:
        if op == 'a':
            reverse_edges.append((u, v, t, 'd'))
        elif op == 'd':
            reverse_edges.append((u, v, t, 'a'))
    # Reverse the entire reversed edge list
    reverse_edges.reverse()
    return reverse_edges
    