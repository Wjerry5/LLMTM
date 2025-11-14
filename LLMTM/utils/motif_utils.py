import random
import numpy as np
from libwon.utils import setup_seed
from typing import List, Dict, Tuple, Optional # Added type hinting

def assign_timestamps(T: int,
                      is_motif_topology: bool,
                      target_edge_order: List[Tuple[int, int]], # Now receives an ordered list
                      motif_time_window: int,
                      seed: int) -> Tuple[Dict[Tuple[int, int], int], int]:
    """
    Assigns timestamps to edges in the graph. (Final simplified version) (English comments)
    If is_motif_topology=True, probabilistically generates Case 1/2/3 timestamps, using the target_edge_order sequence.
    If is_motif_topology=False, assigns random timestamps (Case 4).

    Args:
        T (int): Maximum timestamp + 1 (time range [0, T-1]).
        is_motif_topology (bool): Whether the input graph's topology is isomorphic to the Motif.
        target_edge_order (list): Ordered list of edges in the graph. For Motif topology,
                                  this order reflects the Motif's temporal order; for random graphs,
                                  this order is irrelevant (but still used for iteration).
        motif_time_window (int): Maximum time window delta defined by the Motif.
        seed (int): Random seed.

    Returns:
        tuple: (edge_timestamps (dict), assignment_case (int))
    """
    setup_seed(seed + 2)
    edge_timestamps = {}
    assignment_case = 4 # Default Case 4

    if not target_edge_order: # List is empty
        return edge_timestamps, assignment_case

    num_events_to_assign = len(target_edge_order) # Number of events/edges

    # --- Case A: Topology is not isomorphic to Motif (or random GNM graph) ---
    if not is_motif_topology:
        # print("Case 4: Topology non-isomorphic or random GNM, assigning random timestamps.") # English debug
        timestamps = np.random.randint(0, T, num_events_to_assign)
        # Assign to all edges in target_edge_order (order doesn't matter)
        for i, edge in enumerate(target_edge_order):
            edge_timestamps[edge] = timestamps[i]
        return edge_timestamps, 4

    # --- Case B: Topology is isomorphic to Motif ---
    # Probabilistically decide the target Case and try to generate timestamps (logic unchanged)
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
    base_times = [] # Store the base time sequence for Case 1
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
        print(f"Error generating base sequence for Case 1: {e}")
        base_times = [] # Generation failed
    print(f"base_times: {base_times}")
# If base sequence generation failed, cannot generate Case 1, 2, 3
    if not base_times:
        target_case = 4 # Force Fallback
    else:
        if target_case == 1:
            final_times = base_times
            assignment_case = 1
            generated_successfully = True
        elif target_case == 3:
            shuffled_times = list(base_times) # Copy
            original_sorted = list(base_times) # Keep original sorted
            attempts = 0
            max_shuffle_attempts = 5
            is_actually_shuffled = False
            while attempts < max_shuffle_attempts:
                attempts += 1
                random.shuffle(shuffled_times)
                print(f"shuffled_times: {shuffled_times}")
                # Check if it's really no longer strictly ordered
                is_shuffled_strictly_ordered = all(shuffled_times[i] < shuffled_times[i+1] for i in range(num_events_to_assign-1))
                if not is_shuffled_strictly_ordered:
                    final_times = shuffled_times
                    assignment_case = 3
                    generated_successfully = True
                    # print(f"Successfully shuffled (attempt {attempts}).") # English debug
                    break # Successfully shuffled, exit loop
            if not generated_successfully:
                print("Could not break strict order after multiple shuffles, Fallback to Case 4.")
                target_case = 4 
        elif target_case == 2:
            attempts = 0
            max_construct_attempts = 5 # Try to construct a few times
            while attempts < max_construct_attempts and not generated_successfully:
                attempts += 1
                print("attempt: ", attempts)
                try:
                    t_min = base_times[0]
                    lower_bound_tmax = max(t_min + motif_time_window+1, base_times[-2]+1)
                    upper_bound_tmax = T - 1
                
                    t_max = random.randint(lower_bound_tmax, upper_bound_tmax)

                    # 3. Determine intermediate points
                    intermediate_times = []
                    possible_intermediate = list(range(t_min + 1, t_max))
                    intermediate_times = random.sample(possible_intermediate, num_events_to_assign - 2)
                    # 4. Combine and sort
                    candidate_final_times = sorted([t_min] + intermediate_times + [t_max])

                    # 5. Validate (should theoretically satisfy)
                    final_span = candidate_final_times[-1] - candidate_final_times[0]
                    print(f"candidate_final_times: {candidate_final_times}")
                    print(f"final_span: {final_span}")
                    if final_span > motif_time_window:
                        final_times = candidate_final_times
                        assignment_case = 2
                        generated_successfully = True
                        break 
                    else: 
                        print(f"Case 2 construction validation failed: ordered={final_is_ordered}, span={final_span}>{motif_time_window}")
                    if not generated_successfully:
                        print(f"Case 2 construction validation failed: ordered={final_is_ordered}, span={final_span}>{motif_time_window}")
                except ValueError as e: 
                    continue # Retry

    if not generated_successfully:
        # print(f"Fallback to Case 4 (target: {target_case}, success: {generated_successfully})") # English debug
        assignment_case = 4
        timestamps = np.random.randint(0, T, num_events_to_assign)
        final_times = list(timestamps) 
    # --- Assign the final timestamps ---
    # Use target_edge_order for assignment
    for i, edge in enumerate(target_edge_order):
        # Ensure final_times list length is correct
        if i < len(final_times):
             edge_timestamps[edge] = final_times[i]
        else:
             # This should not happen, but as a safeguard
             print(f"Warning: final_times list length ({len(final_times)}) is shorter than target_edge_order ({len(target_edge_order)}).")
             # Can assign a default or the last value
             edge_timestamps[edge] = final_times[-1] if final_times else random.randint(0, T-1)


    return edge_timestamps, assignment_case