"""
Dynamic Graph Analysis Toolset

This module defines a toolset for dynamic graph analysis, intended for use by LLM agents.
Tools include: graph generation, motif detection, graph modification, and graph statistics.
"""

from langchain.tools import tool
from typing import List, Tuple, Dict, Any, Optional, Union
import json
import numpy as np
import networkx as nx
import re

import sys # Import the sys module for path operations
import os
# --- Add the project root directory to sys.path ---
# Get the absolute path of the current script file
# e.g., /home/hb/LLMDyG_Motif/scripts/example/run_one_task.py
_current_file_path = os.path.abspath(__file__)
# Get the directory where the script is located /home/hb/LLMDyG_Motif/scripts/example
_script_dir = os.path.dirname(_current_file_path)
# Get the project root directory /home/hb/LLMDyG_Motif (two levels up)
_project_root = os.path.dirname(os.path.dirname(_script_dir))
# Add the project root directory to the beginning of the Python interpreter's module search path list
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# --- Path addition complete ---

# Import relevant functions from the project
# Note: Paths may need adjustment based on the actual project structure
try:
    from LLMTM.utils.modif_judge_count import judge, modify, sort_edges, when_direct_link, what_edge_at_time, reverse_graph, multi_motif_judge, multi_motif_counts, multi_motif_first_time
except ImportError:
    # If absolute import fails, try relative import
    from ..utils.modif_judge_count import judge, modify, sort_edges, when_direct_link, what_edge_at_time, reverse_graph, multi_motif_judge, multi_motif_counts, multi_motif_first_time

# Initialize the tool list (will eventually be overwritten by the definitions below)
ALL_TOOLS = []


def parse_tool_input(input_str: str, tool_name: str):
    """
    General tool input parsing function, handles all common format issues, including single and multiple JSON dictionaries
    
    Args:
        input_str: The original input string
        tool_name: The name of the tool, for debugging output
        
    Returns:
        The parsed parameter dictionary
    """
    print(f"{tool_name} received parameters (raw string): {input_str}")
    
    # 0. Extract the JSON part (content between the first { and the last })
    first_brace = input_str.find('{')
    last_brace = input_str.rfind('}')
    if first_brace != -1 and last_brace != -1:
        input_str = input_str[first_brace:last_brace + 1]
    

    # 1. Clean input: remove extra newlines and spaces
    cleaned_input = re.sub(r'\s+', ' ', input_str.strip())
    
    # 2. Fix single quote issue
    fixed_json = cleaned_input.replace("'", '"')
    fixed_json = fixed_json.replace(r'\"', '"')
    fixed_json = re.sub(r'\}\s*,\s*("([^"]+)"\s*:\s*\{)', r', \1', fixed_json)
    
    # 3.1. Fix quoted numbers: convert "5" to 5 (but not dictionary keys)
    fixed_json = re.sub(r'"(\d+)"(?!\s*:)', r'\1', fixed_json)

    # 3.2. [Critical Fix] Fix bare letters a and d
    fixed_json = re.sub(r',\s*([ad])(?=\s*[)\]\}])', r', "\1"', fixed_json)

    # 4. Fix format issue: convert (1,2,3,a) format to [1,2,3,"a"] format
    pattern = r'\(\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*\)'
    replacement = r'[\1, \2, \3, \4]'
    fixed_format = re.sub(pattern, replacement, fixed_json)
    
    # 4.3. [New] Fix 4-tuple format like {5,4,3,"a"}
    #      [^,:}]+ ensures we only match tuples, not dictionaries
    #      e.g.: {5,4,3,"a"} -> [5, 4, 3, "a"]
    pattern_braces_4 = r'\{\s*([^,:}]+?)\s*,\s*([^,:}]+?)\s*,\s*([^,:}]+?)\s*,\s*([^,:}]+?)\s*\}'
    replacement_braces_4 = r'[\1, \2, \3, \4]'
    fixed_format = re.sub(pattern_braces_4, replacement_braces_4, fixed_format)

    # 5. Fix bare identifiers issue: convert u0, u1, t0 etc. to string format
    identifier_pattern = r'(?<!["\\])([ut]\d+)(?!["\\])'
    def quote_identifier(match):
        return f'"{match.group(1)}"'
    fixed_identifiers = re.sub(identifier_pattern, quote_identifier, fixed_format)
    print(f"After fixing identifiers: {fixed_identifiers}")
    

    # Check for missing closing braces at the end
    open_braces = fixed_identifiers.count('{')
    close_braces = fixed_identifiers.count('}')
    
    if open_braces > close_braces:
        missing_count = open_braces - close_braces
        print(f"Warning: Detected {missing_count} missing closing braces, auto-completing...")
        fixed_identifiers += '}' * missing_count


    # 6. First, try parsing as a single dictionary
    try:
        result = json.loads(fixed_identifiers)
        if isinstance(result, dict):
            return clean_keys(result)
    except json.JSONDecodeError:
        pass  # If fails, continue to try multi-dictionary parsing
    
    # 7. Find and merge all JSON dictionaries
    merged_dict = {}
    pos = 0
    while pos < len(fixed_identifiers):
        # Find the start of the next dictionary
        dict_start = fixed_identifiers.find('{', pos)
        if dict_start == -1:
            break
            
        # Find the corresponding closing brace
        brace_count = 1
        dict_end = dict_start + 1
        while dict_end < len(fixed_identifiers) and brace_count > 0:
            if fixed_identifiers[dict_end] == '{':
                brace_count += 1
            elif fixed_identifiers[dict_end] == '}':
                brace_count -= 1
            dict_end += 1
            
        if brace_count == 0:
            # Extract and parse this dictionary
            dict_str = fixed_identifiers[dict_start:dict_end]
            try:
                current_dict = json.loads(dict_str)
                if isinstance(current_dict, dict):
                    merged_dict.update(clean_keys(current_dict))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse dictionary: {str(e)}")
                # Try to fix potential comma issues
                try:
                    # Remove comma after the dictionary
                    dict_str_fixed = re.sub(r'\}\s*,\s*\{', '}', dict_str)
                    current_dict = json.loads(dict_str_fixed)
                    if isinstance(current_dict, dict):
                        merged_dict.update(clean_keys(current_dict))
                except json.JSONDecodeError as e2:
                    print(f"Warning: Still unable to parse dictionary after fix: {str(e2)}")
                    
        pos = dict_end
    
    if not merged_dict:
        raise ValueError("Unable to parse input JSON dictionary")
        
    print(f"{tool_name} parsed parameters (dictionary): {merged_dict}")
    return merged_dict

def clean_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    # d: dict
    new_d = {}
    for k, v in d.items():
        new_k = k.strip() # Remove leading/trailing whitespace from key names
        new_d[new_k] = v
    return new_d

# General helper function to convert string-form quadruple list to Python list
def parse_edge_list(edge_list_data: Union[str, list]) -> List[Tuple[int, int, float, str]]:
    """
    Converts a JSON string or already parsed list of quadruples into a Python list
    
    Args:
        edge_list_data: JSON string or list, representing a list of quadruples, e.g., [[0,1,0,"a"],[1,2,1,"a"]]
        
    Returns:
        List containing (u, v, t, op) quadruples, e.g., [(0, 1, 0, 'a'), (1, 2, 1, 'a')]
        
    Raises:
        ValueError: If the input format is incorrect
    """
    try:
        if isinstance(edge_list_data, str):
            # Fix single quote issue: replace single quotes with double quotes
            fixed_json = edge_list_data.replace("'", '"')
            edges_input = json.loads(fixed_json)
        elif isinstance(edge_list_data, list):
            edges_input = edge_list_data
        else:
            raise ValueError("edge_list_data must be a JSON string or a list")

        # Validate that the top level is a list
        if not isinstance(edges_input, list):
            raise ValueError("Input must be a list of quadruples")
        
        # Validate format and convert
        result = []
            
        for item in edges_input:
            u, v, t_rel, op = item
            # Check if node identifiers need conversion
            if isinstance(u, str) and u.startswith('u'):
                    u = int(u[1:])  # Extract 0 from 'u0'
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # Extract 1 from 'u1'
            else:
                v = int(v)
            if isinstance(t_rel, str) and t_rel.startswith('t'):
                t_rel = int(t_rel[1:])  # Extract 0 from 't0'
            else:
                t_rel = int(t_rel)
            op = str(op)
            result.append((u, v, t_rel, op))
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse JSON formatted edge list: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing edge list: {str(e)}")

@tool(
    "Judge_Is_Contain_Motif", # Modify tool name to conform to Python naming conventions and remove spaces
    description="""Detect whether the given undirected dynamic graph is(contains) the given temporal motif. Input should be a JSON string with keys: "edge_list", "motif_list"."""
)
def judge_is_contain_motif(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "judge_is_contain_motif")

        edge_list_data = params.get("edge_list")
        motif_definition_data = params.get("motif_list")

        edges = parse_edge_list(edge_list_data)
        for motif_name, motif_definition in motif_definition_data.items():
            motif = parse_edge_list(motif_definition["edge_pattern"])
            time_window = motif_definition["time_window"]   
        
        result_bool = judge(edges, motif, time_window)
        return "Yes" if result_bool == "Yes" else "No"
    except Exception as e:
        return f"Error (judge_is_contain_motif): {str(e)}"

@tool(
    "Modify_To_Contain_Motif", # Modify tool name to conform to Python naming conventions and remove spaces
    description="""Detect How to modify the given undirected dynamic graph so that it contains the specified temporal motif? Input should be a JSON string with keys: "edge_list", "motif_list"."""
)
def modify_to_contain_motif(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "modify_to_contain_motif")

        edge_list_data = params.get("edge_list")
        motif_definition_data = params.get("motif_list")

        edges = parse_edge_list(edge_list_data)
        
        # motif_definition_data is a dictionary, needs to be handled correctly
        for motif_name, motif_definition in motif_definition_data.items():
            motif = parse_edge_list(motif_definition["edge_pattern"])  # Parse the edge list here
            time_window = motif_definition["time_window"]   
        
        result_edge = modify(edges, motif, time_window)
        if result_edge:
            # modify function returns a tuple, ensure it can be handled by json.dumps
            return json.dumps(list(result_edge)) if isinstance(result_edge, tuple) else json.dumps(result_edge)
        else:
            return "Could not find a suitable edge to modify"
    except Exception as e:
        return f"Error (modify_to_contain_motif): {str(e)}"


@tool(
    "Detect_Motifs_Presence", # Modify tool name to conform to Python naming conventions and remove spaces
    description="""Detect What motifs present in the given undirected dynamic graph? Input should be a JSON string with keys: "edge_list" and "motif_definitions"."""
)
def detect_motifs_present(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "detect_motifs_present")
        
        edge_list_data = params.get("edge_list")
        motif_lists_data = params.get("motif_definitions")  # Update parameter name
        
        if edge_list_data is None or motif_lists_data is None:
            return "Error: detect_motifs_present missing required parameters 'edge_list' or 'motif_definitions'"
            
        edges = parse_edge_list(edge_list_data)
        print(f"detect_motifs_present parsed edge list: {edges}")

        result_multi_motif = multi_motif_judge(edges, motif_lists_data)
        return json.dumps(result_multi_motif)
    except Exception as e:
        return f"Error (detect_motifs_present): {str(e)}"


@tool(
    "Detect_Motifs_Counts", 
    description="""Detect How many times does each of the above temporal motifs appear in the given undirected dynamic graph? Input should be a JSON string with keys: "edge_list" and "motif_definitions"."""
)
def detect_motifs_counts(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "detect_motifs_counts")
        
        edge_list_data = params.get("edge_list")
        motif_lists_data = params.get("motif_definitions")  # Update parameter name
        
        if edge_list_data is None or motif_lists_data is None:
            return "Error: detect_motifs_counts missing required parameters 'edge_list' or 'motif_definitions'"
            
        edges = parse_edge_list(edge_list_data)

        result_counts = multi_motif_counts(edges, motif_lists_data)
        return json.dumps(result_counts)
    except Exception as e:
        return f"Error (detect_motifs_counts): {str(e)}"


@tool(
    "Detect_Motifs_First_Time", 
    description="""Detect When does each of the above temporal motifs first appear in the given undirected dynamic graph? Input should be a JSON string with keys: "edge_list" and "motif_definitions"."""
)
def detect_motifs_first_time(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "detect_motifs_first_time")
        
        edge_list_data = params.get("edge_list")
        motif_lists_data = params.get("motif_definitions")  # Update parameter name
        
        if edge_list_data is None or motif_lists_data is None:
            return "Error: detect_motifs_first_time missing required parameters 'edge_list' or 'motif_definitions'"
            
        edges = parse_edge_list(edge_list_data)

        result_first_times = multi_motif_first_time(edges, motif_lists_data)
        return json.dumps(result_first_times)
    except Exception as e:
        return f"Error (detect_motifs_first_time): {str(e)}"





@tool(
    "Sort_Edges_By_Time", # Modify tool name, remove space
    description="""Sort the edges by time from earliest to latest. Input should be a JSON string with key: "edge_list"."""
)
def sort_edges_by_time(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "sort_edges_by_time")
        
        edge_list_data = params.get("edge_list")
        if edge_list_data is None:
            return "Error: detect_sorted_edges missing required parameter 'edge_list'"
            
        edges = parse_edge_list(edge_list_data)
        print(f"detect_sorted_edges parsed edge list: {edges}")
        sorted_edges = sort_edges(edges)
        return json.dumps(sorted_edges)
    except Exception as e:
        return f"Error (detect_sorted_edges): {str(e)}"

@tool(
    "Detect_First_Connected_And_Disconnected_Time",
    description="""Detect when the first connected and first disconnected between given two node ids in given dynamic graph. Input should be a JSON string with keys: "edge_list", "u", and "v"."""
)
def detect_first_connected_and_disconnected_time(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "detect_first_connected_and_disconnected_time")

        edge_list_data = params.get("edge_list")
        u_node = params.get("u")
        v_node = params.get("v")

        if edge_list_data is None or u_node is None or v_node is None:
            return "Error: detect_first_connected_and_disconnected_time missing required parameters 'edge_list', 'u', or 'v'"
            
        edges = parse_edge_list(edge_list_data)
        # Ensure u and v are integers
        u = int(u_node)
        v = int(v_node)
        
        first_connected, first_disconnected = when_direct_link(edges, u, v)
        # Return JSON formatted list string
        return json.dumps([first_connected, first_disconnected])
    except Exception as e:
        return f"Error (detect_first_connected_and_disconnected_time): {str(e)}"

@tool(
    "Detect_Edges_Existence_At_Time",
    description="""Detect what edges exist at given time in given dynamic graph. Input should be a JSON string with keys: "edge_list" and "time_point"."""
)
def detect_edges_at_time(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "detect_edges_at_time")

        edge_list_data = params.get("edge_list")
        time_point_str = params.get("time_point")

        if edge_list_data is None or time_point_str is None:
            return "Error: detect_edges_at_time missing required parameters 'edge_list' or 'time_point'"
            
        edges = parse_edge_list(edge_list_data)
        time_point = float(time_point_str)
        
        result_edges = what_edge_at_time(edges, time_point)
        return json.dumps(result_edges)
    except Exception as e:
        return f"Error (detect_edges_at_time): {str(e)}"

@tool(
    "Reverse_Dynamic_Graph", # Modify tool name
    description="""Detect what is one possible sequence of quadruple operations that reverse the dynamic graph from the final time to empty. Input should be a JSON string with key: "edge_list"."""
)
def reverse_dynamic_graph(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "reverse_dynamic_graph")

        edge_list_data = params.get("edge_list")
        if edge_list_data is None:
            return "Error: reverse_dynamic_graph missing required parameter 'edge_list'"

        edges = parse_edge_list(edge_list_data)
        result_reversed_edges = reverse_graph(edges)
        return json.dumps(result_reversed_edges)
    except Exception as e:
        return f"Error (reverse_dynamic_graph): {str(e)}"

# Add all tools to the ALL_TOOLS list
# Ensure the names here match those defined in @tool
ALL_TOOLS = [
    judge_is_contain_motif,
    modify_to_contain_motif,
    detect_motifs_present,
    detect_motifs_counts,
    detect_motifs_first_time,
    sort_edges_by_time,
    detect_first_connected_and_disconnected_time,
    detect_edges_at_time,
    reverse_dynamic_graph
]