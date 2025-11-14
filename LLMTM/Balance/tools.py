import os
import json
import numpy as np
import xgboost as xgb
import joblib
from collections import defaultdict, Counter # Import Counter for frequency statistics
import networkx as nx
from langchain.tools import tool
from typing import List, Tuple, Dict, Any, Optional, Union
import re
# Initialize tools list (will be overridden by definitions at the bottom)
ALL_TOOLS = []


def parse_tool_input(input_str: str, tool_name: str):
    """
    Generic tool input parsing function that handles all common format issues, including single and multiple JSON dictionaries
    
    Args:
        input_str: Raw input string
        tool_name: Tool name for debug output
        
    Returns:
        Parsed parameter dictionary
    """
    print(f"{tool_name} received parameters (raw string): {input_str}")
    
    # 0. Extract JSON part (content between the first { and the last })
    first_brace = input_str.find('{')
    last_brace = input_str.rfind('}')
    if first_brace != -1 and last_brace != -1:
        input_str = input_str[first_brace:last_brace + 1]
    
    # 1. Clean input: remove extra newlines and spaces
    cleaned_input = re.sub(r'\s+', ' ', input_str.strip())
    
    # 2. Fix single quote issues
    fixed_json = cleaned_input.replace("'", '"')
    
    # 3. Fix bare letters a and d (after commas or inside brackets)
    fixed_json = re.sub(r',\s*([ad])(?=\s*[)\]])', r', "\1"', fixed_json)
    
    # 4. Fix format issues: convert (1,2,3,a) format to [1,2,3,"a"] format
    pattern = r'\(\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*\)'
    replacement = r'[\1, \2, \3, \4]'
    fixed_format = re.sub(pattern, replacement, fixed_json)
    
    # 5. Fix bare identifier issues: convert u0, u1, t0 etc. to string format
    identifier_pattern = r'(?<!["\\])([ut]\d+)(?!["\\])'
    def quote_identifier(match):
        return f'"{match.group(1)}"'
    fixed_identifiers = re.sub(identifier_pattern, quote_identifier, fixed_format)
    print(f"After fixing identifiers: {fixed_identifiers}")
    
    # 6. First try to parse as a single dictionary
    try:
        result = json.loads(fixed_identifiers)
        if isinstance(result, dict):
            return clean_keys(result)
    except json.JSONDecodeError:
        pass  # If failed, continue to try multiple dictionary parsing
    
    # 7. Find and merge all JSON dictionaries
    merged_dict = {}
    pos = 0
    while pos < len(fixed_identifiers):
        # Find the start of the next dictionary
        dict_start = fixed_identifiers.find('{', pos)
        if dict_start == -1:
            break
            
        # Find the corresponding closing bracket
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
                # Try to fix possible comma issues
                try:
                    # Remove commas after dictionary
                    dict_str_fixed = re.sub(r'\}\s*,\s*\{', '}', dict_str)
                    current_dict = json.loads(dict_str_fixed)
                    if isinstance(current_dict, dict):
                        merged_dict.update(clean_keys(current_dict))
                except json.JSONDecodeError as e2:
                    print(f"Warning: Dictionary still cannot be parsed after fixing: {str(e2)}")
                
        pos = dict_end
    
    if not merged_dict:
        raise ValueError("Unable to parse input JSON dictionary")
        
    print(f"{tool_name} parsed parameters (dictionary): {merged_dict}")
    return merged_dict

def clean_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    # d: dict
    new_d = {}
    for k, v in d.items():
        new_k = k.strip() # Remove leading and trailing whitespace from key names
        new_d[new_k] = v
    return new_d

# Generic helper function to convert string form of quadruple list to Python list
def parse_edge_list(edge_list_data: Union[str, list]) -> List[Tuple[int, int, float, str]]:
    """
    Convert JSON string or already parsed list form of quadruple list to Python list
    
    Args:
        edge_list_data: JSON string or list representing quadruple list, like [[0,1,0,"a"],[1,2,1,"a"]]
        
    Returns:
        List containing (u, v, t, op) quadruples, like [(0, 1, 0, 'a'), (1, 2, 1, 'a')]
        
    Raises:
        ValueError: If input format is incorrect
    """
    try:
        if isinstance(edge_list_data, str):
            # Fix single quotes issue: replace single quotes with double quotes
            fixed_json = edge_list_data.replace("'", '"')
            edges_input = json.loads(fixed_json)
        elif isinstance(edge_list_data, list):
            edges_input = edge_list_data
        else:
            raise ValueError("edge_list_data must be a JSON string or list")

        # Verify top level is a list
        if not isinstance(edges_input, list):
            raise ValueError("Input must be a list of quadruples")
        
        # Validate format and convert
        result = []
            
        for item in edges_input:
            u, v, t_rel, op = item
            # Check if node identifiers need to be converted
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
        raise ValueError(f"Unable to parse edge list in JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error while parsing edge list: {str(e)}")



def analyze_graph_features(edges):
    """
    Analyze a series of general structural and cognitive complexity metrics of a graph.

    Metrics include:
    - num_nodes: Number of nodes
    - num_edges: Number of edges
    - edge_density: Graph density
    - num_nodes_ge_3: Number of nodes with degree >= 3 (measuring core complexity)
    - num_nodes_eq_2: Number of nodes with degree = 2 (measuring path complexity)
    - cyclomatic_complexity: Cyclomatic complexity (measuring graph "entanglement")
    - edge_locality_score: Edge locality score (measuring LLM's sequential cognitive load)
    - temporal_clash_count: Temporal clash count (measuring temporal dimension ambiguity)
    
    Args:
        edges (list): List of edges, each in the form (u, v, t, op).
    
    Returns:
        dict: A dictionary containing various general features of the graph.
    """
    # --- 1. Data preprocessing and graph construction ---
    G = nx.Graph()
    active_edges = [edge for edge in edges if edge[3] == 'a']
    
    # If there are no active edges, return a dictionary with default values
    if not active_edges:
        return {
            "num_nodes": 0, "num_edges": 0, "edge_density": 0,
            "num_nodes_ge_3": 0, "num_nodes_eq_2": 0, "cyclomatic_complexity": 1,
            "edge_locality_score": 0, "temporal_clash_count": 0, "ratio_nodes_ge_3": 0, "ratio_nodes_eq_2": 0
        }
    
    # Extract node and edge information from active edges to build the graph
    edge_tuples = [(u, v) for u, v, t, op in active_edges]
    G.add_edges_from(edge_tuples)
    
    # --- 2. Basic and structural complexity metrics ---
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Graph density
    if num_nodes > 1:
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        # Ensure the denominator is not 0
        edge_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    else:
        edge_density = 0
        
    # Cyclomatic complexity (E - N + P)
    # P = number of connected components
    num_connected_components = nx.number_connected_components(G)
    cyclomatic_complexity = num_edges - num_nodes + num_connected_components

    # Node degree related metrics
    degrees = dict(G.degree())
    num_nodes_ge_3 = sum(1 for degree in degrees.values() if degree >= 3)
    num_nodes_eq_2 = sum(1 for degree in degrees.values() if degree == 2)
    ratio_nodes_ge_3 = num_nodes_ge_3/num_nodes
    ratio_nodes_eq_2 = num_nodes_eq_2/num_nodes
    
    # Filter core nodes for calculating locality and temporal metrics
    core_nodes = [node for node, degree in degrees.items() if degree >= 3]

    # --- 3. Cognitive and temporal complexity metrics ---
    # Create a mapping from nodes to their related edge information (timestamps and indices)
    node_to_edge_info = defaultdict(list)
    for i, (u, v, t, op) in enumerate(edges):
        if op == 'a':
            info = {'t': t, 'idx': i}
            node_to_edge_info[u].append(info)
            if u != v:
                node_to_edge_info[v].append(info)

    # Edge locality score (calculated only on core nodes)
    center_locality_scores = []
    for node in core_nodes:
        indices = [info['idx'] for info in node_to_edge_info[node]]
        if len(indices) >= 2:
            center_locality_scores.append(np.std(indices))
    
    edge_locality_score = np.mean(center_locality_scores) if center_locality_scores else 0
    
    # Temporal clash count (calculated only on core nodes, then summed)
    total_clashes = 0
    for node in core_nodes:
        timestamps = [info['t'] for info in node_to_edge_info[node]]
        if not timestamps:
            continue
        time_counts = Counter(timestamps)
        for count in time_counts.values():
            if count > 1:
                total_clashes += count * (count - 1) / 2

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


# def predict_with_xgboost(features_dict, model_path=None, model_info=None):
#     """
#     Make predictions using a trained XGBoost model.
    
#     Parameters:
#     features_dict: Dictionary containing feature values, e.g.:
#                   {'num_nodes': 10, 'edge_density': 0.5, ...}
#     model_path: Path to the model file (must be provided if model_info is None)
#     model_info: Dictionary containing model information (must be provided if model_path is None)
    
#     Returns:
#     prediction: 0 indicates using the Agent, 1 indicates using the LLM
#     probability: Probability of using the LLM
#     """
#     if model_info is None and model_path is None:
#         raise ValueError("Either model_path or model_info must be provided")
    
#     if model_info is None:
#         try:
#             # model_info = joblib.load(model_path)
            
#             model = xgb.XGBClassifier()
#             model.load_model(model_path) # 加载 .json 文件
#         except Exception as e:
#             print(f"Error loading model: {str(e)}")
#             return None, None
    
#     # model = model_info['model']
#     # features = model_info['features']
    
#     # Build feature array
#     features = ["num_edges","cyclomatic_complexity","edge_locality_score","ratio_nodes_ge_3","ratio_nodes_eq_2"]
#     X = np.array([[features_dict[f] for f in features]])
    
#     # Make prediction
#     prediction = model.predict(X)[0]
#     probability = model.predict_proba(X)[0][1]  # Get probability of using the LLM
    
#     return prediction, probability


def predict_with_xgboost(features_dict, model_path=None, model_info=None):
    """
    使用训练好的XGBoost模型进行预测。
    
    参数：
    features_dict: 包含特征值的字典，例如：
                  {'num_nodes': 10, 'edge_density': 0.5, ...}
    model_path: 模型文件路径（如果model_info为None则必须提供）
    model_info: 模型信息字典（如果model_path为None则必须提供）
    
    返回：
    prediction: 0表示使用Agent，1表示使用LLM
    probability: 使用LLM的概率
    """
    if model_info is None and model_path is None:
        raise ValueError("必须提供model_path或model_info其中之一")
    
    if model_info is None:
        try:
            model_info = joblib.load(model_path)
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return None, None
    
    model = model_info['model']
    features = model_info['features']
    
    # 构建特征数组
    X = np.array([[features_dict[f] for f in features]])
    
    # 进行预测
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # 获取使用LLM的概率
    
    return prediction, probability

def predict(edges: List[Tuple[int, int, int, str]]) -> str:
    graph = edges
    features = analyze_graph_features(graph)
    # Use project-relative path to the bundled xgboost model info
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "xgboost.joblib")
    prediction, probability = predict_with_xgboost(features, model_path=model_path)
    threshold = 0.5
    use_llm = prediction
    if use_llm and probability > threshold:
        return "LLM"
    else:
        return "Agent"
@tool(
    "predict_llm_agent", # 修改工具名
    description="""Extract edge list from dynamic graph questions and predict whether the question can be answered directly by a Language Model (LLM) or requires an Agent with tools. Input should be a JSON string with key: "edge_list"."""
)
def predict_llm_agent (input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "predict_llm_agent")

        edge_list_data = params.get("edge_list")
        if edge_list_data is None:
            return "Error: predict_llm_agent missing required parameter 'edge_list'"

        # Handle cases where there may be multiple edge lists
        if isinstance(edge_list_data, list) and len(edge_list_data) > 0:
            # If it's a nested list, take the first list (main dynamic graph edges)
            if isinstance(edge_list_data[0], list) and len(edge_list_data[0]) > 0:
                if isinstance(edge_list_data[0][0], list):
                    # Nested edge list, take the first one
                    edges = parse_edge_list(edge_list_data[0])
                else:
                    # Single-level edge list
                    edges = parse_edge_list(edge_list_data)
            else:
                edges = parse_edge_list(edge_list_data)
        else:
            edges = parse_edge_list(edge_list_data)
            
        result = predict(edges)
        return json.dumps(result)
    except Exception as e:
        return f"Error (predict_llm_agent): {str(e)}"
    
ALL_TOOLS = [predict_llm_agent]