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
    通用的工具输入解析函数，处理所有常见的格式问题，包括单个和多个JSON字典的情况
    
    Args:
        input_str: 原始输入字符串
        tool_name: 工具名称，用于调试输出
        
    Returns:
        解析后的参数字典
    """
    print(f"{tool_name}收到的参数 (原始字符串): {input_str}")
    
    # 0. 提取JSON部分（在第一个{和最后一个}之间的内容）
    first_brace = input_str.find('{')
    last_brace = input_str.rfind('}')
    if first_brace != -1 and last_brace != -1:
        input_str = input_str[first_brace:last_brace + 1]
    

    # 1. 清理输入：移除多余的换行符和空格
    cleaned_input = re.sub(r'\s+', ' ', input_str.strip())
    
    # 2. 修复单引号问题
    fixed_json = cleaned_input.replace("'", '"')
    fixed_json = fixed_json.replace(r'\"', '"')
    fixed_json = re.sub(r'\}\s*,\s*("([^"]+)"\s*:\s*\{)', r', \1', fixed_json)
    
    # 3.1. 修复带引号的数字：将 "5" 转换为 5 (但不能是字典键)
    fixed_json = re.sub(r'"(\d+)"(?!\s*:)', r'\1', fixed_json)

    # 3.2. [关键修复] 修复裸露的字母 a 和 d
    fixed_json = re.sub(r',\s*([ad])(?=\s*[)\]\}])', r', "\1"', fixed_json)

    # 4. 修复格式问题：将 (1,2,3,a) 格式转换为 [1,2,3,"a"] 格式
    pattern = r'\(\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*\)'
    replacement = r'[\1, \2, \3, \4]'
    fixed_format = re.sub(pattern, replacement, fixed_json)
    
    # 4.3. [新增] 修复 {5,4,3,"a"} 格式的 4 元组
    #    [^,:}]+ 确保我们只匹配元组，而不匹配字典
    #    例如: {5,4,3,"a"} -> [5, 4, 3, "a"]
    pattern_braces_4 = r'\{\s*([^,:}]+?)\s*,\s*([^,:}]+?)\s*,\s*([^,:}]+?)\s*,\s*([^,:}]+?)\s*\}'
    replacement_braces_4 = r'[\1, \2, \3, \4]'
    fixed_format = re.sub(pattern_braces_4, replacement_braces_4, fixed_format)

    # 5. 修复裸标识符问题：将 u0, u1, t0 等转换为字符串格式
    identifier_pattern = r'(?<!["\\])([ut]\d+)(?!["\\])'
    def quote_identifier(match):
        return f'"{match.group(1)}"'
    fixed_identifiers = re.sub(identifier_pattern, quote_identifier, fixed_format)
    print(f"修复标识符后: {fixed_identifiers}")
    

    # 检查末尾缺失的大括号
    open_braces = fixed_identifiers.count('{')
    close_braces = fixed_identifiers.count('}')
    
    if open_braces > close_braces:
        missing_count = open_braces - close_braces
        print(f"警告：检测到 {missing_count} 个缺失的右大括号，正在自动补全...")
        fixed_identifiers += '}' * missing_count


    # 6. 首先尝试作为单个字典解析
    try:
        result = json.loads(fixed_identifiers)
        if isinstance(result, dict):
            return clean_keys(result)
    except json.JSONDecodeError:
        pass  # 如果失败，继续尝试多字典解析
    
    # 7. 查找并合并所有JSON字典
    merged_dict = {}
    pos = 0
    while pos < len(fixed_identifiers):
        # 找到下一个字典的开始
        dict_start = fixed_identifiers.find('{', pos)
        if dict_start == -1:
            break
            
        # 找到对应的结束括号
        brace_count = 1
        dict_end = dict_start + 1
        while dict_end < len(fixed_identifiers) and brace_count > 0:
            if fixed_identifiers[dict_end] == '{':
                brace_count += 1
            elif fixed_identifiers[dict_end] == '}':
                brace_count -= 1
            dict_end += 1
            
        if brace_count == 0:
            # 提取并解析这个字典
            dict_str = fixed_identifiers[dict_start:dict_end]
            try:
                current_dict = json.loads(dict_str)
                if isinstance(current_dict, dict):
                    merged_dict.update(clean_keys(current_dict))
            except json.JSONDecodeError as e:
                print(f"警告：解析字典失败: {str(e)}")
                # 尝试修复可能的逗号问题
                try:
                    # 移除字典后的逗号
                    dict_str_fixed = re.sub(r'\}\s*,\s*\{', '}', dict_str)
                    current_dict = json.loads(dict_str_fixed)
                    if isinstance(current_dict, dict):
                        merged_dict.update(clean_keys(current_dict))
                except json.JSONDecodeError as e2:
                    print(f"警告：修复后仍然无法解析字典: {str(e2)}")
                
        pos = dict_end
    
    if not merged_dict:
        raise ValueError("无法解析输入的JSON字典")
        
    print(f"{tool_name}解析后的参数 (字典): {merged_dict}")
    return merged_dict

def clean_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    # d: dict
    new_d = {}
    for k, v in d.items():
        new_k = k.strip() # 去除键名首尾的空白字符
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
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "xgboost_Pangu.joblib")
    prediction, probability = predict_with_xgboost(features, model_path=model_path)
    threshold = 0.5
    use_llm = prediction
    if use_llm and probability > threshold:
        return "LLM"
    else:
        return "Agent"
@tool(
    "predict_llm_agent", 
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
