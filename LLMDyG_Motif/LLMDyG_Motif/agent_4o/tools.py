"""
动态图分析工具集

本模块定义了用于动态图分析的工具集，可供LLM代理使用。
工具包括：图生成、motif检测、图修改和图统计等。
"""

from langchain.tools import tool
from typing import List, Tuple, Dict, Any, Optional, Union
import json
import numpy as np
import networkx as nx
import re

import sys # 导入 sys 模块用于路径操作
import os
# --- 将项目根目录添加到 sys.path ---
# 获取当前脚本文件的绝对路径
# 例如 /home/hb/LLMDyG_Motif/scripts/example/run_one_task.py
_current_file_path = os.path.abspath(__file__)
# 获取脚本所在目录 /home/hb/LLMDyG_Motif/scripts/example
_script_dir = os.path.dirname(_current_file_path)
# 获取项目根目录 /home/hb/LLMDyG_Motif (向上两级)
_project_root = os.path.dirname(os.path.dirname(_script_dir))
# 将项目根目录添加到 Python 解释器的模块搜索路径列表的开头
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# --- 路径添加完毕 ---

# 导入项目中的相关函数
# 注意：路径可能需要根据实际项目结构调整
try:
    from LLMDyG_Motif.utils.modif_judge_count import judge, modify, sort_edges, when_direct_link, what_edge_at_time, reverse_graph, multi_motif_judge, multi_motif_counts, multi_motif_first_time
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    from ..utils.modif_judge_count import judge, modify, sort_edges, when_direct_link, what_edge_at_time, reverse_graph, multi_motif_judge, multi_motif_counts, multi_motif_first_time

# 初始化工具列表（最终会被底部的定义覆盖）
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
    
    # 3. 修复裸露的字母 a 和 d（在逗号后面或括号内）
    fixed_json = re.sub(r',\s*([ad])(?=\s*[)\]])', r', "\1"', fixed_json)
    
    # 4. 修复格式问题：将 (1,2,3,a) 格式转换为 [1,2,3,"a"] 格式
    pattern = r'\(\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*\)'
    replacement = r'[\1, \2, \3, \4]'
    fixed_format = re.sub(pattern, replacement, fixed_json)
    
    # 5. 修复裸标识符问题：将 u0, u1, t0 等转换为字符串格式
    identifier_pattern = r'(?<!["\\])([ut]\d+)(?!["\\])'
    def quote_identifier(match):
        return f'"{match.group(1)}"'
    fixed_identifiers = re.sub(identifier_pattern, quote_identifier, fixed_format)
    print(f"修复标识符后: {fixed_identifiers}")
    
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

# 通用辅助函数，将字符串形式的四元组列表转换为Python列表
def parse_edge_list(edge_list_data: Union[str, list]) -> List[Tuple[int, int, float, str]]:
    """
    将JSON字符串或已解析的列表形式的四元组列表转换为Python列表
    
    Args:
        edge_list_data: JSON字符串或列表，表示四元组列表，如[[0,1,0,"a"],[1,2,1,"a"]]
        
    Returns:
        列表，包含(u, v, t, op)四元组，如[(0, 1, 0, 'a'), (1, 2, 1, 'a')]
        
    Raises:
        ValueError: 如果输入格式不正确
    """
    try:
        if isinstance(edge_list_data, str):
            # 修复单引号问题：将单引号替换为双引号
            fixed_json = edge_list_data.replace("'", '"')
            edges_input = json.loads(fixed_json)
        elif isinstance(edge_list_data, list):
            edges_input = edge_list_data
        else:
            raise ValueError("edge_list_data必须是JSON字符串或列表")

        # 验证顶层是否为列表
        if not isinstance(edges_input, list):
            raise ValueError("输入必须是一个四元组列表")
        
        # 验证格式并转换
        result = []
            
        for item in edges_input:
            u, v, t_rel, op = item
            # 检查是否需要转换节点标识符
            if isinstance(u, str) and u.startswith('u'):
                    u = int(u[1:])  # 从 'u0' 提取 0
            else:
                u = int(u)
            if isinstance(v, str) and v.startswith('u'):
                v = int(v[1:])  # 从 'u1' 提取 1
            else:
                v = int(v)
            if isinstance(t_rel, str) and t_rel.startswith('t'):
                t_rel = int(t_rel[1:])  # 从 't0' 提取 0
            else:
                t_rel = int(t_rel)
            op = str(op)
            result.append((u, v, t_rel, op))
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析JSON格式的边列表: {str(e)}")
    except Exception as e:
        raise ValueError(f"解析边列表时出错: {str(e)}")

@tool(
    "Judge_Is_Contain_Motif", # 修改工具名以符合Python命名习惯，并移除空格
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
        return f"错误 (judge_is_contain_motif): {str(e)}"

@tool(
    "Modify_To_Contain_Motif", # 修改工具名以符合Python命名习惯，并移除空格
    description="""Detect How to modify the given undirected dynamic graph so that it contains the specified temporal motif? Input should be a JSON string with keys: "edge_list", "motif_list"."""
)
def modify_to_contain_motif(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "modify_to_contain_motif")

        edge_list_data = params.get("edge_list")
        motif_definition_data = params.get("motif_list")

        edges = parse_edge_list(edge_list_data)
        
        # motif_definition_data 是一个字典，需要正确处理
        for motif_name, motif_definition in motif_definition_data.items():
            motif = parse_edge_list(motif_definition["edge_pattern"])  # 这里才解析边列表
            time_window = motif_definition["time_window"]  
        
        result_edge = modify(edges, motif, time_window)
        if result_edge:
            # modify 函数返回的是元组，确保它能被json.dumps处理
            return json.dumps(list(result_edge)) if isinstance(result_edge, tuple) else json.dumps(result_edge)
        else:
            return "未找到合适的修改边"
    except Exception as e:
        return f"错误 (modify_to_contain_motif): {str(e)}"


@tool(
    "Detect_Motifs_Presence", # 修改工具名以符合Python命名习惯，并移除空格
    description="""Detect What motifs present in the given undirected dynamic graph? Input should be a JSON string with keys: "edge_list" and "motif_definitions"."""
)
def detect_motifs_present(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "detect_motifs_present")
        
        edge_list_data = params.get("edge_list")
        motif_lists_data = params.get("motif_definitions")  # 更新参数名
        
        if edge_list_data is None or motif_lists_data is None:
            return "错误: detect_motifs_present缺少必要的参数 'edge_list' 或 'motif_definitions'"
            
        edges = parse_edge_list(edge_list_data)
        print(f"detect_motifs_present解析后的边列表: {edges}")

        result_multi_motif = multi_motif_judge(edges, motif_lists_data)
        return json.dumps(result_multi_motif)
    except Exception as e:
        return f"错误 (detect_motifs_present): {str(e)}"


@tool(
    "Detect_Motifs_Counts", 
    description="""Detect How many times does each of the above temporal motifs appear in the given undirected dynamic graph? Input should be a JSON string with keys: "edge_list" and "motif_definitions"."""
)
def detect_motifs_counts(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "detect_motifs_counts")
        
        edge_list_data = params.get("edge_list")
        motif_lists_data = params.get("motif_definitions")  # 更新参数名
        
        if edge_list_data is None or motif_lists_data is None:
            return "错误: detect_motifs_counts缺少必要的参数 'edge_list' 或 'motif_definitions'"
            
        edges = parse_edge_list(edge_list_data)

        result_counts = multi_motif_counts(edges, motif_lists_data)
        return json.dumps(result_counts)
    except Exception as e:
        return f"错误 (detect_motifs_counts): {str(e)}"


@tool(
    "Detect_Motifs_First_Time", 
    description="""Detect When does each of the above temporal motifs first appear in the given undirected dynamic graph? Input should be a JSON string with keys: "edge_list" and "motif_definitions"."""
)
def detect_motifs_first_time(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "detect_motifs_first_time")
        
        edge_list_data = params.get("edge_list")
        motif_lists_data = params.get("motif_definitions")  # 更新参数名
        
        if edge_list_data is None or motif_lists_data is None:
            return "错误: detect_motifs_first_time缺少必要的参数 'edge_list' 或 'motif_definitions'"
            
        edges = parse_edge_list(edge_list_data)

        result_first_times = multi_motif_first_time(edges, motif_lists_data)
        return json.dumps(result_first_times)
    except Exception as e:
        return f"错误 (detect_motifs_first_time): {str(e)}"





@tool(
    "Sort_Edges_By_Time", # 修改工具名，移除空格
    description="""Sort the edges by time from earliest to latest. Input should be a JSON string with key: "edge_list"."""
)
def sort_edges_by_time(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "sort_edges_by_time")
        
        edge_list_data = params.get("edge_list")
        if edge_list_data is None:
            return "错误: detect_sorted_edges缺少必要的参数 'edge_list'"
            
        edges = parse_edge_list(edge_list_data)
        print(f"detect_sorted_edges解析后的边列表: {edges}")
        sorted_edges = sort_edges(edges)
        return json.dumps(sorted_edges)
    except Exception as e:
        return f"错误 (detect_sorted_edges): {str(e)}"

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
            return "错误: detect_first_connected_and_disconnected_time缺少必要的参数 'edge_list', 'u', 或 'v'"
            
        edges = parse_edge_list(edge_list_data)
        # 确保 u 和 v 是整数
        u = int(u_node)
        v = int(v_node)
        
        first_connected, first_disconnected = when_direct_link(edges, u, v)
        # 返回JSON格式的列表字符串
        return json.dumps([first_connected, first_disconnected])
    except Exception as e:
        return f"错误 (detect_first_connected_and_disconnected_time): {str(e)}"

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
            return "错误: detect_edges_at_time缺少必要的参数 'edge_list' 或 'time_point'"
            
        edges = parse_edge_list(edge_list_data)
        time_point = float(time_point_str)
        
        result_edges = what_edge_at_time(edges, time_point)
        return json.dumps(result_edges)
    except Exception as e:
        return f"错误 (detect_edges_at_time): {str(e)}"

@tool(
    "Reverse_Dynamic_Graph", # 修改工具名
    description="""Detect what is one possible sequence of quadruple operations that reverse the dynamic graph from the final time to empty. Input should be a JSON string with key: "edge_list"."""
)
def reverse_dynamic_graph(input_str: str) -> str:
    try:
        params = parse_tool_input(input_str, "reverse_dynamic_graph")

        edge_list_data = params.get("edge_list")
        if edge_list_data is None:
            return "错误: reverse_dynamic_graph缺少必要的参数 'edge_list'"

        edges = parse_edge_list(edge_list_data)
        result_reversed_edges = reverse_graph(edges)
        return json.dumps(result_reversed_edges)
    except Exception as e:
        return f"错误 (reverse_dynamic_graph): {str(e)}"

# 将所有工具添加到ALL_TOOLS列表
# 确保这里的名称与 @tool 中定义的名称一致
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
