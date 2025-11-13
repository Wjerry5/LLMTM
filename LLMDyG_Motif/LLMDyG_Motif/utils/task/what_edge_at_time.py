from .base import DyGraphTask
import numpy as np
import re
import random

# 辅助函数：模拟图演化以找到特定时间存在的边
# 移植并修改自 QA_1.py 的 process_temporal_graph
def get_existing_edges(temporal_edges_typed, selected_time):
    """
    处理单个时序图数据，找出在 selected_time 时刻存在的边。

    参数:
    temporal_edges_typed - 按时间排序的类型化四元组列表 [(u, v, t, op), ...] (int, int, int, str)
    selected_time - 目标时间戳 (int)

    返回:
    existing_edges_at_t - 在selected_time存在的边的原始添加四元组列表 [(u, v, t_add, 'a'), ...]
    """
    if not temporal_edges_typed:
        return []

    current_edges = {} # {(u, v): original_add_quadruple}

    # 遍历所有时间 <= selected_time 的事件
    for u, v, t_event, op in temporal_edges_typed:
        if t_event > selected_time:
             break # 因为已排序，后续事件时间更大，无需处理

        edge = tuple(sorted((u, v))) # 规范化边表示 (int, int)

        if op == 'a':
            # 记录添加操作，存储原始四元组 (int, int, int, str)
            current_edges[edge] = (u, v, t_event, op)
        elif op == 'd':
            # 如果边存在，则删除
            if edge in current_edges:
                del current_edges[edge]

    # 提取在selected_time结束时存在的边的原始四元组
    existing_edges_at_t = list(current_edges.values())

    return existing_edges_at_t

# 辅助函数：从列表格式字符串解析四元组
# 移植自 evaluate_1.py
# def parse_quadruples_from_list_string(list_string):
#     """从列表格式的字符串中解析四元组集合"""
#     quadruples = set()
#     # 查找所有元组，例如 (0, 1, 0, 'a') 或 (0, 1, 0, a)
#     # 稍微宽松的正则，允许引号可选
#     tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', list_string)
#     for t in tuple_matches:
#         try:
#             quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
#             quadruples.add(quad)
#         except ValueError:
#              print(f"  警告 (WhatEdgeAtTime - 解析): 跳过解析错误的元组: {t}")
#     return quadruples

def _parse_text_for_quadruples(text_to_search):
    """
    辅助函数：在给定的文本块上运行所有正则表达式模式来查找元组。
    """
    quadruples_list = []
    
    # 模式1: 标准格式 (0, 1, 0, 'a') 或 (0, 1, 0, a)
    tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', text_to_search)
    
    # 模式2: 更宽松的格式 (保留你原来的逻辑)
    if not tuple_matches:
        tuple_matches = re.findall(r'\(\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*([ad])\s*\)', text_to_search)
    
    # 模式3: 最宽松的格式 (保留你原来的逻辑)
    if not tuple_matches:
        tuple_matches = re.findall(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([ad])', text_to_search)

    for t in tuple_matches:
        try:
            quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
            quadruples_list.append(quad)
        except (ValueError, IndexError):
            pass
    
    return quadruples_list
def parse_quadruples_from_response_content(content):
    """
    从模型响应内容中按优先级顺序解析四元组列表
    P1: [...] -> P2: Answer: -> P3: 全文
    """
    
    # --- 优先级 1: 查找 [...] ---
    # 查找所有 [...]，并取最后一个
    list_matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
    if list_matches:
        last_list_content = list_matches[-1].strip()
        result = _parse_text_for_quadruples(last_list_content)
        # 只有当 [...] 里的内容 *真的* 包含元组时，才返回
        if result:
            return result
    
    # --- 优先级 2: 查找 "Answer:" ---
    # (如果 P1 失败，则执行 P2)
    # re.IGNORECASE 忽略大小写, re.DOTALL 使 . 匹配换行符
    answer_match = re.search(r'Answer:\s*(.*)', content, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        result = _parse_text_for_quadruples(answer_content)
        if result:
            return result
    # --- 优先级 3: 查找全部内容 ---
    # (如果 P1 和 P2 都失败，则执行 P3)
    # 辅助函数会在这里解析完整的 content 字符串
    return _parse_text_for_quadruples(content)

class DyGraphTaskWhatEdgeAtTime(DyGraphTask):
    """
    任务：询问在给定时间点动态图中存在哪些边。
    对应原始脚本：What.py, QA_1.py, evaluate_1.py
    """
    def generate_qa(self, info, *args, **kwargs):
        """
        生成 QA 对。移植自 QA_1.py。
        随机选择一个时间点，计算该时间点存在的边。
        """
        context_orig = info['edge_index']
        # 类型转换和排序
        context_typed = []
        timestamps = set()
        for item in context_orig:
            try:
                u, v, t, op = item
                u_int, v_int, t_int = int(u), int(v), int(t)
                op_str = str(op)
                context_typed.append((u_int, v_int, t_int, op_str))
                timestamps.add(t_int)
            except ValueError as e:
                print(f"警告 (WhatEdgeAtTime - QA): 跳过无法解析的行: {item} - {e}")
                continue
            except Exception as e:
                print(f"警告 (WhatEdgeAtTime - QA): 处理行时出错: {item} - {e}")
                continue

        if not context_typed or not timestamps:
            print("错误 (WhatEdgeAtTime - QA): 没有有效的上下文数据或时间戳。")
            return None # 无法生成 QA

        # 按时间排序
        context_typed.sort(key=lambda x: x[2])

        # 随机选择一个有效的时间戳
        selected_time = random.choice(list(timestamps))

        # 计算在该时间点存在的边 (使用辅助函数)
        existing_edges = get_existing_edges(context_typed, selected_time)

        # 答案是存在边的列表
        answer = existing_edges # 已经是 [(u, v, t_add, 'a'), ...] 格式

        qa = {
            "context": context_orig, # 原始格式的上下文
            "query": selected_time, # 查询的是时间点
            "answer": answer,       # 答案是四元组列表
            "task": self.task
        }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        # 移植自 What.py
        # return (f"Your task is to answer what edges exist at a given moment in dynamic graph? (The order of edges doesn't matter)")
        return (f"Your task is to answer what edges exist at a given moment in dynamic graph? (The order of edges doesn't matter)")
    def generate_instructor_answer(self, *args, **kwargs):
        # 移植自 What.py
        return "Give the answer as a list of 4-tuples (u, v, t, a/d) at the end of your response after 'Answer:'."

    def generate_prompt_examplars(self, num, *args, **kwargs):
        # 移植自 What.py，确保格式正确
        qa_examples_raw = [
            (
                [(0, 9, 0, 'a'), (1, 9, 0, 'a'), (2, 5, 0, 'a'), (1, 2, 1, 'a'), (2, 6, 1, 'a'), (3, 7, 1, 'a'), (1, 9, 2, 'd'), (4, 5, 2, 'a'), (4, 7, 2, 'a'), (7, 8, 2, 'a')],
                1, # query (time)
                """**Chain of Thought:**
1. Objective: Determine which edges are active (connected) at the specific query_time of 1. An edge is active if it was added at or before time 1 and has NOT been deleted at or before time 1.
2. Initialize Edge Status Tracker: Prepare a record to track the current state of each unique edge (u, v). Initially, assume no edges exist.
3. Process Events Chronologically (up to query_time=1):
    * Go through each event (u, v, t, operation) in the input list.
    * Crucial Filter: If t > 1, ignore this event as it happens after the moment we are interested in.
    * For events where t <= 1:
        * If operation is 'a': Add the edge (u, v, t, a) to our set of currently existing edges.
        * If operation is 'd': Remove the edge (u, v, t, a) from our set of currently existing edges.
4. Final Active Edges at t=1: After processing all events that occurred at or before time 1, the edges remaining in our collection are the ones that exist at t=1.
5. Output: Present this final list of active edges.""", # CoT (思路) - 为了兼容base.py的4元素格式
                [(0, 9, 0, 'a'), (1, 9, 0, 'a'), (2, 5, 0, 'a'), (1, 2, 1, 'a'), (2, 6, 1, 'a'), (3, 7, 1, 'a')], # answer (edges existing at t=1)
            )
        ]
        # 转换格式 - 修复：正确处理4个元素
        qa_formatted = [ [list(c), q, s, list(a)] for c, q, s, a in qa_examples_raw]
        return self.make_qa_example(num, qa_formatted)
        
    def generate_prompt_question(self, query = None, *args, **kwargs):
        # 移植自 What.py
        selected_time = query
        return f" What edges exist at time {selected_time} in dynamic graph?"

    def evaluate(self, qa, response, use_agent):
        """
        评估模型响应。移植自 evaluate_1.py。
        解析模型输出的四元组列表，并与答案集合比较。
        返回值:
            metric: 1 (正确), 0 (错误但格式正确), -1 (无法解析格式)
            extracted_answer: 解析出的四元组集合或 None
        """
        true_quads_set = set(tuple( (int(u), int(v), int(t), str(op)) ) for u,v,t,op in qa['answer'])

#         # 尝试多种方式提取答案
#         extracted_part = None
        
# # 查找最后一个 "Answer:" 之后的内容
#         answer_marker = "Answer:"
#         answer_indices = [m.start() for m in re.finditer(re.escape(answer_marker), response)]
        
#         if len(answer_indices) >= 2:
#             # 如果有多个Answer，取倒数第二个Answer到最后一个Answer之间的内容
#             start_index = answer_indices[-2]
#             end_index = answer_indices[-1]
#             extracted_part = response[start_index + len(answer_marker):end_index].strip()
#         elif len(answer_indices) == 1:
#             # 如果只有一个Answer，取该Answer到结尾的内容
#             start_index = answer_indices[0]
#             extracted_part = response[start_index + len(answer_marker):].strip()
#         else:
#             # 如果没有找到Answer标记
#             return -1, None
#         # extracted_part = response.split(answer_marker)[-1].strip()
#         # 尝试解析列表内容


#         # 1. 首先尝试查找最后一个 "Answer:" 之后的内容
#         # answer_markers = ["Answer:", "答案:", "The answer is:", "最终答案:", "Final answer:"]
#         # for marker in answer_markers:
#         #     if marker in response:
#         #         answer_indices = [m.start() for m in re.finditer(re.escape(marker), response)]
#         #         if answer_indices:
#         #             # 取最后一个标记之后的内容
#         #             start_index = answer_indices[-1]
#         #             extracted_part = response[start_index + len(marker):].strip()
#         #             break
        
#         # # 2. 如果没找到标记，尝试直接解析整个响应
#         # if extracted_part is None:
#         #     # 尝试直接从响应中提取
#         #     extracted_part = response.strip()
        
#         # # 3. 如果提取的内容中包含多个段落，优先使用最后一个非空段落
#         # if extracted_part:
#         #     paragraphs = [p.strip() for p in extracted_part.split('\n\n') if p.strip()]
#         #     if paragraphs:
#         #         extracted_part = paragraphs[-1]

#         # 4. 尝试解析列表内容
        

        extracted_part = parse_quadruples_from_response_content(response)
        predicted_quads_set = set(extracted_part)

        if predicted_quads_set is not None: # 如果解析成功 (可能为空集)
            metric = 1 if predicted_quads_set == true_quads_set else 0
            if metric == 0:
                print(f"  基准集合: {true_quads_set}") # 调试用
                print(f"  模型集合: {predicted_quads_set}") # 调试用
                print(f"  提取的文本: {extracted_part}") # 调试用
            return metric, predicted_quads_set
        else:
            # 解析返回 None 通常意味着内部有警告，但这里我们视为格式错误
            print(f"  警告: 无法从以下文本中解析出有效的四元组:\n{extracted_part}")
            return -1, None # 格式错误 