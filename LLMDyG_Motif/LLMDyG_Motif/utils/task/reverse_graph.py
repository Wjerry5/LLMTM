from .base import DyGraphTask
import numpy as np
import re
import os

# 辅助函数：模拟图演化以获取最终状态
# 移植并修改自 evaluate_2.py 的 get_final_state


# 辅助函数：应用模型生成的操作序列
# 移植并修改自 evaluate_2.py 的 apply_result_sequence
def apply_reverse_sequence(initial_edges, result_quadruples):
    """
    将LLM生成的操作序列应用于给定的初始边集合。

    Args:
        initial_edges (set): 初始的边集合 { (u, v), ... } (规范化 tuple)。
        result_quadruples (list): LLM结果解析出的四元组列表 [(u, v, t, op), ...] (int, int, int, str)。

    Returns:
        set: 应用操作序列后的最终边集合。
    """
    quad_list = initial_edges+result_quadruples
    quad_list = sorted(quad_list, key=lambda x: (x[2], 0 if x[3] == 'a' else 1))
    print(quad_list)
    simulated_edges = set()
    for u, v, t, op in quad_list:
        edge = tuple(sorted((u, v)))
        if op == 'a':
            simulated_edges.add(edge)
        elif op == 'd':
            simulated_edges.discard(edge)
    print(simulated_edges)
    return simulated_edges

# 辅助函数：从响应内容解析四元组列表（复用 WhatEdgeAtTime的）
def parse_quadruples_from_list_string(list_string):
    """从列表格式的字符串中解析四元组列表，支持多种格式"""
    quadruples = [] # 这里返回列表，因为顺序可能重要（尽管 apply 时不重要）
    
    # 策略1: 尝试解析嵌套列表格式 [[1, 5, 0, "d"], [5, 6, 1, "d"], ...]
    nested_pattern = r'\[\s*(?:\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[\'"]?[ad][\'"]?\s*\]\s*,?\s*)+\]'
    nested_match = re.search(nested_pattern, list_string)
    if nested_match:
        nested_content = nested_match.group(0)
        # 提取每个子列表
        sub_lists = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\]', nested_content)
        for u, v, t, op in sub_lists:
            try:
                quad = (int(u), int(v), int(t), op)
                quadruples.append(quad)
            except ValueError:
                print(f"  警告 (ReverseGraph - 解析): 跳过解析错误的嵌套列表元素: {(u, v, t, op)}")
        return quadruples
    
    # 策略2: 尝试解析元组格式 [(1, 5, 0, "d"), (5, 6, 1, "d"), ...]
    tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', list_string)
    for t in tuple_matches:
        try:
            quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
            quadruples.append(quad)
        except ValueError:
             print(f"  警告 (ReverseGraph - 解析): 跳过解析错误的元组: {t}")
    return quadruples


class DyGraphTaskReverseGraph(DyGraphTask):
    """
    任务：找出能将动态图从最终状态反转回空图的操作序列。
    对应原始脚本：How.py, evaluate_2.py
    """
    def generate_qa(self, info, *args, **kwargs):
        """
        生成 QA 对。此任务的 'answer' 比较特殊，评估是看应用模型输出后图是否为空。
        我们在 QA 中预先计算图的最终状态，方便评估。
        """
        context_orig = info['edge_index']
        # 类型转换和排序
        context_typed = []
        for item in context_orig:
            try:
                u, v, t, op = item
                context_typed.append((int(u), int(v), int(t), str(op)))
            except ValueError as e:
                print(f"警告 (ReverseGraph - QA): 跳过无法解析的行: {item} - {e}")
                continue
            except Exception as e:
                print(f"警告 (ReverseGraph - QA): 处理行时出错: {item} - {e}")
                continue

        if not context_typed:
             print("错误 (ReverseGraph - QA): 没有有效的上下文数据。")
             return None

        # 计算最终状态的边集合
        final_edges_set = get_final_state_edges(context_typed)

        qa = {
            "context": context_orig, # 原始格式上下文
            "query": None,          # query 为空，问题固定
            "answer": None,         # answer 为空，评估不直接比较答案
            "_final_edges_set": final_edges_set, # 存储最终状态用于评估
            "task": self.task
        }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        # 移植自 How.py
        return (f"Your task is to answer what is one possible sequence of quadruple operations that reverse the dynamic graph from the final time to empty?")

    def generate_instructor_answer(self, *args, **kwargs):
        # 移植自 How.py
        return "Give the answer as a list of 4-tuples at the end of your response after 'Answer:'."

    def generate_prompt_examplars(self, num, *args, **kwargs):
            # 移植自 How.py，确保格式正确
            qa_examples_raw = [
                (
                  [(1, 2, 0, 'a'), (1, 2, 1, 'd'), (2, 3, 3, 'a'), (0, 3, 4, 'a'), (2, 3, 4, 'd')],
                  None, # query
                    """**Chain of Thought:**
1. Objective: Transform the input list of dynamic graph operations by reversing each operation type and then reversing the entire sequence of operations.
2. Reverse Operations: Iterate through each 4-tuple (u, v, t, op_type) in the original edge list.
    * If op_type is 'a', change it to 'd'.
    * If op_type is 'd', change it to 'a'.
    * Keep u, v, and t the same for each tuple.
3. Reverse Sequence: Take the new list of transformed tuples and arrange them in reverse order. The last tuple becomes the first, the second to last becomes the second, and so on.
4. Final Output: Present the newly ordered list of 4-tuples as the result.""",  # CoT - 为了兼容base.py的4元素格式
                 [(2, 3, 4, 'a'), (0, 3, 4, 'd'), (2, 3, 3, 'd'), (1, 2, 1, 'a'), (1, 2, 0, 'd')]
                )
            ]


#[(0, 7, 0, a), (0, 9, 1, a), (1, 8, 1, a), (2, 8, 1, a), (0, 3, 2, a), (1, 8, 2, d), (8, 9, 2, a), (0, 5, 3, a), (0, 9, 3, d), (4, 9, 3, a), (6, 9, 3, a), (8, 9, 3, d), (0, 3, 4, d), (0, 5, 4, d), (3, 5, 4, a), (4, 9, 4, d), (5, 6, 4, a)]             
#[(5, 6, 4, d), (4, 9, 4, a), (3, 5, 4, d), (0, 5, 4, a), (0, 3, 4, a), (8, 9, 3, a), (6, 9, 3, d), (4, 9, 3, d), (0, 9, 3, a), (0, 5, 3, d), (8, 9, 2, d), (1, 8, 2, a), (0, 3, 2, d), (2, 8, 1, d), (1, 8, 1, d), (0, 9, 1, d), (0, 7, 0, d)]
            # 转换格式 - 修复：正确处理4个元素
            qa_formatted = [ [list(c), q, s, list(a)] for c, q, s, a in qa_examples_raw]
            return self.make_qa_example(num, qa_formatted)


    def generate_prompt_question(self, query = None, *args, **kwargs):
        # 移植自 How.py，问题是固定的，只依赖 context
        # context 已经在 generate_context_prompt 中处理了
        return f" What is one possible sequence of quadruple operations that reverse the dynamic graph from the final time to empty?"

    def evaluate(self, qa, response):
        """
        评估模型响应。移植自 evaluate_2.py。
        获取图的最终状态，应用模型生成的操作序列，检查结果图是否为空。
        支持多种格式：
        1. "Answer: [(1, 5, 0, 'd'), (5, 6, 1, 'd')]"
        2. "[(1, 5, 0, 'd'), (5, 6, 1, 'd')]" (直接列表格式)
        返回值:
            metric: 1 (正确，图为空), 0 (错误，图不为空), -1 (无法解析格式或 QA 数据错误)
            extracted_answer: 解析出的四元组列表或 None
        """
        final_original_edges = qa.get("context")
        predicted_quads_list = None
        
        # 策略1: 查找最后一个 "Answer:" 之后的内容
        answer_marker = "Answer:"
        answer_start_index = response.rfind(answer_marker)
        if answer_start_index != -1:
            extracted_part = response[answer_start_index + len(answer_marker):].strip()
            # 尝试解析列表内容
            predicted_quads_list = parse_quadruples_from_list_string(extracted_part)

        # 策略2: 如果没找到 "Answer:" 标记或解析失败，直接解析整个响应
        if predicted_quads_list is None or len(predicted_quads_list) == 0:
            predicted_quads_list = parse_quadruples_from_list_string(response)

        if predicted_quads_list is not None and len(predicted_quads_list) >= 0: # 解析成功（可能为空列表）
            # 应用模型生成的操作序列
            final_simulated_edges = apply_reverse_sequence(final_original_edges, predicted_quads_list)

            # 判断结果图是否为空
            if len(final_simulated_edges) == 0:
                metric = 1 # 正确
                print(f"  评估: 正确 - 图变为空。")
            else:
                metric = 0 # 错误
                print(f"  评估: 错误 - 图未变为空。剩余边: {final_simulated_edges}")
            return metric, predicted_quads_list
        else:
            # 解析返回 None
            return -1, None # 格式错误