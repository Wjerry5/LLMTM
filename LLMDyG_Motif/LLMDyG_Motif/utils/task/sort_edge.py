from .base import DyGraphTask
import numpy as np
import re
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
def judge_ordered(con):
    con = np.array(con)
    for i in range(len(con)-1):
        if int(con[i,2]) > int(con[i+1,2]):
            return False
    return True
    
# def parse_quadruples_from_response_content(content):
#     """从模型响应内容中解析四元组列表"""
#     quadruples_list = []
    
#     # 直接查找所有的列表格式 [...]，并取最后一个
#     list_matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
    
#     if list_matches:
#         # 取最后一个匹配的列表
#         list_content = list_matches[-1].strip()
#         # 尝试多种匹配模式
#         tuple_matches = []
        
#         # 模式1: 标准格式 (0, 1, 0, 'a') 或 (0, 1, 0, a)
#         tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', list_content)
        
#         # 模式2: 更宽松的格式，允许可选的逗号
#         if not tuple_matches:
#             tuple_matches = re.findall(r'\(\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*([ad])\s*\)', list_content)
        
#         # 模式3: 最宽松的格式，只要是数字-数字-数字-字母的组合
#         if not tuple_matches:
#             tuple_matches = re.findall(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([ad])', list_content)
        
#         for t in tuple_matches:
#             try:
#                 # 解析为 (int, int, int, str)
#                 quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
#                 quadruples_list.append(quad)
#             except (ValueError, IndexError):
#                  pass  # 静默跳过解析错误的元组
    
#     return quadruples_list
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

class DyGraphTaskSortEdge(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info.get('edge_index', [])
        # 将context中的内容乱序
        import random
        context_shuffled = context.copy()
        random.shuffle(context_shuffled)
        context = context_shuffled
        # 确保 context 是 (int, int, int, str) 的列表
        context_typed = [(int(u), int(v), int(t), str(op)) for u, v, t, op in context]

        qa = {
            "context": context_typed,
            # query 对于排序任务通常为空
            "query": [],
            # answer 在这里可以设为空，因为评估是基于完整性和排序性
            "answer": [],
            "task": self.task
        }
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to SORT the edges by time from earliest to latest, Ensure that the number of edges remains consistent with the original."
        # return f"**Your task** is to **SORT** the edges by time from **Earliest to Latest**. Ensure that **the number of edges remains CONSISTENT** with the original input.\n\n"
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a list of 4-tuples at the end of your response after 'Answer:'. Do not output any other content (such as code, text or explanation)."
        # return "**Give the answer** as a **list of 4-tuples** at the end of your response, after `Answer:`.\n"
#     def generate_instructor_cot(self, *args, **kwargs):
# #         return """## Step-by-Step Reasoning:
# # 1. **Sort the Edges**: Sort all edges by the third element `t`, from smallest to largest, if there are multiple edges with the same timestamp, preserve their original order.
# # 2. **Edge Consistency**: Do not remove or duplicate any edges. The number of edges in the output must exactly match the input.
# # 3. **Output the Answer**: Return the sorted list of edges as a list of 4-tuples at the end of response."""
#         return """## Step-by-Step Reasoning:
# 1. **Determine Timestamps**: Scan the raw data to identify all the timestamps(the third element `t`) that appear, and sort them in ascending order.
# 2. **Group Data**: For each timestamp, scan the raw data to obtain the set of edges with the same timestamp.
# 3. **Merge Edge Sets**: Combine all the edge sets in the order of their timestamps to form the final list.
# 4. **Edge Consistency**: Ensure the number of edges in the output must exactly match the input.
# 5. **Output the Answer**: Return the sorted list of edges as a list of 4-tuples at the end of the response."""
#     
    def generate_prompt_examplars(self, num, *args, **kwargs):
        # 移植自 temporal_understand.py，并确保格式正确
        # 注意：需要确保这里的元组元素是正确的类型（int, int, int, str）
        qa_examples_raw = [
            (
            [(1, 8, 3, 'a'), (2, 9, 0, 'd'), (0, 7, 2, 'a'), (5, 6, 4, 'd'), (4, 7, 1, 'a'), (3, 8, 5, 'a'), (1, 6, 2, 'd'), (0, 5, 0, 'a'), (4, 9, 3, 'a'), (2, 5, 1, 'a'), (6, 7, 4, 'a'), (0, 3, 3, 'd'), (1, 5, 5, 'd'), (2, 8, 4, 'a'), (3, 9, 0, 'a')], # context (会被 make_qa_example 格式化)
            [], # query
            """**Chain of Thought:**
1. My goal is to sort the list of edges based on their timestamp. The timestamp is the third element in each tuple.
2. I'll go through the input list and group the edges by their timestamp, from the earliest to the latest.
    * t=0: I found three edges: (2, 9, 0, d), (0, 5, 0, a), and (3, 9, 0, a). I'll keep them in their original relative order.
    * t=1: I found two edges: (4, 7, 1, a) and (2, 5, 1, a). I'll keep them in their original relative order.
    * t=2: I found two edges: (0, 7, 2, a) and (1, 6, 2, d). I'll keep them in their original relative order.
    * t=3: I found three edges: (1, 8, 3, a), (4, 9, 3, a), and (0, 3, 3, d). I'll keep them in their original relative order.
    * t=4: I found two edges: (5, 6, 4, d) and (6, 7, 4, a). I'll keep them in their original relative order.
    * t=5: I found two edges: (3, 8, 5, a) and (1, 5, 5, d). I'll keep them in their original relative order.
3. Now, I will combines these groups in chronological order (0, 1, 2, 3, 4, 5) to create the final sorted list.
4. I'll do a final check: the input had 15 edges, and my sorted output also has 15 edges. The count is consistent.
5. This leads to the final answer.\n\n""", # COT
            [(2, 9, 0, 'd'), (0, 5, 0, 'a'), (3, 9, 0, 'a'), (4, 7, 1, 'a'), (2, 5, 1, 'a'), (0, 7, 2, 'a'), (1, 6, 2, 'd'), (1, 8, 3, 'a'), (4, 9, 3, 'a'), (0, 3, 3, 'd'), (5, 6, 4, 'd'), (6, 7, 4, 'a'), (2, 8, 4, 'a'), (3, 8, 5, 'a'), (1, 5, 5, 'd')]# answer (会被 make_qa_example 格式化)
            )
        ]
        # 转换内部表示以匹配 make_qa_example 的期望格式 list[list, list, list, list] - 修复：正确处理4个元素
        qa_formatted = [ [list(c), list(q), s, list(a)] for c, q, s, a in qa_examples_raw]

        return self.make_qa_example(num, qa_formatted)
        
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f" Sort the edges by time from earliest to latest." # 移除了 "in the dynamic graph" 以匹配原始脚本
    
    
    
    def evaluate(self, qa, response, *args, **kwargs):
        """
        评估模型响应的完整性和排序性。
        移植自 evaluate.py (Temporal)。
        返回值:
            metric: 1 (正确), -1 (排序错误), -2 (完整性错误), -3 (解析错误)
            extracted_answer_list: 解析出的四元组列表或 None
        """
        # 获取use_agent参数，默认为False
        use_agent = kwargs.get('use_agent', False)
        # 1. 解析模型响应
        if use_agent:

            # 格式1: [数字, 数字, 数字, "字符"] - 双引号，允许后面有任意字符
            quads = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*"(\w+)"\]', response)

            # 格式2: [数字, 数字, 数字, '字符'] - 单引号，允许后面有任意字符
            if not quads:
                quads = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*'(\w+)'\]", response)
            # 格式3: [数字, 数字, 数字, 字符] - 无引号，允许后面有任意字符
            if not quads:
                quads = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\w+)\]', response)
            response_quads_list = [(int(u), int(v), int(t), str(op)) for u, v, t, op in quads]
        else:
            response_quads_list = parse_quadruples_from_response_content(response)
        if response_quads_list is None or not response_quads_list: # 如果解析失败或为空列表
            # 检查原始 context 是否也为空
            if not qa.get("context"):
                 return 1, [] # 如果原始 context 也为空，视为空列表匹配，算正确
            return -1, None # 解析错误

        # 2. 完整性检查
        # 从 qa 中获取原始上下文集合 (应在 generate_qa 中创建)
        original_quads_set = set(tuple(item) for item in qa.get("context"))
        response_quads_set = set(response_quads_list)
        print(original_quads_set)
        print(response_quads_set)
        is_complete = (original_quads_set == response_quads_set)

        flag = -1
        # 3. 排序检查 (仅在完整性通过时进行)
        is_sorted = judge_ordered(response_quads_list) # 使用 judge_ordered 检查

        if is_complete and is_sorted:
            flag = 1
        elif is_complete and not is_sorted:
            flag = 2
        elif not is_complete and is_sorted:
            flag = 3
        elif not is_complete and not is_sorted:
            flag = 4
        return flag, response_quads_list # 正确
        
