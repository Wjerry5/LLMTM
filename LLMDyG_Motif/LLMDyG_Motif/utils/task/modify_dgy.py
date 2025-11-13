from LLMDyG_Motif.utils.modif_judge_count import judge, check_temporal_constraints
from .base import DyGraphTask
import re
# def parse_quadruples_from_response_content(content):
#     """从模型响应内容中解析四元组列表，支持多种格式"""
#     quadruples_list = []
    
#     def extract_quadruples_from_text(text):
#         """从文本中提取四元组"""
#         quads = []
        
#         # 方法1：查找元组格式 (1, 2, 3, 'a') 或 (1, 2, 3, "a")
#         tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', text)
#         for t in tuple_matches:
#             try:
#                 quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
#                 quads.append(quad)
#             except ValueError:
#                 print(f"  警告 (解析): 跳过解析错误的元组: {t}")
        
#         # 方法2：查找列表格式 [1, 2, 3, "a"] 或 [1, 2, 3, 'a']
#         if not quads:
#             list_matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\]', text)
#             for t in list_matches:
#                 try:
#                     quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
#                     quads.append(quad)
#                 except ValueError:
#                     print(f"  警告 (解析): 跳过解析错误的列表: {t}")
        
#         return quads
    
#     # 策略1：查找最后一个 "Answer:" 标记后的内容
#     answer_marker = "Answer:"
#     answer_start_index = content.rfind(answer_marker)
#     if answer_start_index != -1:
#         list_part = content[answer_start_index + len(answer_marker):].strip()
#         # 使用正则表达式查找列表内容，例如 [...]
#         match = re.search(r'\[(.*?)\]', list_part, re.DOTALL)
#         if match:
#             list_content = match.group(1).strip()
#             # 查找所有元组，例如 (0, 1, 0, 'a') 或 (0, 1, 0, a)
#             tuple_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?([ad])[\'"]?\s*\)', list_content)
#             for t in tuple_matches:
#                 try:
#                     # 解析为 (int, int, int, str)
#                     quad = (int(t[0]), int(t[1]), int(t[2]), t[3])
#                     quadruples_list.append(quad)
#                 except ValueError:
#                      print(f"  警告 (SortEdge - 解析): 跳过解析错误的元组: {t}")
#         else:
#             # 如果没找到列表格式，尝试直接解析Answer:后的内容
#             quadruples_list = extract_quadruples_from_text(list_part)
#     else:
#         # 策略2：如果没找到Answer:标记，尝试直接解析整个response
#         # quadruples_list = extract_quadruples_from_text(content)
    
#     # 策略3：如果还没找到，尝试解析特定的单个四元组格式
#     # if not quadruples_list:
#     #     # 查找形如 [1, 2, 3, "a"] 的单个四元组
#     #     single_quad_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]([ad])[\'"]\s*\]'
#     #     single_match = re.search(single_quad_pattern, content)
#     #     if single_match:
#     #         try:
#     #             quad = (int(single_match.group(1)), int(single_match.group(2)), 
#     #                    int(single_match.group(3)), single_match.group(4))
#     #             quadruples_list.append(quad)
#     #         except ValueError:
#     #             print(f"  警告 (解析): 跳过解析错误的单个四元组")
    
#     # if not quadruples_list:
#         print(f"  警告 (解析): 未能在响应中找到任何有效的四元组")
    
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

class DyGraphTaskModifyDyG(DyGraphTask):
    """
    任务：修改动态图使得包含指定的motif
    """
    def __init__(self, task, args):
        super().__init__(task, args)
        self.motif_name = args.motif_name

    def generate_qa(self, info, *args, **kwargs):
        """
        生成 QA 对。根据生成信息确定真实答案 ("Yes" 或 "No")。(中文注释)
        """
        context = info.get('edge_index', []) # 待判断的动态图边列表
        original_motif = info.get('target_motif', []) # Motif 定义
        # 确保 context 和 predefined_motif 是正确的类型列表
        context = [(int(u), int(v), int(t), str(op)) for u, v, t, op in context]
        N = info.get('motif_nodes')
        M = info.get('motif_edges')
        T = info.get('W')
        answer = info.get('answer',[])
        qa = {
            "context": context,       # 待判断的图
            "query": [original_motif, [N, M, T], [self.motif_name]], # 对于判断任务，query 通常嵌入在问题模板中
            "answer": answer,               # 真实答案 "Yes" 或 "No"
            "task": self.task,
            }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        """生成任务指令 (中文注释)"""
        return "Your task is to answer How to modify the given undirected dynamic graph so that it contains the given temporal motif?"
    def generate_instructor_answer(self, *args, **kwargs):
        """生成答案格式指令 (中文注释)"""

        return "Give the answer as a list of 4-tuples at the end of your response after 'Answer:'"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        """生成 Few-shot 示例 (中文注释)"""
        if self.motif_name == "4-cycle":
            qa_examples_raw = [
                (
                    [(8, 9, 3, 'a'), (3, 7, 4, 'a'), (0, 3, 5, 'a'), (0, 3, 5, 'd'), (3, 7, 5, 'd'), (0, 5, 6, 'a'), (0, 5, 6, 'd'), (2, 7, 6, 'a'), (8, 9, 7, 'd'), (2, 7, 8, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')],
                    [4, 4, 5],
                    ["4-cycle"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "4-cycle" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 2, 3, 5, 7, 8, 9
    * Edges: (8, 9, 3, 'a'), (3, 7, 4, 'a'), (0, 3, 5, 'a'), (0, 5, 6, 'a'), (2, 7, 6, 'a')
3. Since the current graph definitely doesn't contain "4-cycle", I will now search the graph for the "4-cycle with its last edge removed" temporal motif.
    * The pattern for a "4-cycle with its last edge removed" is a sequence of four edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 4.
    * In this undirected graph, the edges (7, 3, 4, 'a'), (3, 0, 5, 'a'), (0, 5, 6, 'a' ) form the above pattern.
    * The timestamps 4, 5, 6 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 6 - 4 = 2, 2 < 4, which satisfies the time window constraint.
4. Therefore, I will add the last edge (5, 7, 7, 'a') within the time window to make the graph contain a complete "4-cycle" temporal motif.
5. This edge is the final answer.""",
                    [(5, 7, 7, 'a')] # answer (会被 make_qa_example 格式化)
                )
            ]
        elif self.motif_name == "4-tailedtriangle":
            qa_examples_raw = [
                (
                    [(1, 6, 0, 'a'), (0, 1, 4, 'a'), (1, 6, 4, 'a'), (1, 2, 5, 'a'), (2, 3, 6, 'a'), (2, 3, 6, 'd'), (0, 1, 7, 'd'), (6, 8, 7, 'a'), (1, 2, 8, 'd'), (6, 8, 8, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')],
                    [4, 4, 5],
                    ["4-tailedtriangle"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "4-tailedtriangle" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 6, 8
    * Edges: (1, 6, 0, 'a'), (0, 1, 4, 'a'), (1, 6, 4, 'a'), (1, 2, 5, 'a'), (2, 3, 6, 'a'), (6, 8, 7, 'a')
3. Since the current graph definitely doesn't contain "4-tailedtriangle", I will now search the graph for the "4-tailedtriangle with its last edge removed" temporal motif.
    * The pattern for a "4-tailedtriangle with its last edge removed" is a sequence of five edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 4.
    * In this undirected graph, the edges (0, 1, 4, 'a'), (1, 2, 5, 'a'), (2, 3, 6, 'a') form the above pattern.
    * The timestamps 4, 5, 6 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 6 - 4 = 2, 2 < 4, which satisfies the time window constraint.
4. Therefore, I will add the last edge (3, 1, 7, 'a') within the time window to make the graph contain a complete "4-tailedtriangle" temporal motif.
5. This edge is the final answer.""",
                    [(3, 1, 7, 'a')] # answer (会被 make_qa_example 格式化)
                )
            ]
        elif self.motif_name == "4-clique":
            qa_examples_raw = [
                (
                    [(2, 4, 1, 'a'), (0, 3, 4, 'a'), (0, 7, 5, 'a'), (0, 8, 7, 'a'), (7, 8, 8, 'a'), (0, 7, 9, 'd'), (6, 8, 10, 'a'), (4, 7, 11, 'a'), (3, 8, 12, 'a'), (0, 8, 14, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')],
                    [4, 6, 10],
                    ["4-clique"],
                     """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "4-clique" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 2, 3, 4, 6, 7, 8
    * Edges: (2, 4, 1, 'a'), (0, 3, 4, 'a'), (0, 7, 5, 'a'), (0, 8, 7, 'a'), (7, 8, 8, 'a'), (6, 8, 10, 'a'), (4, 7, 11, 'a'), (3, 8, 12, 'a')
3. Since the current graph definitely doesn't contain "4-clique", I will now search the graph for the "4-clique with its last edge removed" temporal motif.
    * The pattern for a "4-clique with its last edge removed" is a sequence of five edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 9.
    * In this undirected graph, the edges (3, 0, 4, 'a'), (0, 7, 5, 'a'), (0, 8, 7, 'a'), (7, 8, 8, 'a'), (8, 3, 12, 'a') form the above pattern.
    * The timestamps 4, 5, 7, 8, 12 are strictly increasing. 
    * The difference between the maximum and minimum timestamp is 12 - 4 = 8, 8 < 9, which satisfies the time window constraint.
4. Therefore, I will add the last edge (3, 7, 13, 'a') within the time window to make the graph contain a complete "4-clique" temporal motif.
5. This edge is the final answer.""",
                    [(3, 7, 13, 'a')] # answer (会被 make_qa_example 格式化)
                )
            ]
        elif self.motif_name == "4-chordalcycle":
            qa_examples_raw = [
                (
                    [(3, 4, 2, 'a'), (6, 8, 3, 'a'), (2, 8, 4, 'a'), (0, 2, 7, 'a'), (0, 2, 8, 'd'), (0, 7, 8, 'a'), (2, 7, 10, 'a'), (1, 5, 13, 'a'), (3, 4, 13, 'd'), (2, 8, 14, 'd')],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')],
                    [4, 5, 10],
                    ["4-chordalcycle"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "4-chordalcycle" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 1, 2, 3, 4, 5, 6, 7, 8
    * Edges: (3, 4, 2, 'a'), (6, 8, 3, 'a'), (2, 8, 4, 'a'), (0, 2, 7, 'a'), (0, 7, 8, 'a'), (2, 7, 10, 'a'), (1, 5, 13, 'a')
3. Since the current graph definitely doesn't contain "4-chordalcycle", I will now search the graph for the "4-chordalcycle with its last edge removed" temporal motif.
    * The pattern for a "4-chordalcycle with its last edge removed" is a sequence of four edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 9.
    * In this undirected graph, the edges (8, 2, 4, 'a'), (2, 0, 7, 'a'), (0, 7, 8, 'a'), (7, 2, 10, 'a') form the above "4-chordalcycle with its last edge removed" pattern.
    * The timestamps 4, 7, 8, 10 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 10 - 4 = 6, 6 < 9, which satisfies the time window constraint.
4. Therefore, I will add the last edge (7, 8, 11, 'a') within the time window to make the graph contain a complete "4-chordalcycle" temporal motif.
5. This edge is the final answer.""",
                    [(7, 8, 11, 'a')] # answer (会被 make_qa_example 格式化)
                )
            ]
        elif self.motif_name == "bitriangle":
            qa_examples_raw = [
                (
                    [(3, 4, 2, "a"), (2, 8, 4, "a"), (2, 7, 6, "a"), (2, 7, 7, "d"), (7, 9, 7, "a"), (5, 6, 8, "a"), (6, 9, 8, "a"), (0, 6, 10, "a"), (0, 6, 10, "d"), (2, 8, 14, "d")],
                    [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')],
                    [6, 6, 10],
                    ["bitriangle"],
                    """**Chain of Thought:**
1. My goal is to modify the given undirected dynamic graph so that it contains "bitriangle" temporal motif.
2. I'll start by identifying the nodes and all 'a' operation edges in the graph.
    * Nodes: 0, 2, 3, 4, 5, 6, 7, 8, 9
    * Edges: (3, 4, 2, 'a'), (2, 8, 4, 'a'), (2, 7, 6, 'a'), (7, 9, 7, 'a'), (5, 6, 8, 'a'), (6, 9, 8, 'a'), (0, 6, 10, 'a')
3. Since the current graph definitely doesn't contain "bitriangle", I will now search the graph for the "bitriangle with its last edge removed" temporal motif.
    * The pattern for a "bitriangle with its last edge removed" is a sequence of five edges: ('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), where timestamps are strictly increasing, and the difference between the maximum and minimum timestamp is at most 9.
    * In this undirected graph, the edges (8, 2, 4, 'a'), (2, 7, 6, 'a'), (7, 9, 7, 'a'), (9, 6, 8, 'a'), (6, 0, 10, 'a') form the above pattern.
    * The timestamps 4, 6, 7, 8, 10 are strictly increasing.
    * The difference between the maximum and minimum timestamp is 10 - 4 = 6, 6 < 9, which satisfies the time window constraint.
4. Therefore, I will add the last edge (0, 8, 11, 'a') within the time window to make the graph contain a complete "bitriangle" temporal motif.
5. This edge is the final answer.""",
                    [(0, 8, 11, 'a')] # answer (会被 make_qa_example 格式化)
                )
            ]

        # 转换为 make_qa_example 需要的格式: list[dict, list, str]
        # 需要一个包含 motif_definition 和 context 的字典作为第一个元素
        qa_formatted = []
        for graph_ctx, motif_def, NMT, motif_name, cot, answer in qa_examples_raw:
             example_qa_dict = {
                 "context": graph_ctx,
                 "query": [motif_def , NMT, motif_name], # query 仍然为空
                 "cot": cot,
                 "answer" : answer
             }
             qa_formatted.append([example_qa_dict["context"], example_qa_dict["query"], example_qa_dict["cot"], example_qa_dict["answer"]]) # 第二个列表是 query (空), 第三个是 answer

        return self.make_qa_example(num, qa_formatted)

    def generate_prompt_question(self, query = None, *args, **kwargs):
        """
        生成包含 Motif 定义和图上下文的问题。(中文注释)
        kwargs 应该包含 'motif_definition' 和 'context'。
        """
        print(f"query: {query}")
        motif_def = query[0]
        N = query[1][0]
        M = query[1][1]
        T = query[1][2]
        motif_name = query[2][0]
        def format_list_for_prompt(data_list):
             return "[" + ", ".join([f"({str(u)}, {str(v)}, {str(t)}, {str(op)})" for u, v, t, op in data_list]) + "]"

        motif_def_str = format_list_for_prompt(motif_def)

        # 使用你提供的问题模板
        return f" Given a {motif_name} temporal motif which is a {N}-node, {M}-edge, {T}-temporal motif with the edges{motif_def_str}. How to modify the given undirected dynamic graph so that it contains the given temporal motif?"

    def evaluate(self, qa, response, w):
        """
        评估模型响应是否为 "Yes" 或 "No"，并与真实答案比较。(中文注释)

        Args:
            qa (dict): 包含真实答案 'answer' ("Yes"或"No") 的字典。
            response (str): 模型生成的完整响应字符串。

        Returns:
            tuple: (metric, extracted_answer)
                   metric: 1 (正确), 0 (错误), -3 (解析失败)
                   extracted_answer: 解析出的 "Yes" 或 "No" (小写)，或 None
        """
        metric = -3 # 默认为解析失败
        response_quads_list = parse_quadruples_from_response_content(response)
        context = qa.get("context")
        original_motif = qa.get("query")[0]
        processed_motif = []
        for item in original_motif:
            if len(item) == 4:
                u, v, t_rel, event_type = item
                # 检查是否需要转换节点标识符
                if isinstance(u, str) and u.startswith('u'):
                    try:
                        u = int(u[1:])  # 从 'u0' 提取 0
                    except ValueError:
                        pass  # 保持原样
                if isinstance(v, str) and v.startswith('u'):
                    try:
                        v = int(v[1:])  # 从 'u1' 提取 1
                    except ValueError:
                        pass  # 保持原样
                if isinstance(t_rel, str) and t_rel.startswith('t'):
                    try:
                        t_rel = int(t_rel[1:])  # 从 't0' 提取 0
                    except ValueError:
                        pass  # 保持原样
                processed_motif.append((u, v, t_rel, event_type))
            else:
                processed_motif.append(item)  # 保持原样
        print(judge(context, processed_motif, w))
        print(response_quads_list)
        if response_quads_list and judge(context + response_quads_list, processed_motif, w) == "Yes":
            print("Yes")
            metric = 1
        else:
            metric = 0
        return metric, response_quads_list
