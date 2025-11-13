from .base import DyGraphTask
import numpy as np
import re
import random
from collections import defaultdict

class DyGraphTaskWhenDirectLink(DyGraphTask):
    """
    任务：询问给定两个节点首次直接连接的时间，以及之后首次断开直接连接的时间。
    对应原始脚本：When.py, QA.py (Operation), evaluate.py (Operation)
    """
    def generate_qa(self, info, *args, **kwargs):
        """
        生成 QA 对。移植自 QA.py (Operation)。
        选择一条至少有一次添加和一次删除操作的边，找出首次添加和首次删除的时间。
        """
        context_orig = info['edge_index']
        # 确保 u, v, t 是整数，op 是字符串
        context_typed = []
        for item in context_orig:
            try:
                u, v, t, op = item
                # 规范化边表示 (u <= v)
                u_int, v_int = int(u), int(v)
                if u_int > v_int:
                    u_int, v_int = v_int, u_int
                context_typed.append((u_int, v_int, int(t), str(op)))
            except ValueError as e:
                print(f"警告 (WhenDirectLink - QA): 跳过无法解析的行: {item} - {e}")
                continue
            except Exception as e:
                print(f"警告 (WhenDirectLink - QA): 处理行时出错: {item} - {e}")
                continue

        if not context_typed:
            print("错误 (WhenDirectLink - QA): 没有有效的上下文数据。")
            # 在这种情况下，无法生成有效 QA，可以返回 None 或抛出错误
            # 返回 None 让调用者（如 runner.py 中的 gen）知道跳过此实例
            return None

        # 按时间排序
        context_typed.sort(key=lambda x: x[2])

        # 找出所有至少有一次 'a' 和一次 'd' 操作的边
        edge_operations = defaultdict(list)
        for u, v, t, op in context_typed:
            edge = (u, v) # 已经是规范化的
            edge_operations[edge].append((t, op))

        eligible_edges = []
        for edge, operations in edge_operations.items():
            has_add = any(op == 'a' for _, op in operations)
            has_delete = any(op == 'd' for _, op in operations)
            if has_add and has_delete:
                eligible_edges.append(edge)

        if not eligible_edges:
            print(f"警告 (WhenDirectLink - QA): 没有找到既有添加又有删除操作的边。")
            # 同样，无法生成有效 QA
            return None

        # 随机选择一条符合条件的边
        selected_edge = random.choice(eligible_edges)
        u_query, v_query = selected_edge # 提取查询的节点对

        # 找出该边的首次增加和首次删除操作的时间
        operations = edge_operations[selected_edge]
        first_add_time = -1
        first_delete_time = -1

        # 查找首次 'a'
        for t, op in operations:
            if op == 'a':
                first_add_time = t
                break # 找到第一个就停止

        # 查找首次 'd' (必须在首次 'a' 之后发生)
        if first_add_time != -1:
             for t, op in operations:
                  if op == 'd' and t >= first_add_time: # 确保删除不早于添加
                       first_delete_time = t
                       break # 找到第一个就停止

        # 必须同时找到首次添加和首次删除时间
        if first_add_time == -1 or first_delete_time == -1:
             print(f"警告 (WhenDirectLink - QA): 边 {selected_edge} 未找到有效的首次添加/删除对。")
             return None # 无法生成有效 QA

        # 答案是 (首次添加时间, 首次删除时间)
        answer = (first_add_time, first_delete_time)

        # context 用于 prompt 的是原始未排序、未类型转换的列表
        qa = {
            "context": context_orig, # 使用原始格式传递给 prompt
            "query": (u_query, v_query),
            "answer": answer,
            "task": self.task
        }
        return qa

    def generate_instructor_task(self, *args, **kwargs):
        # 移植自 When.py
        return (f"Your task is to answer when the given two nodes are first connected, and after that, when they are first disconnected in dynamic graph? Two nodes are connected if and only if there exists a direct edge between them, ignoring indirect connections. (The problem guarantees that the two nodes are connected at least once and disconnected at least once).")
        # return (f"Your task is to answer when the given two nodes are first connected, and after that, when they are first disconnected in dynamic graph? Two nodes are connected when the direct edge between them added. The two edges are disconnected when the direct edge deleted. (The problem guarantees that the two nodes are connected directly at least once and deleted at least once).")

    def generate_instructor_answer(self, *args, **kwargs):
        # 移植自 When.py
        return "Give the answer as two integer numbers at the last of your response after 'Answer:'."
    def generate_prompt_examplars(self, num, *args, **kwargs):
        # 移植自 When.py，确保格式正确
        qa_examples_raw = [
            (
                [(0, 5, 0, 'a'), (2, 7, 0, 'a'), (6, 8, 0, 'a'), (0, 5, 1, 'd'), (5, 7, 2, 'a'), (7, 8, 2, 'a'), (1, 2, 3, 'a'), (0, 2, 4, 'a'), (0, 3, 4, 'a'), (0, 6, 4, 'a'), (1, 2, 4, 'd'), (2, 3, 4, 'a'), (4, 5, 4, 'a'), (6, 7, 4, 'a')], # context
                (0, 5), # query
                """**Chain of Thought:**
1. Objective: Locate two specific timestamps for the edge between node 0 and node 5. First, the earliest time this edge is added (operation 'a'). Second, the earliest time this edge is deleted (operation 'd').
2. Scan for Connection: Go through all edges in the provided list. Identify the tuple (u, v, t, 'a') where u and v are 0 and 5 (order doesn't matter for undirected graphs). Record the t value from the first such occurrence. This is first_connected_time.
3. Scan for Disconnection: Similarly, go through all edges in the provided list. Identify the tuple (u, v, t, 'd') where u and v are 0 and 5. Record the t value from the first such occurrence. This is first_disconnected_time.
4. Final Output: Present first_connected_time and first_disconnected_time.""", # COT
                (0, 1) # answer - (first_add_time, first_delete_time)
            )
        ]
        # 转换格式
        qa_formatted = []
        for c, q, s, a in qa_examples_raw:  # 修复：现在正确解包4个元素
             # 答案需要特殊处理，因为 make_qa_example 可能不直接支持元组答案的格式化
             # 我们将其格式化为 "t1, t2" 字符串
             answer_str = f"{a[0]}, {a[1]}"
             qa_formatted.append([list(c), list(q), s, answer_str])  # 包含CoT元素s


        # 修改基类的 make_qa_example 以支持这种自定义答案字符串
        # 或者在这里直接构建示例字符串
        if num == 0:
            return ""
        examples = []
        for c, q, s, a_str in qa_formatted[:num]:  # 修复：正确解包4个元素
             context_prompt = self.generate_context_prompt(c) # 使用基类方法格式化 context
             question_prompt = self.generate_prompt_question(q) # 使用本类方法格式化 question
             if self.args.add_cot == 1:
                 example = f"{context_prompt}{question_prompt}\n{s}\nAnswer: {a_str}"
             else:
                 example = f"{context_prompt}{question_prompt} Answer: {a_str}" # 直接拼接答案字符串
             examples.append(example)

        if num == 1:
            prompt = "Here is an example: " + "\n".join(examples)
        else:
            prompt = f"Here are {num} examples: " + "\n".join(examples)
        return prompt


    def generate_prompt_question(self, query = None, *args, **kwargs):
        # 移植自 When.pys
        u, v = query
        return f" When are node {u} and node {v} first connected, and after that, when they are first disconnected in dynamic graph?"

    def evaluate(self, qa, response, use_agent):
        """
        评估模型响应。移植自 evaluate.py (Operation)。
        解析响应末尾的两个整数，并与答案元组比较。
        返回值:
            metric: 1 (正确), 0 (错误但格式正确), -1 (无法解析格式)
            extracted_answer: 解析出的 (t1, t2) 元组或 None
        """
        true_times = qa['answer'] # (t_add, t_del)

        # 查找最后一个 "Answer:" 之后的内容
        answer_marker = "Answer:"
        answer_start_index = response.rfind(answer_marker)
        if answer_start_index != -1 or use_agent == 1:
            if use_agent == 1:
                extracted_part = response[0:].strip()
            else:
                extracted_part = response[answer_start_index + len(answer_marker):].strip()
            extracted_part = extracted_part.lower()
            
            # 尝试匹配格式: "The nodes are first connected at time \\boxed{2} and then disconnected at time \\boxed{3}."
            match_nodes_boxed = re.search(r'nodes are first connected at time \\+boxed\{(\d+)\}.*?disconnected at time \\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_nodes_boxed:
                try:
                    appear_time = int(match_nodes_boxed.group(1))
                    disconnect_time = int(match_nodes_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First connected at time \\boxed{3}, first disconnected at time \\boxed{4}."
            match_first_boxed = re.search(r'first connected at time \\+boxed\{(\d+)\}.*?first disconnected at time \\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_first_boxed:
                try:
                    appear_time = int(match_first_boxed.group(1))
                    disconnect_time = int(match_first_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "### Final Answer\n- **First Connected:** \\boxed{2}\n- **First Disconnected:** \\boxed{4}"
            match_final_answer_boxed = re.search(r'###\s*final answer.*?\*\*first connected\*\*.*?\\+boxed\{(\d+)\}.*?\*\*first disconnected\*\*.*?\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_final_answer_boxed:
                try:
                    appear_time = int(match_final_answer_boxed.group(1))
                    disconnect_time = int(match_final_answer_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "**First Connected:** \\boxed{2}\n- **First Disconnected:** \\boxed{4}"
            # 专门匹配这种具体格式，处理各种空白字符
            match_connected_disconnected_boxed = re.search(r'\*\*first connected\*\*:.*?\\+boxed\{(\d+)\}.*?-\s*\*\*first disconnected\*\*:.*?\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_connected_disconnected_boxed:
                try:
                    appear_time = int(match_connected_disconnected_boxed.group(1))
                    disconnect_time = int(match_connected_disconnected_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 更精确的匹配，处理冒号后的空白字符和可能的换行
            match_markdown_boxed_precise = re.search(r'\*\*first connected\*\*:\s*\\+boxed\{(\d+)\}.*?\*\*first disconnected\*\*:\s*\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_markdown_boxed_precise:
                try:
                    appear_time = int(match_markdown_boxed_precise.group(1))
                    disconnect_time = int(match_markdown_boxed_precise.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "**First Connected:** \\boxed{2}\n- **First Disconnected:** \\boxed{4}" (更通用的匹配)
            match_markdown_boxed = re.search(r'\*\*first connected\*\*.*?\\+boxed\{(\d+)\}.*?\*\*first disconnected\*\*.*?\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_markdown_boxed:
                try:
                    appear_time = int(match_markdown_boxed.group(1))
                    disconnect_time = int(match_markdown_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First connected at t=1, first disconnected at t=3."
            match_simple_t_format = re.search(r'first connected at t=(\d+).*?first disconnected at t=(\d+)', extracted_part, re.DOTALL)
            if match_simple_t_format:
                try:
                    appear_time = int(match_simple_t_format.group(1))
                    disconnect_time = int(match_simple_t_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First connected at **t=1**, first disconnected at **t=2**"
            match_t_format = re.search(r'first connected at \*\*t=(\d+)\*\*.*?first disconnected at \*\*t=(\d+)\*\*', extracted_part, re.DOTALL)
            if match_t_format:
                try:
                    appear_time = int(match_t_format.group(1))
                    disconnect_time = int(match_t_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First connected at time: 0  \nFirst disconnected at time: 2"
            match_time_colon_newline = re.search(r'first connected at time:\s*(\d+).*?first disconnected at time:\s*(\d+)', extracted_part, re.DOTALL)
            if match_time_colon_newline:
                try:
                    appear_time = int(match_time_colon_newline.group(1))
                    disconnect_time = int(match_time_colon_newline.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First Connection: 2  \nFirst Disconnection: 3"
            match_connection_disconnection = re.search(r'first connection:\s*(\d+).*?first disconnection:\s*(\d+)', extracted_part, re.DOTALL)
            if match_connection_disconnection:
                try:
                    appear_time = int(match_connection_disconnection.group(1))
                    disconnect_time = int(match_connection_disconnection.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First Connection Time: 3  \nFirst Disconnection Time: 4"  
            match_connection_time = re.search(r'first connection time:\s*(\d+).*?first disconnection time:\s*(\d+)', extracted_part, re.DOTALL)
            if match_connection_time:
                try:
                    appear_time = int(match_connection_time.group(1))
                    disconnect_time = int(match_connection_time.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First Connection: 3\n- First Disconnect: 4"
            match_connection_format = re.search(r'first connection:\s*(\d+).*?first disconnect:\s*(\d+)', extracted_part, re.DOTALL)
            if match_connection_format:
                try:
                    appear_time = int(match_connection_format.group(1))
                    disconnect_time = int(match_connection_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "The nodes first connect at time `3` and then disconnect at time `4`."
            match_nodes_connect_backtick = re.search(r'nodes first connect at time `(\d+)`.*?disconnect at time `(\d+)`', extracted_part, re.DOTALL)
            if match_nodes_connect_backtick:
                try:
                    appear_time = int(match_nodes_connect_backtick.group(1))
                    disconnect_time = int(match_nodes_connect_backtick.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "The nodes first connect at time 0 and disconnect at time 4."
            match_nodes_connect_simple = re.search(r'nodes first connect at time (\d+).*?disconnect at time (\d+)', extracted_part, re.DOTALL)
            if match_nodes_connect_simple:
                try:
                    appear_time = int(match_nodes_connect_simple.group(1))
                    disconnect_time = int(match_nodes_connect_simple.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "The nodes are first connected at timestamp 0 and then disconnected at timestamp 4."
            match_nodes_timestamp_then = re.search(r'nodes are first connected at timestamp (\d+).*?then disconnected at timestamp (\d+)', extracted_part, re.DOTALL)
            if match_nodes_timestamp_then:
                try:
                    appear_time = int(match_nodes_timestamp_then.group(1))
                    disconnect_time = int(match_nodes_timestamp_then.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "The nodes are first connected at timestamp 2 and first disconnected at timestamp 4."
            match_nodes_timestamp = re.search(r'nodes are first connected at timestamp (\d+).*?first disconnected at timestamp (\d+)', extracted_part, re.DOTALL)
            if match_nodes_timestamp:
                try:
                    appear_time = int(match_nodes_timestamp.group(1))
                    disconnect_time = int(match_nodes_timestamp.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "The nodes are first connected at time 2 and then disconnected at time 3."
            match_nodes_are_connected = re.search(r'nodes are first connected at time (\d+).*?disconnected at time (\d+)', extracted_part, re.DOTALL)
            if match_nodes_are_connected:
                try:
                    appear_time = int(match_nodes_are_connected.group(1))
                    disconnect_time = int(match_nodes_are_connected.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "- First connected at time 0.\n- First disconnected at time 1."
            match_list_time = re.search(r'-\s*first connected at time (\d+).*?-\s*first disconnected at time (\d+)', extracted_part, re.DOTALL)
            if match_list_time:
                try:
                    appear_time = int(match_list_time.group(1))
                    disconnect_time = int(match_list_time.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "- First connected at timestamp 3.\n- First disconnected at timestamp 4."
            match_list_timestamp = re.search(r'-\s*first connected at timestamp (\d+).*?-\s*first disconnected at timestamp (\d+)', extracted_part, re.DOTALL)
            if match_list_timestamp:
                try:
                    appear_time = int(match_list_timestamp.group(1))
                    disconnect_time = int(match_list_timestamp.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First Connected: 4  \nFirst Disconnected: 4"
            match_first_connected_disconnected = re.search(r'first connected:\s*(\d+).*?first disconnected:\s*(\d+)', extracted_part, re.DOTALL)
            if match_first_connected_disconnected:
                try:
                    appear_time = int(match_first_connected_disconnected.group(1))
                    disconnect_time = int(match_first_connected_disconnected.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First connected at time `t = 0`, first disconnected at time `t = 1`."
            match_t_equals_format = re.search(r'first connected at time `t = (\d+)`.*?first disconnected at time `t = (\d+)`', extracted_part, re.DOTALL)
            if match_t_equals_format:
                try:
                    appear_time = int(match_t_equals_format.group(1))
                    disconnect_time = int(match_t_equals_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "Connect Time: 1, Disconnect Time: 3"
            match_connect_disconnect_time = re.search(r'connect time:\s*(\d+).*?disconnect time:\s*(\d+)', extracted_part, re.DOTALL)
            if match_connect_disconnect_time:
                try:
                    appear_time = int(match_connect_disconnect_time.group(1))
                    disconnect_time = int(match_connect_disconnect_time.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "First connected at 3, first disconnected at 4"
            match_simple_connected_at = re.search(r'first connected at (\d+),\s*first disconnected at (\d+)', extracted_part, re.DOTALL)
            if match_simple_connected_at:
                try:
                    appear_time = int(match_simple_connected_at.group(1))
                    disconnect_time = int(match_simple_connected_at.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: 任何包含两个 boxed 数字的文本，不管中间有什么内容
            match_any_two_boxed = re.search(r'\\+boxed\{(\d+)\}.*?\\+boxed\{(\d+)\}', extracted_part, re.DOTALL)
            if match_any_two_boxed:
                try:
                    appear_time = int(match_any_two_boxed.group(1))
                    disconnect_time = int(match_any_two_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    return -1, None
            
            # 尝试匹配格式: "\\boxed{3}, \\boxed{4}" 或 "\boxed{3}, \boxed{4}"
            match_boxed = re.search(r'\\+boxed\{(\d+)\},\s*\\+boxed\{(\d+)\}', extracted_part)
            if match_boxed:
                try:
                    appear_time = int(match_boxed.group(1))
                    disconnect_time = int(match_boxed.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 如果解析为整数失败
                    return -1, None # 格式错误
            
            # 尝试匹配格式: "**Answer:**\n- First connected at time **1**.\n- First disconnected at time **3**."
            match_markdown_format = re.search(r'first connected at time \*\*(\d+)\*\*.*?first disconnected at time \*\*(\d+)\*\*', extracted_part, re.IGNORECASE | re.DOTALL)
            if match_markdown_format:
                try:
                    appear_time = int(match_markdown_format.group(1))
                    disconnect_time = int(match_markdown_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 如果解析为整数失败
                    return -1, None # 格式错误
            
            # 尝试匹配格式: "First connected at time 0, first disconnected at time 4."
            match_simple_format = re.search(r'first connected at time (\d+),\s*first disconnected at time (\d+)', extracted_part, re.IGNORECASE)
            if match_simple_format:
                try:
                    appear_time = int(match_simple_format.group(1))
                    disconnect_time = int(match_simple_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 如果解析为整数失败
                    return -1, None # 格式错误
            
            # 尝试匹配格式: "First connected at time: 1, first disconnected at time: 5"
            match_colon_format = re.search(r'first connected at time:\s*(\d+),\s*first disconnected at time:\s*(\d+)', extracted_part, re.IGNORECASE)
            if match_colon_format:
                try:
                    appear_time = int(match_colon_format.group(1))
                    disconnect_time = int(match_colon_format.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 如果解析为整数失败
                    return -1, None # 格式错误
            
            # 尝试匹配格式: "First connected at time `3`, first disconnected at time `4`"
            match_formatted = re.search(r'first connected at time `(\d+)`, first disconnected at time `(\d+)`', extracted_part, re.IGNORECASE)
            if match_formatted:
                try:
                    appear_time = int(match_formatted.group(1))
                    disconnect_time = int(match_formatted.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 如果解析为整数失败
                    return -1, None # 格式错误
            
            # 尝试匹配格式: "Answer: 3, 3" (在提取部分开头直接跟数字)
            # 检查提取部分是否以数字开头
            match_answer_direct = re.search(r'^\s*(\d+),\s*(\d+)', extracted_part)
            if match_answer_direct:
                try:
                    appear_time = int(match_answer_direct.group(1))
                    disconnect_time = int(match_answer_direct.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 如果解析为整数失败
                    return -1, None # 格式错误
            
            # 尝试匹配多种答案格式
            patterns = [
                r'\*\*answer:\*\*\s*(\d+),\s*(\d+)(?:[.\s]*$|[.\s]*\n.*?(?:therefore|thus|hence|so),?\s+(?:the\s+)?answer\s+is\s*\1,\s*\2)',  # **Answer:** 2, 3\nTherefore, the answer is 2, 3
                r'\*\*answer:\*\*\s*(\d+),\s*(\d+)[.\s]*$',  # **Answer:** 2, 3
                r'answer:\s*(\d+),\s*(\d+)',  # Answer: 2, 3
            ]
            
            for pattern in patterns:
                match = re.search(pattern, extracted_part, re.IGNORECASE)
                if match:
                    try:
                        appear_time = int(match.group(1))
                        disconnect_time = int(match.group(2))
                        predicted_times = [appear_time, disconnect_time]
                        print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                        metric = 1 if predicted_times == true_times else 0
                        return metric, predicted_times
                    except ValueError:
                        continue
            if use_agent == 1:
                # 处理字符串形式的列表格式 "[3, 4]"
                try:
                    # 移除方括号和空格，然后按逗号分割
                    clean_part = extracted_part.strip().strip('[]')
                    numbers = [int(x.strip()) for x in clean_part.split(',')]
                    if len(numbers) == 2:
                        appear_time = numbers[0]
                        disconnect_time = numbers[1]
                        predicted_times = [appear_time, disconnect_time]
                        print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                        metric = 1 if predicted_times == true_times else 0
                        return metric, predicted_times
                    else:
                        return -1, None
                except (ValueError, IndexError):
                    return -1, None
            # 如果上述模式都没有匹配，尝试最基本的数字匹配
            match = re.search(r'(\d+)[,\s]+(\d+)\s*$', extracted_part)
            if match:
                try:
                    appear_time = int(match.group(1))
                    disconnect_time = int(match.group(2))
                    predicted_times = [appear_time, disconnect_time]
                    print(f"predicted_times: {predicted_times}, true_times: {true_times}")
                    metric = 1 if predicted_times == true_times else 0
                    return metric, predicted_times
                except ValueError:
                    # 如果解析为整数失败
                    return -1, None # 格式错误
            else:
                # 如果找不到符合格式的两个数字
                return -1, None # 格式错误
        else:
            # 如果找不到 "Answer:" 标记
             return -1, None # 格式错误 