import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 根据输入的整数 imp 返回对应的改进提示语 (Improvement Prompt)
def get_imp(imp):
    """
    根据整数 imp 返回预设的提示改进文本。
    这些文本旨在指导 LLM 的思考过程或输出格式。

    Args:
        imp (int): 指示所需改进类型的整数。
            0: 不添加任何改进提示。
            24: 提示先考虑时间再考虑节点。
            25: 提示先考虑节点再考虑时间。
            26: 提示 LLM 深呼吸并逐步解决问题 (Chain-of-Thought 引导)。
            27: 提示先思考节点再思考时间。
            28: 提示先思考时间再思考节点。
            其他值: 抛出 NotImplementedError。

    Returns:
        str: 对应的改进提示文本。
    """
    if imp == 0:
        return f"" # 不添加
    elif imp == 24:
        return f'Pick time and then nodes.' # 先选时间再选节点
    elif imp == 25:
        return f'Pick nodes and then time.' # 先选节点再选时间
    elif imp == 26:
        return f"Take a deep breath and work on this problem step-by-step." # CoT 引导
    elif imp == 27:
        return f'Think nodes and then time.' # 先思考节点再思考时间
    elif imp == 28:
        return f'Think time and then nodes.' # 先思考时间再思考节点
    else:
        raise NotImplementedError()
    
import types
import re

# 定义用于构建动态图任务提示的类
class DyGraphPrompt:
    """
    根据配置和任务需求，组装用于向 LLM 提问的完整提示文本。
    """
    def __init__(self, obj_task, args) -> None:
        """
        初始化 DyGraphPrompt。

        Args:
            obj_task: 一个代表具体任务的对象 (例如，继承自 BaseTask)，
                      需要实现 generate_*_prompt 相关方法。
            args: 包含配置选项的参数对象 (通常来自 argparse)。
                需要包含 add_cot, add_role, num_examplars, dyg_type, imp, short 等属性。
        """
        # 从 args 中获取配置
        add_cot = args.add_cot         # 是否添加 Chain-of-Thought 提示
        add_role = args.add_role       # 是否添加角色扮演提示
        num_examplars = args.num_examplars # 使用多少个示例 (Few-shot learning)
        dyg_type = args.dyg_type     # 动态图表示方式的类型 (影响描述文本)
        imp = args.imp               # 改进提示类型 (见 get_imp 函数)
        short = args.short             # 是否要求简短回答或添加时间格式说明
        
        # --- 预设的提示组件 --- 
        # 角色描述
        self.instructor_role = "You are an excellent dynamic network analyzer, with a good understanding of the structure of the graph and its evolution through time."
        # 动态图数据格式描述
        if dyg_type == 0:
            # 描述为元组列表 (u, v, t)
            self.instructor_dyg = f"A dynamic graph is represented as a list of 4-tuples, where each tuple (u, v, t, a) denotes that there is a connection between node u and node v at time t, and (u, v, t, d) denotes that the edge between node u and node v is disconnected at time t. For example, (6, 5, 2, a) indicates that node 6 is connected to node 5 at time 2, while (6, 5, 4, d) indicates that node 6 is disconnected from node 5 at time 4."
        elif dyg_type == 1:
            # 描述为无向图的元组列表
            self.instructor_dyg = f"In an undirected dynamic graph, (u, v, t, a) means that node u and node v are linked with an undirected edge at time t, (u, v, t, d) means that node u and node v are deleted with an undirected edge at time t."
        elif dyg_type == 2:
            # 描述为有向图的元组列表
            self.instructor_dyg = f"In an undirected dynamic graph, a quadruple (u, v, t, op) represents an edge, where u and v are the nodes connected by the edge, t is the timestamp, and op denotes the operation. The operation can be 'a' for adding the edge or 'd' for deleting it. Specifically, (u, v, t, a) means an undirected edge is added between nodes u and v at time t, while (u, v, t, d) means the edge is deleted at time t."
        else:
            raise NotImplementedError(f"dyg_type {dyg_type} not implemented")
        
        if args.motif == 0:
            self.motif_dyg = ""
        elif args.motif == 1:
            self.motif_dyg = f"A k-nosde, l-edge, δ-temporal motif is a time-ordered sequence of k nodes and l distinct edges within δ duration, i.e., M = (u1, v1, t1, a), (u2, v2, t2, a) ..., (ul, vl, tl, a), this edges (u1, v1), (u2, v2) ..., (ul, vl) form a specific pattern, t1 < t2 < ... < tl are strictly increasing, and tl - t1 ≤ δ. That is each edge that forms a specific pattern occurs in specific order within a fixed time window. Each consecutive events shares at least one common node. When searching for a specific temporal motif in the undirected dynamic graph, it is necessary to match pattern, edge order and time window. Node IDs and exact timestamps are irrelevant. Meanwhile,you should only focus on added edges. Note that some patterns are symmetric, so the order of the corresponding timestamps may be unimportant."
            # self.motif_dyg = f" A temporal motif is represented by a string of its numerical nodes, arranged in the order of their edge connections. For example, the edges (0, 1), (1, 2), (2, 0) form the string '0120'."
        if args.multi_motif == 0:
            self.multi_motif_dyg = ""
        elif args.multi_motif == 1:
            self.multi_motif_dyg = f"""Possible temporal motifs in the dynamic graph include:
"triangle": a 3-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u0, t2, a)]
"3-star": a 4-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u0, u2, t1, a), (u0, u3, t2, a)]
"4-path": a 4-node, 3-edge, 3-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a)]
"4-cycle": a 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u0, t3, a)]
"4-tailedtriangle": 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u1, t3, a)]
"butterfly": a 4-node, 4-edge, 6-temporal motif with the edges[(u0, u1, t0, a), (u1, u3, t1, a), (u3, u2, t2, a), (u2, u0, t3, a)]
"4-chordalcycle": a 4-node, 5-edge, 14-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u2, u3, t2, a), (u3, u1, t3, a), (u3, u0, t4, a)]
"4-clique": a 4-node, 6-edge, 15-temporal motif with the edges[(u0, u1, t0, a), (u1, u2, t1, a), (u1, u3, t2, a), (u2, u3, t3, a), (u3, u0, t4, a), (u0, u2, t5, a)]
"bitriangle": 6-node, 6-edge, 15-temporal motif with the edges[(u0, u1, t0, a), (u1, u3, t1, a), (u3, u5, t2, a), (u5, u4, t3, a), (u4, u2, t4, a), (u2, u0, t5, a)]"""
        
        self.args = args
        if args:
            imp = self.args.__dict__.get("imp", 0)
        else:
            imp = 0
        self.prompt_imp = get_imp(imp)
        self.add_cot = add_cot
        self.prompt_cot = "Let's think step by step." if self.add_cot else ""
        # self.prompt_cot = f"Let's think step by step.\n\n**Chain of Thought:**" if self.add_cot else ""
        # if self.args.k > 0 and self.args.add_cot > 0:
        #     # self.prompt_cot = f"**Now, following the example's format, provide the step-by-step reasoning and the final sorted list.**\n\n**Chain of Thought:**"
        #     self.prompt_cot = f"**Now, following the example's format, provide the step-by-step reasoning and the final sorted list.**"
        self.add_role = add_role
        self.num_examplars = num_examplars
        self.obj_task = obj_task # 存储任务对象，用于调用其生成提示的方法

        if short == 0:
            self.short = ""
        elif short==1:
            self.short = f"Give a short answer."
        elif short==2:
            self.short = f"Note that the time represents year, month, day, for example, 20200925 means 25th day in September in 2020, and 19990102 < 20200925 < 20231207"
        elif short==3:
            self.short = f"Note that the time represents unix timestamp, for example, 1348839350 < 1476979078 < 1547036558"


    def generate_prompt_qa(self, context = None, query = None, answer = None, *args, **kwargs):
        """
        生成包含完整提示 (prompt) 和可选的标准答案 (answer) 的字典。

        Args:
            context: 包含图数据的上下文信息 (通常是图的边列表等)。
            query (optional): 针对该图的具体问题。
            answer (optional): 该问题的标准答案。
            *args, **kwargs: 传递给任务对象的 generate_*_prompt 方法的额外参数。

        Returns:
            dict: 包含 "prompt" 和 "answer" 的字典。
        """
        # --- 生成提示的各个部分 --- 
        # 1. 基础指令 (可选)
        instructor_role = self.instructor_role if self.add_role else ""
        instructor_dyg = self.instructor_dyg
        motif_dyg = self.motif_dyg
        multi_motif_dyg = self.multi_motif_dyg
        # 2. 任务相关指令 (通过任务对象生成)
        prompt_context = self.obj_task.generate_context_prompt(context, **kwargs) # 图数据上下文
        instructor_task = self.obj_task.generate_instructor_task(**kwargs)         # 任务描述
        instructor_answer = self.obj_task.generate_instructor_answer(**kwargs)    # 答案格式说明
        
        # 3. 示例 (Few-shot, 通过任务对象生成)
        prompt_examplars = self.obj_task.generate_prompt_examplars(self.num_examplars, **kwargs) if self.num_examplars > 0 else ""
        print(self.num_examplars)
        # 4. 具体问题 (通过任务对象生成)
        prompt_question = self.obj_task.generate_prompt_question(query, **kwargs)
        # 5. 改进和 CoT 提示
        prompt_imp = self.prompt_imp
        prompt_cot = self.prompt_cot
        
        # # --- 按顺序组装提示 --- 
        # prompt_seq = [
        #     instructor_role,    # 角色
        #     instructor_dyg,     # 图表示说明
        #     instructor_task,    # 任务描述
        #     prompt_imp,         # 改进提示
        #     instructor_answer,  # 答案格式
        #     prompt_examplars,   # 示例
        #     prompt_context,     # 图数据
        #     prompt_question,    # 具体问题
        #     prompt_cot,          # CoT 提示
        #     f"\nAnswer:"
        # ]
        
        # 6. 添加额外的简短回答或时间格式提示 (根据 args.short)
        # short = self.short     
        # # prompt
        # Line = "\n"+"---\n"+"\n"
        # zero_shot_cot = """Your response should first provide a step-by-step "Chain of Thought" and then """ if self.args.add_cot > 0 else ""
        # if self.args.k > 0:
        #     prompt = instructor_role + instructor_dyg + motif_dyg + multi_motif_dyg + instructor_task + prompt_imp + instructor_answer + Line+ prompt_examplars + \
        #             Line + "## Here is what you need to answer:\n" + prompt_context +prompt_question + prompt_cot + short
        # else:
        #     prompt = instructor_role + instructor_dyg + motif_dyg + multi_motif_dyg + instructor_task + prompt_imp + zero_shot_cot + instructor_answer + Line+ prompt_examplars + \
        #             "## Here is what you need to answer:\n" + prompt_context +prompt_question + prompt_cot + short
        # print(prompt)
        # --- 合并所有部分 --- 
#         if self.args.model == "Llama_8B":
#             prompt = """[INST]
# {instructor_role}
# {instructor_dyg}
# {motif_dyg}
# {multi_motif_dyg}
# {instructor_task}
# {prompt_imp}
# {instructor_answer}
# {prompt_examplars}
# ## Here is what you need to answer:
# {prompt_context}{prompt_question}
# {prompt_cot}
# {short}
# [/INST]"""
#         else:
#             prompt = """{instructor_role}
# {instructor_dyg}
# {motif_dyg}
# {multi_motif_dyg}
# {instructor_task}
# {prompt_imp}
# {instructor_answer}
# {prompt_examplars}
# ## Here is what you need to answer:
# {prompt_context}{prompt_question}
# {prompt_cot}
# {short}
# Answer:"""

        prompt = """{instructor_role}
{instructor_dyg}
{motif_dyg}
{multi_motif_dyg}
{instructor_task}
{prompt_imp}
{instructor_answer}
{prompt_examplars}
Here is what you need to answer:
{prompt_context}{prompt_question}
{prompt_cot}
Answer:"""
#         prompt = """{instructor_role}
# {instructor_dyg}
# {motif_dyg}
# {instructor_task}
# {prompt_imp}
# {instructor_answer}
# {prompt_examplars}
# {prompt_context}{prompt_question}
# {prompt_cot}
# {short}
# """

        prompt = prompt.format(
            instructor_role=instructor_role, 
            instructor_dyg=instructor_dyg, 
            motif_dyg=motif_dyg,
            multi_motif_dyg=multi_motif_dyg,  # 添加缺失的参数，设置为空字符串
            instructor_task=instructor_task, 
            prompt_imp=prompt_imp, 
            instructor_answer=instructor_answer, 
            prompt_examplars=prompt_examplars, 
            prompt_context=prompt_context, 
            prompt_question=prompt_question, 
            prompt_cot=prompt_cot
        )
        prompt = re.sub(r'\n\s*\n', '\n', prompt)  # 去掉空行
        prompt = prompt.strip()  # 去掉开头和结尾的多余空白
        if not prompt.endswith('\n'):
            prompt += '\n'
        # prompt += "Do not output the thought process; provide the answer directly.\n"
        # prompt += "/no_think"
        # --- 返回结果字典 --- 
        prompt_qa = {
            "prompt": prompt,
            "answer": answer, # 将原始答案也包含在内，方便后续评估
        }
        return prompt_qa

