# Return improvement prompt text based on input integer imp
def get_imp(imp):
    """
    Return predefined improvement prompt text based on integer imp.
    These texts aim to guide LLM's thinking process or output format.

    Args:
        imp (int): Integer indicating type of improvement needed.
            0: Add no improvement prompt.
            24: Prompt to consider time first then nodes.
            25: Prompt to consider nodes first then time.
            26: Prompt LLM to take a deep breath and solve step by step (Chain-of-Thought guidance).
            27: Prompt to think about nodes first then time.
            28: Prompt to think about time first then nodes.
            Other values: Raise NotImplementedError.

    Returns:
        str: Corresponding improvement prompt text.
    """
    if imp == 0:
        return f"" # No addition
    elif imp == 24:
        return f'Pick time and then nodes.' # Time first then nodes
    elif imp == 25:
        return f'Pick nodes and then time.' # Nodes first then time
    elif imp == 26:
        return f"Take a deep breath and work on this problem step-by-step." # CoT guidance
    elif imp == 27:
        return f'Think nodes and then time.' # Think nodes first then time
    elif imp == 28:
        return f'Think time and then nodes.' # Think time first then nodes
    else:
        raise NotImplementedError()
    
import types
import re

# Define class for building dynamic graph task prompts
class DyGraphPrompt:
    """
    Assemble complete prompt text for querying LLM based on configuration and task requirements.
    """
    def __init__(self, obj_task, args) -> None:
        """
        Initialize DyGraphPrompt.

        Args:
            obj_task: Object representing specific task (e.g., inheriting from BaseTask),
                      needs to implement generate_*_prompt related methods.
            args: Parameter object containing configuration options (usually from argparse).
                Must include add_cot, add_role, num_examplars, dyg_type, imp, short attributes.
        """
        # Get configuration from args
        add_cot = args.add_cot         # Whether to add Chain-of-Thought prompts
        add_role = args.add_role       # Whether to add role-playing prompts
        num_examplars = args.num_examplars # How many examples to use (Few-shot learning)
        dyg_type = args.dyg_type     # Dynamic graph representation type (affects description text)
        imp = args.imp               # Improvement prompt type (see get_imp function)
        short = args.short             # Whether to request short answers or add time format notes
        
        # --- Predefined prompt components --- 
        # Role description
        self.instructor_role = "You are an excellent dynamic network analyzer, with a good understanding of the structure of the graph and its evolution through time."
        # Dynamic graph data format description
        if dyg_type == 0:
            # Describe as tuple list (u, v, t)
            self.instructor_dyg = f"A dynamic graph is represented as a list of 4-tuples, where each tuple (u, v, t, a) denotes that there is a connection between node u and node v at time t, and (u, v, t, d) denotes that the edge between node u and node v is disconnected at time t. For example, (6, 5, 2, a) indicates that node 6 is connected to node 5 at time 2, while (6, 5, 4, d) indicates that node 6 is disconnected from node 5 at time 4."
        elif dyg_type == 1:
            # Describe as undirected graph tuple list
            self.instructor_dyg = f"In an undirected dynamic graph, (u, v, t, a) means that node u and node v are linked with an undirected edge at time t, (u, v, t, d) means that node u and node v are deleted with an undirected edge at time t."
        elif dyg_type == 2:
            # Describe as directed graph tuple list
            self.instructor_dyg = f"In an undirected dynamic graph, a quadruple (u, v, t, op) represents an edge, where u and v are the nodes connected by the edge, t is the timestamp, and op denotes the operation. The operation can be 'a' for adding the edge or 'd' for deleting it. Specifically, (u, v, t, a) means an undirected edge is added between nodes u and v at time t, while (u, v, t, d) means the edge is deleted at time t."
        else:
            raise NotImplementedError(f"dyg_type {dyg_type} not implemented")
        
        if args.motif == 0:
            self.motif_dyg = ""
        elif args.motif == 1:
            self.motif_dyg = f"A k-node, l-edge, δ-temporal motif is a time-ordered sequence of k nodes and l distinct edges within δ duration, i.e., M = (u1, v1, t1, a), (u2, v2, t2, a) ..., (ul, vl, tl, a), this edges (u1, v1), (u2, v2) ..., (ul, vl) form a specific pattern, t1 < t2 < ... < tl are strictly increasing, and tl - t1 ≤ δ. That is each edge that forms a specific pattern occurs in specific order within a fixed time window. Each consecutive events shares at least one common node. When searching for a specific temporal motif in the undirected dynamic graph, it is necessary to match pattern, edge order and time window. Node IDs and exact timestamps are irrelevant. Meanwhile,you should only focus on added edges. Note that some patterns are symmetric, so the order of the corresponding timestamps may be unimportant."
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
        self.obj_task = obj_task

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
        Generate dictionary containing complete prompt and optional standard answer.

        Args:
            context: Context information containing graph data (usually edge list etc.).
            query (optional): Specific question about the graph.
            answer (optional): Standard answer to the question.
            *args, **kwargs: Additional parameters passed to task object's generate_*_prompt methods.

        Returns:
            dict: Dictionary containing "prompt" and "answer".
        """
        # --- Generate parts of the prompt --- 
        # 1. Basic instructions (optional)
        instructor_role = self.instructor_role if self.add_role else ""
        instructor_dyg = self.instructor_dyg
        motif_dyg = self.motif_dyg
        multi_motif_dyg = self.multi_motif_dyg
        # 2. Task-related instructions (generated through task object)
        prompt_context = self.obj_task.generate_context_prompt(context, **kwargs) # Graph data context
        instructor_task = self.obj_task.generate_instructor_task(**kwargs)         # Task description
        instructor_answer = self.obj_task.generate_instructor_answer(**kwargs)    # Answer format description
        
        # 3. Examples (Few-shot, generated through task object)
        prompt_examplars = self.obj_task.generate_prompt_examplars(self.num_examplars, **kwargs) if self.num_examplars > 0 else ""
        print(self.num_examplars)
        # 4. Specific question (generated through task object)
        prompt_question = self.obj_task.generate_prompt_question(query, **kwargs)
        # 5. Improvement and CoT prompts
        prompt_imp = self.prompt_imp
        prompt_cot = self.prompt_cot

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


        prompt = prompt.format(
            instructor_role=instructor_role, 
            instructor_dyg=instructor_dyg, 
            motif_dyg=motif_dyg,
            multi_motif_dyg=multi_motif_dyg,  # Add missing parameter, set to empty string
            instructor_task=instructor_task, 
            prompt_imp=prompt_imp, 
            instructor_answer=instructor_answer, 
            prompt_examplars=prompt_examplars, 
            prompt_context=prompt_context, 
            prompt_question=prompt_question, 
            prompt_cot=prompt_cot
        )
        prompt = re.sub(r'\n\s*\n', '\n', prompt)  # Remove empty lines
        prompt = prompt.strip()  # Remove extra whitespace at start and end
        if not prompt.endswith('\n'):
            prompt += '\n'

        # --- Return result dictionary --- 
        prompt_qa = {
            "prompt": prompt,
            "answer": answer, # Include original answer for later evaluation
        }
        return prompt_qa
