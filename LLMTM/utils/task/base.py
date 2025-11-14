import re
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
class DyGraphTask:
    def __init__(self, task, args) -> None:
        self.task = task
        self.args = args
        
    def generate_qa(self, info, *args, **kwargs):
        pass
    
    def generate_instructor_task(self, *args, **kwargs):
        pass
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a list of 4-tuples at the end of your response after 'Answer:'."
    
    def generate_prompt_examplars(self, num, *args, **kwargs):
        pass
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        pass
    def generate_instructor_cot(self, *args, **kwargs):
        pass
        
    def generate_context_prompt(self, context, flag = False, **kwargs):
        """
        Formats the graph context (edge list) for the prompt.
        Ensures quadruples are formatted without quotes around the operation.
        """
        edge_type = self.args.__dict__.get("edge_type", 0) if self.args else 0
        # --- Modification Start ---
        context_str_list = []
        if isinstance(context, list):
            for edge_tuple in context:
                if isinstance(edge_tuple, (list, tuple)) and len(edge_tuple) == 4:
                    u, v, t, op = edge_tuple
                    # Format without quotes around op
                    context_str_list.append(f"({int(u)}, {int(v)}, {int(t)}, {str(op)})")
                else:
                    print(f"Warning (BaseTask - Context): Skipping non-quadruple item: {edge_tuple}")
                    context_str_list.append(str(edge_tuple)) # Fallback
        else:
             print(f"Warning (BaseTask - Context): context is not a list: {type(context)}")
             return f"Question: Given an undirected dynamic graph with the edges {str(context)}." # Fallback
        if flag:
            context_str = "[" + ", ".join(context_str_list) + "]"
        else:
            context_str = "[" + ", ".join(context_str_list) + "]"
        # Unify return format
        return f"Question: Given an undirected dynamic graph with the edges {context_str}."
        # --- Modification End ---
    
    # def evaluate(self, qa, response):
    #     ans = qa['answer']
    #     match = re.search(r"Answer:\s*\[([\d,\s]+)\]", response)
    #     if match:
    #         numbers_str = match.group(1)
    #         numbers = [int(num) for num in numbers_str.split(',')]
    #         metric = (set(numbers) == set(ans))
    #         return metric
    #     else:
    #         return -1
    # def make_qa_example(self, num, qa):
    #     if num == 0:
    #         return ""
    #     examples = []
    #     for c,q,a in qa:
    #         example = f"{self.generate_context_prompt(c)}{self.generate_prompt_question(q)}Answer:{a}\n"
    #         examples.append(example)
        
    #     if num == 1:
    #         prompt = "Here is an example: " + "\n".join(examples[:num])
    #     else:
    #         prompt = f"Here are {num} examples: " + "\n".join(examples[:num])
    #     return prompt
    def make_qa_example(self, num, qa):
        """
        Formats QA pairs into example strings for the prompt.
        Ensures quadruples (context and answer) are formatted without quotes around the operation.
        """
        if num == 0 or not qa:
            return ""
        examples = []
        for c,q,s,a in qa[:num]: # Iterate through the provided QA examples
            # Format context using the (fixed) generate_context_prompt
            context_prompt = self.generate_context_prompt(c, True) # Pass context 'c'

            # Format question using the task's specific method
            question_prompt = self.generate_prompt_question(q)
            # a = a[0]
            # --- Modification Start: Format Answer ---
            answer_str = ""
            if isinstance(a, list):
                 # Check if it's a list of quadruples
                 if a and isinstance(a[0], (list, tuple)) and len(a[0]) == 4:
                     # Format quadruples without quotes around op
                     answer_str = "[" + ", ".join([f"({int(u)}, {int(v)}, {int(t)}, {str(op)})" for u, v, t, op in a]) + "]"
                 # Add other potential list formats if needed (e.g., list of ints)
                 elif a and isinstance(a[0], int):
                     answer_str = "[" + ", ".join(map(str, a)) + "]"
                 else: # Default list representation
                     answer_str = str(a)
            else: # Handle non-list answers (int, str, etc.)
                 answer_str = str(a)
            # --- Modification End ---
            if self.task == "judge_motif" or self.task == "judge_contain_motif":
                answer_str = a[0]
            # Combine parts into the example string
            if self.args.add_cot == 1:
                example = f"{context_prompt}{question_prompt}\n{s}\nAnswer: {answer_str}\n"
                # example = f"{context_prompt}{question_prompt} Answer: {answer_str}"
            else:
                example = f"{context_prompt}{question_prompt} Answer: {answer_str}\n"
            examples.append(example)

        # Format the final prompt section with examples
        if num == 1:
            prompt = "Here is an example: " + "\n".join(examples)
        else:
            prompt = f"Here are {num} examples: " + "\n".join(examples)
        return prompt