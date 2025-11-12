import os
from .utils import  remove_dir, load_task
import json
from .utils import DyGraphGenContainMotif, DyGraphPrompt, DyGraphGenERCon, DyGraphGenIsMotif, DyGraphGenModifyMotif,load_task # Ensure load_task etc. are imported
from tqdm import tqdm
import numpy as np
import joblib
import time
import torch # Import torch
from transformers import AutoModelForCausalLM, AutoTokenizer # Import transformers classes
from collections import defaultdict, Counter # Import Counter for frequency statistics
import networkx as nx
from collections import Counter, defaultdict

try:
    from .utils import visualization
except ImportError:
    print("Warning: LLMTM.utils.visualization not found or failed to import. Visualization features will be unavailable.")
    visualization = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plotting_available = True
except ImportError:
    print("Warning: matplotlib or seaborn not installed. Data analysis plotting features will be unavailable.")
    plotting_available = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


import sys
import torch
import transformers
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)

PREDEFINED_MOTIFS = {
    "2-star":       [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a')],                             # k=3, l=2
    "triangle":     [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')],             # k=3, l=3
    "3-star":       [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')],             # k=4, l=3 (center node is 0)
    "4-path":       [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')],             # k=4, l=3
    "4-cycle":      [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')], # k=4, l=4
    "4-tailedtriangle": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')], # k=4, l=4
    "butterfly":    [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')], # k=4, l=4 (same topology as 4-cycle, different timing? Ambiguous based on image, temporarily excluded)
    "4-chordalcycle": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')], # k=4, l=5
    "4-clique":     [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')], # k=4, l=6
    "bitriangle":   [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')], # k=6, l=6
}



MODEL_PATHS = {
    # Use project-relative default locations under "models/". Users can override via CLI/env as needed.
    "Qwen_14B": "models/DeepSeek-R1-Distill-Qwen-14B",
    "Qwen_7B": "models/DeepSeek-R1-Distill-Qwen-7B",
    "gpt-4o-mini": "",
    "deepseek-r1-250528": "",
    "QwQ": "models/QwQ-32B",
    "Qwen2.5_32B": "models/Qwen2.5-32B-Instruct",
    "Qwen_32B": "models/DeepSeek-R1-Distill-Qwen-32B",
    "o3-2025-04-16": "",
}

def initialize_model(model):
    """Initialize the model and tokenizer in the main process"""
    model_path = MODEL_PATHS[model]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", # Changed to auto for simpler multi-GPU handling
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model_obj.eval()
    return model_obj, tokenizer

def chat_with_model(model, tokenizer, prompt, temperature, max_tokens):
    """
    Interact with the loaded model.
    """


    torch.cuda.empty_cache() # Clear cache

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    prompt_tokens = inputs.input_ids.shape[1]

    # --- Add debug code: write prompt to a file ---
    try:
        # Note: The filename can be customized to avoid conflicts
        debug_prompt_file = "prompt_debug_runner.txt"
        with open(debug_prompt_file, "w", encoding="utf-8") as f_debug:
            f_debug.write(prompt)
        print(f"[Debug Info] Prompt saved to: {debug_prompt_file}")
    except Exception as e_debug:
        print(f"[Debug Info] Error saving prompt to file: {e_debug}")
    # --- End of debug code ---
    with torch.no_grad():
        # Ensure inputs are on the correct device
        device_inputs = inputs.to(model.device)
        outputs = model.generate(
            **device_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=False, # Only sample if temperature > 0
            pad_token_id=tokenizer.eos_token_id, # <--- Add explicit setting
            repetition_penalty=1.0           # <--- Add explicit setting
        )
    completion_token_ids = outputs[0][prompt_tokens:]
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response_text)
    completion_tokens = len(completion_token_ids)
    total_tokens = prompt_tokens + completion_tokens

    # Clean up GPU memory
    del inputs, outputs, completion_token_ids, device_inputs # Clean up device_inputs
    torch.cuda.empty_cache()

    result = {
        "content": response_text.strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    return result


# Helper function: Generate a unique prompt configuration ID based on args
def _generate_prompt_suffix(args):
    """Generates a unique suffix based on prompt configuration arguments."""
    # Use args.__dict__.get to provide default values, avoiding errors if some attributes are missing in args
    k = args.__dict__.get('k', args.__dict__.get('num_examplars', 0)) # Compatible with k and num_examplars
    return (f"_cot{args.__dict__.get('add_cot', 0)}"
            f"_role{args.__dict__.get('add_role', 0)}"
            f"_k{k}"
            f"_dyg{args.__dict__.get('dyg_type', 0)}"
            f"_edge{args.__dict__.get('edge_type', 0)}"
            f"_imp{args.__dict__.get('imp', 0)}"
            f"_short{args.__dict__.get('short', 0)}"
            f"_temperature{args.__dict__.get('temperature', 0)}"
            f"_maxtokens{args.__dict__.get('max_tokens', 0)}"
            f"_motifname{args.__dict__.get('motif_name', 0)}"
            f"_motif{args.__dict__.get('motif', 0)}"
            f"_change{args.__dict__.get('change', 0)}"
            f"_use_agent{args.__dict__.get('use_agent', 0)}"
            f"_api{args.__dict__.get('api', 0)}"
            f"_balance{args.__dict__.get('balance', 0)}")

class Runner:
    """
    Manages the entire workflow of evaluating LLMs on dynamic graph tasks, including:
    1. Generating data and questions (gen) - only generates graph.json and qa.json
    2. Calling LLM to get answers (run) - dynamically generates prompts, saves answers with suffixes
    3. Evaluating model performance (evaluate) - loads answers with suffixes, saves results with suffixes
    4. Checking run status (check) - checks answer files with suffixes
    5. Summarizing and displaying results (show)
    """
    def __init__(self, args, try_all = False) -> None:
        """
        Initializes the Runner.

        Args:
            args (object): An object containing all configuration parameters (usually from argparse).
            try_all (bool, optional): Whether to continuously try running in 'run' mode until all tasks are completed. 
                                      Defaults to False.
        """
        self.args = args
        self.try_all = try_all
        
    def check(self, task_folder):
        """
        Checks the run status of a specific configuration in the given task folder.

        Args:
            task_folder (str): The root directory where task data and results are stored.

        Returns:
            int: The number of instances to be run (torun).
        """
        args = self.args
        model = args.model
        prompt_suffix = _generate_prompt_suffix(args)

        # --- Modified: Load instance_list.json ---
        instance_list_path = os.path.join(task_folder, "instance_list.json")
        try:
            with open(instance_list_path, "r") as f:
                 instance_info = json.load(f)
            instance_folders = instance_info["instance_list"] # Use the new key
            print(instance_folders)
        except FileNotFoundError:
             print(f"Error: instance_list.json not found in {task_folder}. Please run '-t gen' first.")
             return -1
        except json.JSONDecodeError:
             print(f"Error: instance_list.json in {task_folder} is corrupted.")
             return -1
        # --- End of modification ---

        finish = []
        torun = []
        sdict = {"num_edges": [], "num_nodes": [], "num_time": []}

        # --- Modified: Use instance_folders ---
        for i, folder_name in enumerate(instance_folders):
            folder_path = os.path.join(task_folder, folder_name)
            # Load the graph information for this instance
            try:
                graph = json.load(open(os.path.join(folder_path, "graph.json")))
                # Collect statistics
                for k, v in sdict.items():
                    v.append(graph.get(k, 0)) # Use get to prevent errors if some keys are missing
            except FileNotFoundError:
                print(f"Warning: graph.json not found in {folder_path}")
                torun.append(i) # If the graph file is missing, consider it as to be run
                continue
            except json.JSONDecodeError:
                 print(f"Warning: graph.json in {folder_path} is corrupted.")
                 torun.append(i) # If the graph file is corrupted, consider it as to be run
                 continue

            # --- Modified: The answer file is now in the model subdirectory ---
            model_subfolder_path = os.path.join(folder_path, model)
            answer_path = os.path.join(model_subfolder_path, f"answer_{model}{prompt_suffix}.json")
            # --- End of modification ---

            if os.path.exists(answer_path):
                try:
                    # Optional: Check if the answer file is valid
                    json.load(open(answer_path, "r"))
                    finish.append(i)
                except json.JSONDecodeError:
                    print(f"Warning: Answer file {answer_path} is corrupted.")
                    torun.append(i) # If the answer file is corrupted, consider it as to be run
            else:
                torun.append(i)
                
        print(f"--- Checking Status ---")
        print(f"Task Folder: {task_folder}")
        print(f"Model: {model}")
        print(f"Prompt Suffix: {prompt_suffix}")
        print(f"Finish {len(finish)}, ToRun {len(torun)} (Total {len(instance_folders)})")
        if sdict["num_edges"]: # Only print if data has been collected
             print("Graph Stats (Avg±Std): " + "".join(f"{k}:{np.mean(v):.2f}±{np.std(v):.2f} " for k,v in sdict.items() if v))
        else:
             print("Graph Stats: No valid graph data found to compute statistics.")
        return len(torun)
        
    
    def generate_random(self, dir, T, N, p, seed, *targs ,**kwargs):
        """
        Generates dynamic graph data (`graph.json`) and question-answer pairs (`qa.json`) for a single instance,
        and saves them to a specified subdirectory. **Does not save prompt_qa.json anymore**.

        Args:
            dir (str): The parent directory to save the instance data.
            T (int): The number of time steps for the dynamic graph.
            N (int): The number of nodes in the dynamic graph.
            p (float): The probability parameter for the dynamic graph generation model.
            seed (int): The random seed for generating this instance.
            *targs: Additional positional arguments to pass to obj_task.generate_qa.
            **kwargs: Additional keyword arguments to pass to obj_task.generate_qa (e.g., label).

        Returns:
            str: The folder name of the generated instance (e.g., "T_N_p_seed").
                 Returns None if generation fails.
        """
        folder_setting = f"T{T}_N{N}_p{p}_seed{seed}" # Restore folder naming format
        args = self.args
        task = args.task
        
        info = None
        if args.task == "judge_contain_motif" and (args.api == 1 or args.use_agent == 1):
            if p < 1:
                dygen = DyGraphGenERCon()
                info = dygen.sample_dynamic_graph(T = T, N = N , p = p, seed = seed)
            else:
                p = int(p)
                folder_setting = f"T{T}_N{N}_M{p}_seed{seed}"
                dygen = DyGraphGenContainMotif()
                info = dygen.generate_graph_with_motif_control(M = p, seed = seed, motif_name = args.motif_name, T = T)
        elif args.motif == 1 and task == "judge_is_motif":
            folder_setting = f"T{T}_N{N}_p{p}_seed{seed}"
            p = int(p)
            M = PREDEFINED_MOTIFS[args.motif_name]
            dygen = DyGraphGenIsMotif()
            info = dygen.sample_dynamic_graph(T + T, predefined_motif = M, motif_time_window = T, seed=seed)
        elif args.motif == 1 and task == "modify_motif":
            folder_setting = f"T{T}_N{N}_p{p}_seed{seed}"
            target_motif = PREDEFINED_MOTIFS[args.motif_name]
            p = int(p)
            dygen = DyGraphGenModifyMotif()
            info = dygen.sample_dynamic_graph(T_total_time = T, N_total_nodes = N , M_target_edges = p, W_motif_window = args.w[0], target_motif_definition = target_motif, seed = seed)
            if info is None:
                print(f"Warning (generate_random): Data generation failed (DyGraphGenMorMotifCon) for task '{task}', T={T}, N={N}, M={p}, W={args.w[0] if isinstance(args.w, list) else args.w}, seed={seed}. Skipping this instance.")
                return None # Return None to indicate failure for this seed
        else:
            dygen = DyGraphGenERCon()
            info = dygen.sample_dynamic_graph(T = T, N = N , p = p, seed = seed)
        obj_task = load_task(task, args)
        dygprompt = DyGraphPrompt(obj_task, args = args)
        
  
        # 2. Generate question-answer pairs based on graph information and task
        if info is None: # General check, covers cases if info is not set in other branches or if the specific check was missed
            print(f"Warning (generate_random): 'info' is None before QA generation for task '{task}', T={T}, N={N}, p/M={p}, seed={seed}. Skipping this instance.")
            return None
            
        qa = obj_task.generate_qa(info, *targs, **kwargs)

        # If generate_qa returns None (e.g., no valid data to generate QA), return directly
        if qa is None:
             print(f"Warning (generate_random): Task {task} failed to generate valid QA data for T={T}, N={N}, p={p}, seed={seed}. Skipping this instance.")
             return None

        # --- Save files (restore try-except) ---
        instance_folder = os.path.join(dir, folder_setting) # Instance data folder path
        os.makedirs(instance_folder, exist_ok=True) # Create instance folder
        info_file = os.path.join(instance_folder, f"graph.json") # Graph information file path
        qa_file = os.path.join(instance_folder, f"qa.json") # Question-answer pair file path
        
        # write files
        json.dump(info, open(info_file, "w"))

        # --- Modified: Convert set to list before JSON serialization ---
        if isinstance(qa.get("_original_context_set"), set):
            # Assume set elements are tuples, can be directly converted to a list
            qa["_original_context_set"] = list(qa["_original_context_set"])
        if isinstance(qa.get("_final_edges_set"), set):
             # Assume set elements are tuples, can be directly converted to a list
             qa["_final_edges_set"] = list(qa["_final_edges_set"])
        # --- End of modification ---

        json.dump(qa, open(qa_file, "w"))

        # --- Call visualization (unchanged, but inside generate_random) ---
        if visualization:
            try:
                vis_root_dir = os.path.join(dir, "visualizations")
                snapshot_dir = os.path.join(vis_root_dir, "snapshots")
                gif_dir = os.path.join(vis_root_dir, "gifs")
                os.makedirs(snapshot_dir, exist_ok=True)
                os.makedirs(gif_dir, exist_ok=True)

                snapshot_filename = f"snapshots_{folder_setting}.png"
                gif_filename = f"animation_{folder_setting}.gif"
                snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                gif_path = os.path.join(gif_dir, gif_filename)

                temporal_edges = info.get('edge_index')
                num_nodes = info.get('N')

                if temporal_edges and num_nodes is not None:
                    # print(f"  Generating visualization for {folder_setting}...") # Can be uncommented for detailed logs
                    visualization.visualize_graph(num_nodes, temporal_edges, snapshot_path)
                    visualization.create_colored_animation(num_nodes, temporal_edges, gif_path)
                else:
                    print(f"Warning (generate_random): Missing data for visualization (edge_index or N) for {folder_setting}")

            except Exception as e: # Keep catching visualization errors
                print(f"Error (generate_random): Error calling visualization function for {folder_setting}: {e}")
                import traceback
                traceback.print_exc()

        return folder_setting

    

    # run
    def gen(self, dir):
        """
        Generates all problem instance data (`graph.json`, `qa.json`) and visualizations for the specified task.
        Automatically performs timestamp data analysis after generation is complete.
        """
        print(f'--- Generating Base Data Files & Visualizations ---')
        args = self.args
        os.makedirs(dir, exist_ok=True)
        # Save gen args
        try:
            with open(os.path.join(dir, 'gen_args.json'), "w") as f:
                 json.dump(args.__dict__, f, indent=4)
        except Exception as e:
             print(f"Warning: Could not save gen_args.json: {e}")

        # --- Modified: Rename prompt_files -> instance_list ---
        instance_list = []
        label = 0
        task = args.task
        total_combinations = len(args.T) * len(args.N) * len(args.p)

        with tqdm(total=total_combinations * args.num_seed, desc="Generating Instances") as pbar:
            for T_val in args.T:
                for N_val in args.N:
                    for p_val in args.p:
                        seed = 0
                        successful_seeds_for_combo = 0
                        attempts = 0
                        max_attempts = args.num_seed * 5 + 10
                        while successful_seeds_for_combo < args.num_seed and attempts < max_attempts:
                             current_seed = seed
                             if args.task == "multi_motif_judge":
                                current_seed += 1
                             folder_setting = self.generate_random(dir, T_val, N_val, p_val, current_seed, label=label)
                             if folder_setting:
                                 instance_list.append(folder_setting) # Use new list name
                                 successful_seeds_for_combo += 1
                                 pbar.update(1)
                                 label = 1 - label
                             else:
                                 print("folder_setting is empty")
                             seed += 1
                             attempts += 1

                        if successful_seeds_for_combo < args.num_seed:
                             print(f"\nWarning: Generated {successful_seeds_for_combo}/{args.num_seed} instances for T={T_val}, N={N_val}, p={p_val} after {attempts} attempts.")

        # --- Modified: Save instance_list.json ---
        instance_list_path = os.path.join(dir, f"instance_list.json")
        try:
            # Use new key "instance_list"
            with open(instance_list_path, "w") as f:
                json.dump({"instance_list": instance_list}, f)
            print(f"\nSuccessfully generated data for {len(instance_list)} instances.")
            print(f"instance_list.json saved in {dir}")
            if visualization:
                 print(f"Visualizations saved in {os.path.join(dir, 'visualizations')}")

            # --- Add analysis step (depends on instance_list) ---
            if instance_list:
                 print("\n--- Starting Post-Generation Data Analysis ---")
                 self.analyze_data(dir)
            else:
                 print("\nSkipping data analysis as no instances were successfully generated.")
            # --- End of analysis step ---

        except Exception as e:
             print(f"Error saving instance_list.json: {e}")
             print("\nSkipping data analysis due to error saving instance_list.json.")
        # --- End of modification ---
    
    def run_one(self, task_folder):
        """
        Run a single batch of LLM calls to process all instance problems.
        
        Batch processing independence guarantees:
        1. Agent is initialized with disable_memory=True to disable memory functionality
        2. Each get_response call sets clear_history=True to clear history
        3. Ensures no memory association between 100 questions
        
        Both prompt_qa and answer files are saved to the model subdirectory.
        """
        args = self.args
        model = args.model
        use_agent = args.use_agent
        use_api = args.api
        print(f"use_api: {use_api}")
        if args.balance == 1:
            from .Balance.agent_manager import AgentManager
            agent_manager = AgentManager(
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_new_tokens=10240,
            memory_k=5,
            verbose=True,
            max_iterations=5,  # Reduce maximum iterations to force Agent to finish quickly
            handle_parsing_errors=True,
            disable_memory=False
           )
        elif use_api == 1:
            from .API.api import OpenAIAPI
            api = OpenAIAPI(key=args.key)
        elif use_agent == 1:
            from .Agent.agent_manager import AgentManager
            print("Initializing Agent Manager for batch processing...")
            agent_manager = AgentManager(
                key=args.key,
                model_name="gpt-4o-mini",
                temperature=0.1,
                max_new_tokens=10240,
                memory_k=5,
                verbose=True,
                max_iterations=5,  # Reduce maximum iterations to force Agent to finish quickly
                handle_parsing_errors=True,
                disable_memory=False
            )
            print("Agent Manager initialization complete, memory functionality disabled to ensure question independence")
        else:
            model_obj, tokenizer = initialize_model(model)
        try:
            obj_task = load_task(args.task, args)
            dygprompt = DyGraphPrompt(obj_task, args=args)
        except Exception as e:
            print(f"Error initializing task or prompt generator: {e}")
            return

        prompt_suffix = _generate_prompt_suffix(args)

        # --- Load instance_list.json ---
        instance_list_path = os.path.join(task_folder, "instance_list.json")
        try:
            with open(instance_list_path, "r") as f:
                 instance_info = json.load(f)
            instance_folders = instance_info["instance_list"] # Use new key
        except FileNotFoundError:
             print(f"Error: instance_list.json not found in {task_folder}. Cannot run.")
             return
        except json.JSONDecodeError:
             print(f"Error: instance_list.json in {task_folder} is corrupted. Cannot run.")
             return
        # --- End of modification ---
        print(f"--- Running LLM Inference ---")
        print(f"Model: {model}, Prompt Suffix: {prompt_suffix}")
        # --- Modified: Use instance_folders ---
        average_times = 0
        average_tokens = 0
        with tqdm(instance_folders, desc="Processing Instances") as bar:
            for folder_name in bar:
                try:
                    folder_path = os.path.join(task_folder, folder_name)
                    qa_file_path = os.path.join(folder_path, "qa.json")

                    model_subfolder_path = os.path.join(folder_path, model)
                    prompt_qa_filename = f"prompt_qa{prompt_suffix}.json"
                    prompt_qa_path = os.path.join(model_subfolder_path, prompt_qa_filename)

                    answer_filename = f"answer_{model}{prompt_suffix}.json"
                    answer_path = os.path.join(model_subfolder_path, answer_filename) 
                    
                    agent_filename = f"agent_{prompt_suffix}.log"
                    agent_log_path = os.path.join(model_subfolder_path, agent_filename)
                    
                    balance_agent_filename = f"balance_agent_{prompt_suffix}.log"
                    balance_agent_log_path = os.path.join(model_subfolder_path, balance_agent_filename)
                    
                    api_filename = f"api_{prompt_suffix}.log"
                    api_log_path = os.path.join(model_subfolder_path, api_filename)
                    

                    # Check if answer file exists
                    if os.path.exists(answer_path):
                        bar.set_postfix_str("Skipped (Answer Exists)")
                        continue

                    # Load qa data
                    try:
                        with open(qa_file_path, "r") as f:
                            qa = json.load(f)
                    except FileNotFoundError:
                         print(f"\nSkipping {folder_name}: qa.json not found.")
                         continue
                    except json.JSONDecodeError:
                         print(f"\nSkipping {folder_name}: qa.json is corrupted.")
                         continue    

                    # Dynamically generate prompt_qa dictionary
                    prompt_qa = dygprompt.generate_prompt_qa(**qa)
                    prompt = prompt_qa['prompt']
                    # prompt = prompt.encode('utf-8').decode('unicode_escape')
                    # --- Modified: Ensure model subdirectory exists before saving prompt_qa and answer ---
                    try:
                        os.makedirs(model_subfolder_path, exist_ok=True) # Create model subdirectory
                    except OSError as e:
                         print(f"\nError creating model subfolder {model_subfolder_path}: {e}")
                         continue # Cannot save, skip this instance

                    # Save prompt_qa
                    try:
                        with open(prompt_qa_path, "w") as f:
                             json.dump(prompt_qa, f, indent=4)
                    except Exception as e:
                         print(f"\nWarning: Could not save prompt_qa file to {prompt_qa_path}: {e}")
                    
                    if args.balance == 1:
                        answer_content = agent_manager.get_response(
                        task="predict_llm_agent",
                        prompt=prompt, 
                        agent_log_path=balance_agent_log_path,
                        clear_history=True 
                      )
                        response_text = answer_content["content"].strip().lower()
                        if "llm" in response_text:
                            use_api = 1
                            use_agent = 0
                            from .API.api import OpenAIAPI
                            api = OpenAIAPI(key=args.key)
                        else:
                            use_api = 0
                            use_agent = 1
                            from .Agent.agent_manager import AgentManager
           
                            agent_manager = AgentManager(
                                key=args.key,
                                model_name="gpt-4o-mini",
                                temperature=0.1,
                                max_new_tokens=10240,
                                memory_k=5,
                                verbose=True,
                                max_iterations=5,  # Reduce maximum iterations to force Agent to finish quickly
                                handle_parsing_errors=True,
                                disable_memory=False
                            )


                    if use_api == 1:
                        answer_content = api.get_response(model=model, prompt=prompt, api_log_path=api_log_path, max_tokens=args.max_tokens)
                        average_times += answer_content.get("duration", 0)
                        average_tokens += answer_content.get("usage", {}).get("total_tokens", 0)
                    elif use_agent == 1:
                        # Batch processing: each question is independent, no memory association
                        # get_response method will automatically clear any residual memory state
                        print(f"Processing independent question: {folder_name}")
                        answer_content = agent_manager.get_response(
                            task=args.task,
                            prompt=prompt, 
                            agent_log_path=agent_log_path,
                            clear_history=True  # Explicitly specify clearing history to ensure question independence
                        )
                        average_times += answer_content.get("time", 0)
                        average_tokens += float(answer_content.get("tokens", 0).get("total_tokens", 0))
                        print("Agent response content:")
                        print(answer_content["content"])
                        if args.verbose and answer_content.get("intermediate_steps"):
                            print("Execution steps:")
                            print(answer_content["intermediate_steps"]) 
                    else:
                    
                        answer_content = chat_with_model(model_obj, tokenizer, prompt, temperature=args.temperature, max_tokens=args.max_tokens)
                        tokens_used = answer_content.get('total_tokens', 0)
                        bar.set_postfix_str(f"Tokens Used: {tokens_used}")

                    # Save answer (to model subdirectory)
                    try:
                        with open(answer_path, "w") as f:
                             json.dump(answer_content, f) # Save complete dictionary containing content and token
                    except Exception as e:
                         print(f"\nError saving answer file {answer_path}: {e}")

                except FileNotFoundError as e:
                    bar.set_postfix_str(f"Error: File not found - {e}")
                    print(f"\nSkipping {folder_name}: Required file not found - {e}")
                except json.JSONDecodeError as e:
                    # Locate which specific file failed to parse
                    failed_file = "unknown json"
                    if 'qa_file_path' in locals() and qa_file_path in str(e): failed_file = qa_file_path
                    bar.set_postfix_str(f"Error: JSON decode - {failed_file}")
                    print(f"\nSkipping {folder_name}: JSON decode error in {failed_file} - {e}")
                except Exception as e:
                    bar.set_postfix_str(f"Error: {type(e).__name__}")
                    print(f"\nError processing {folder_name}: {type(e).__name__} - {e}")
                    # Can add more detailed error handling or logging here
                
        average_filename = f"average{prompt_suffix}.json"
        average_path = os.path.join(task_folder, average_filename)            
        with open(average_path, "w") as f:
            json.dump({"average_times": average_times/len(instance_folders), "average_tokens": average_tokens/len(instance_folders)}, f)
        print(f"Average times: {average_times/len(instance_folders)}, Average tokens: {average_tokens/len(instance_folders)}")

    def run(self, task_folder):
        """
        Execute LLM calls.
        If self.try_all is True, will loop calling run_one until all tasks are complete.
        Otherwise, only calls run_one once.

        Args:
            task_folder (str): Root directory for task data and results storage.
        """
        print(f"--- Starting LLM Run ---")
        print(f"Task: '{self.args.task}', Folder: {task_folder}")
        if self.try_all:
            while True:
                self.run_one(task_folder)
                print("\nChecking run status...")
                # Check completion status for specific configuration
                torun = self.check(task_folder)
                if torun == 0:
                    print("All instances seem to be processed.")
                    break
                print("Continue Try to Run")
                time.sleep(5)
        else:
            self.run_one(task_folder)
            

    def evaluate(self, task_folder):
        """
        Evaluate model performance.
        """
        args = self.args
        model = args.model
        task = args.task
        prompt_suffix = _generate_prompt_suffix(args) # Suffix is needed to load correct answer file
        obj_task = load_task(task, args)

        # --- Modified: Load instance_list.json ---
        instance_list_path = os.path.join(task_folder, "instance_list.json")
        try:
            with open(instance_list_path, "r") as f:
                 instance_info = json.load(f)
            instance_folders = instance_info["instance_list"]
        except FileNotFoundError:
             print(f"Error evaluating: instance_list.json not found in {task_folder}.")
             return
        except json.JSONDecodeError:
             print(f"Error evaluating: instance_list.json in {task_folder} is corrupted.")
             return
        # --- End of modification ---

        metrics = []
        total_tokens = []
        prompt_tokens = []
        completion_tokens = []
        # Restore original error folder lists for simplicity as requested previously
        # temporal_wrong_folders = []
        # completion_wrong_folders = []
        wrong_folders = [] # Add list to record metric = 0 (error) cases
        fail_folders = [] # Add list to record metric = -1 (failure/parsing error) cases
        num_times = []
        num_edges = []
        num_nodes = []

        print(f"--- Evaluating Results ---")
        # --- Modified: Print with suffix ---
        print(f"Task: {task}, Model: {model}, Suffix: {prompt_suffix}")
        # instance_folders = instance_folders[:68]
        # --- Modified: Use instance_folders ---
        for folder_name in tqdm(instance_folders, desc="Evaluating Instances"):
            folder_path = os.path.join(task_folder, folder_name)
            qa_file_path = os.path.join(folder_path, "qa.json")
            graph_path = os.path.join(folder_path, f"graph.json")
            # --- Modified: Load answer file from model subdirectory ---
            model_subfolder_path = os.path.join(folder_path, model)
            answer_path = os.path.join(model_subfolder_path, f"answer_{model}{prompt_suffix}.json")
            with open(qa_file_path, "r") as f:
                qa = json.load(f)   
            with open(answer_path, "r") as f:
                answer = json.load(f) # answer is the dict with 'content', 'total_tokens' etc.
            with open(graph_path, "r") as f:
                graph = json.load(f)

            # Now that sets in qa have been restored, they can be safely passed to obj_task.evaluate

            if args.task == "modify_motif":
                metric, extracted_answer = obj_task.evaluate(qa, answer["content"], args.w[0])
            elif args.task == "judge_contain_motif":
                metric, extracted_answer = obj_task.evaluate(qa, answer["content"], use_agent=args.use_agent, balance=args.balance)
            else:
                # Safely pass use_agent parameter, avoid errors for unsupported task classes
                try:
                    metric, extracted_answer = obj_task.evaluate(qa, answer["content"], use_agent=args.use_agent)
                except TypeError:
                    # If task class doesn't support use_agent parameter, don't pass it
                    metric, extracted_answer = obj_task.evaluate(qa, answer["content"])
            metrics.append(metric)
            
            # --- Restore: Original error classification (including fail_folders) ---
            if metric < 0:
                fail_folders.append(folder_name) # Parsing/format error
            elif metric == 0 or metric == 2:
               wrong_folders.append(folder_name) # Response error
            # --- Restore: Record tokens and stats ---
            total_tokens.append(answer.get('total_tokens', 0))
            prompt_tokens.append(answer.get('prompt_tokens', 0))
            completion_tokens.append(answer.get("completion_tokens", 0))
            num_times.append(graph.get('num_time', 0))
            num_edges.append(graph.get('num_edges', 0))
            num_nodes.append(graph.get('num_nodes', 0))
        # --- Restore: Original statistics calculation (modified to include fail_rate) ---
        num_all = len(metrics)
        if num_all == 0:
            print("Error: No instances were successfully evaluated.")
            return
        print(num_all)
        print(metrics)
        right_rate = sum(metric for metric in metrics if metric > 0 and metric <= 1) / num_all if num_all > 0 else 0
        # Error rate: metric = 0, -1, -2 (logic error, sorting error, completeness error)
        wrong_rate = sum(1 for m in metrics if m == 0 or m == 2) / num_all if num_all > 0 else 0
        # Failure rate: metric = -3 (parsing error, file not found, JSON error, unexpected error during evaluation)
        fail_rate = sum(1 for m in metrics if m == -1) / num_all if num_all > 0 else 0

        total_tokens_sum = sum(t for t in total_tokens if t is not None) # Ensure None values are ignored in sum
        valid_token_counts = len([t for t in total_tokens if t is not None])
        average_tokens_calc = total_tokens_sum / valid_token_counts if valid_token_counts > 0 else 0

        # --- Restore: Original results structure (updated error/failure rates) ---
        results = {
            "task": task,
            "model": model,
            "prompt_config_suffix": prompt_suffix,
            "correct_rate": right_rate, # Correct rate (metric=1)
            "error_rate": wrong_rate,   # Error rate (metric=0, -1, -2)
            "failure_rate": fail_rate,  # Failure rate (metric=-3)
            "average_tokens": average_tokens_calc,
            "total_tokens": total_tokens_sum,
            "average_num_times": np.mean(num_times) if num_times else 0,
            "average_num_edges": np.mean(num_edges) if num_edges else 0,
            "average_num_nodes": np.mean(num_nodes) if num_nodes else 0,
            "metrics": metrics, # Original metric list
            "total_tokens_list": total_tokens,
            "prompt_tokens_list": prompt_tokens,
            "completion_tokens_list": completion_tokens,
            "error_folders": wrong_folders, # Combine logic errors and completeness errors
            "failure_folders": fail_folders, # Separately list failed folders
            # Keep old classifications if backward compatibility is needed
            # "temporal_wrong_folders": temporal_wrong_folders,
            # "completion_wrong_folders": completion_wrong_folders,
        }

        # --- Modified: Save results file with suffix ---
        results_filename = f"results_{args.motif_name}_{model}{prompt_suffix}.json"
        results_save_path = os.path.join(task_folder, results_filename)
        try:
             with open(results_save_path, "w") as f:
                 json.dump(results, f, indent=4)
             print(f"\nEvaluation results saved to: {results_save_path}")
        except Exception as e:
            print(f"\nError saving evaluation results to {results_save_path}: {e}")

        # --- Restore: Original print format (updated error/failure rates) ---
        print(f"\n--- Evaluation Summary ---")
        print(f"Task: {task}, Model: {model}, Suffix: {prompt_suffix}") # Add Suffix
        # Update print output to reflect new rates
        print(f"Correct Rate: {right_rate:.4f}, Error Rate: {wrong_rate:.4f}, Failure Rate: {fail_rate:.4f}")
        print(f"Average Tokens: {average_tokens_calc:.2f}, Total Tokens: {total_tokens_sum}")
        # Restore np.std calculation
        time_std = np.std(num_times) if num_times else 0
        edge_std = np.std(num_edges) if num_edges else 0
        node_std = np.std(num_nodes) if num_nodes else 0
        print(f"Num_time : {results['average_num_times']:.2f}+-{time_std:.2f} Num_edges : {results['average_num_edges']:.2f}+-{edge_std:.2f} Num Nodes : {results['average_num_nodes']:.2f}+-{node_std:.2f}")
        print(f"Total Instances Evaluated: {num_all}")
        if args.task == "sort_edge":
            print("--------------------------------")
            print("is_complete and not is_sorted: ", sum(1 for m in metrics if m == 2) / num_all if num_all > 0 else 0)
            print("not_complete and is_sorted: ", sum(1 for m in metrics if m == 3) / num_all if num_all > 0 else 0)
            print("not_complete and not is_sorted: ", sum(1 for m in metrics if m == 4) / num_all if num_all > 0 else 0)
            print("--------------------------------")

    def show(self, dir):
        """
        Display results for a single task. (Load results file with suffix)
        """
        args = self.args
        task = args.task
        task_folder = args.task_folder
        model = args.model
        prompt_suffix = _generate_prompt_suffix(args)
        # --- Modified: Load results file with suffix ---
        results_file = os.path.join(task_folder, f"results_{model}{prompt_suffix}.json")

        print(f"--- Showing Results ---") 
        print(f"Attempting to load: {results_file}")

        try:
             with open(results_file, "r") as f:
                  results_data = json.load(f)
        except FileNotFoundError:
             print(f"Error: Results file not found. Please ensure evaluation was run with matching parameters.")
             return
        except json.JSONDecodeError:
             print(f"Error: Results file is corrupted.")
             return

        # --- Restore: Print more information ---
        print(f"\n--- Results Summary (from {os.path.basename(results_file)}) ---")
        print(f"Task: {results_data.get('task', 'N/A')}")
        print(f"Model: {results_data.get('model', 'N/A')}")
        print(f"Suffix: {results_data.get('prompt_config_suffix', 'N/A')}") # Display suffix
        print(f"Average Accuracy (metric>0): {results_data.get('average_acc', 'N/A'):.4f}")
        print(f"Wrong Rate (metric<0): {results_data.get('wrong_rate', 'N/A'):.2f}")
        print(f"Average Tokens: {results_data.get('average_tokens', 'N/A'):.2f}")
        print(f"Total Tokens: {results_data.get('total_tokens', 'N/A')}")
        # print(f"Temporal Errors (metric=-1): {len(results_data.get('temporal_wrong_folders', []))}")
        # print(f"Completion Errors (metric=-2): {len(results_data.get('completion_wrong_folders', []))}")

        # --- Restore: Remove grouping logic explanation ---
        # print("\nNote: Detailed breakdown by T, N, p requires adapting the 'show' method or storing more data in results.")


    def analyze_data(self, task_folder):
        """
        Analyze timestamp, node count, and edge count distributions of generated data,
        and save statistics and visualizations.
        """
        if not plotting_available:
            print("Error: Plotting libraries (matplotlib/seaborn) not available. Cannot perform analysis.")
            return

        args = self.args
        instance_list_path = os.path.join(task_folder, "instance_list.json")
        if not os.path.exists(instance_list_path):
             print(f"Error analyzing: instance_list.json not found in {task_folder}.")
             return
        try:
            with open(instance_list_path, "r") as f:
                 instance_info = json.load(f)
            instance_folders = instance_info["instance_list"]
        except json.JSONDecodeError:
             print(f"Error analyzing: instance_list.json in {task_folder} is corrupted.")
             return


        all_timestamps = []
        all_node_counts = []
        all_edge_counts = []

        valid_instances = 0
        folders_to_analyze = instance_folders

        print(f"Analyzing graph properties from {len(folders_to_analyze)} instances...")
        for folder_name in tqdm(folders_to_analyze, desc="Analyzing Graphs"): # Update description
            graph_path = os.path.join(task_folder, folder_name, "graph.json")
            try:
                with open(graph_path, "r") as f:
                    graph_data = json.load(f)


                node_count = graph_data.get('N', None)
                edge_count = graph_data.get('num_edges', None) # num_edges usually equals len(edge_index)
                timestamps_in_file = [int(edge[2]) for edge in graph_data.get('edge_index', []) if len(edge) > 2] # Ensure edge tuple is valid

                if node_count is not None and edge_count is not None:
                    all_node_counts.append(node_count)
                    all_edge_counts.append(edge_count)
                    all_timestamps.extend(timestamps_in_file) # Only add timestamps when nodes and edges are valid? Or handle separately? Separate handling is better
                    valid_instances += 1
                else:
                    print(f"\nWarning: Skipping instance {folder_name} due to missing 'N' or 'num_edges' in graph.json.")


            except FileNotFoundError:
                 print(f"\nWarning: graph.json not found for instance {folder_name}. Skipping.")
            except json.JSONDecodeError:
                 print(f"\nWarning: graph.json for instance {folder_name} is corrupted. Skipping.")
            except Exception as e:
                 print(f"\nWarning: Error processing {folder_name} during analysis: {e}")

        valid_nodes = [n for n in all_node_counts if n is not None]
        valid_edges = [e for e in all_edge_counts if e is not None]
        valid_timestamps = [t for t in all_timestamps if t is not None] # Ensure timestamp list is also clean


        if not valid_nodes or not valid_edges or not valid_timestamps:
            print(f"Error: Could not collect sufficient graph property data from any valid instances (requires nodes, edges, and timestamps). Valid instances: {valid_instances}")
            return

        print(f"Analysis complete. Collected data from {valid_instances}/{len(instance_folders)} valid instances.")
        print(f"  Node count: {len(valid_nodes)}, Edge count: {len(valid_edges)}, Timestamp events: {len(valid_timestamps)}")

        # --- Modified: Calculate all statistics ---
        stats_data = {
            "nodes": {
                "count": len(valid_nodes),
                "min": int(np.min(valid_nodes)) if valid_nodes else 0,
                "max": int(np.max(valid_nodes)) if valid_nodes else 0,
                "mean": float(np.mean(valid_nodes)) if valid_nodes else 0,
                "std": float(np.std(valid_nodes)) if valid_nodes else 0,
            },
            "edges": {
                "count": len(valid_edges),
                "min": int(np.min(valid_edges)) if valid_edges else 0,
                "max": int(np.max(valid_edges)) if valid_edges else 0,
                "mean": float(np.mean(valid_edges)) if valid_edges else 0,
                "std": float(np.std(valid_edges)) if valid_edges else 0,
            },
            "timestamps": {
                "count": len(valid_timestamps),
                "min": int(np.min(valid_timestamps)) if valid_timestamps else 0,
                "max": int(np.max(valid_timestamps)) if valid_timestamps else 0,
                "mean": float(np.mean(valid_timestamps)) if valid_timestamps else 0,
                "std": float(np.std(valid_timestamps)) if valid_timestamps else 0,
                "counts_per_step": dict(Counter(valid_timestamps)) if valid_timestamps else {}
            }
        }


        try:
            vis_dir = os.path.join(task_folder, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            stats_save_path = os.path.join(vis_dir, "graph_analysis_stats.json")
            with open(stats_save_path, "w") as f:
                json.dump(stats_data, f, indent=4)
            print(f"Graph property statistics saved to: {stats_save_path}")


            fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # 1 row 3 columns

            # Subplot 1: Node count distribution
            sns.histplot(valid_nodes, ax=axes[0], kde=False, discrete=True)
            axes[0].set_title('Node Count Distribution')
            axes[0].set_xlabel('Number of Nodes (N)')
            axes[0].set_ylabel('Frequency')
            max_node = stats_data["nodes"]["max"]
            if max_node <= 30:
                 axes[0].set_xticks(range(stats_data["nodes"]["min"], max_node + 1))
            elif max_node > 0:
                 tick_step = max(1, max_node // 10)
                 axes[0].set_xticks(range(stats_data["nodes"]["min"], max_node + tick_step, tick_step))


            # Subplot 2: Edge count distribution
            sns.histplot(valid_edges, ax=axes[1], kde=True) # Use KDE to see a smoother distribution
            axes[1].set_title('Edge Count Distribution')
            axes[1].set_xlabel('Number of Edges')
            axes[1].set_ylabel('Frequency / Density')

            # Subplot 3: Timestamp distribution (keep original histogram logic)
            max_T = stats_data["timestamps"]["max"] + 1 if valid_timestamps else 1
            bins = range(max_T + 1) if max_T > 1 else [0, 1]
            sns.histplot(valid_timestamps, bins=bins, kde=False, stat="count", discrete=True, ax=axes[2])
            axes[2].set_title('Timestamp Distribution (Event Count)')
            axes[2].set_xlabel('Timestamp (t)')
            axes[2].set_ylabel('Number of Events')
            if max_T <= 20 and max_T > 1:
                 axes[2].set_xticks(bins[:-1])
            elif max_T > 1:
                 tick_step = max(1, (max_T - 1) // 10)
                 axes[2].set_xticks(range(0, max_T, tick_step))


            plt.tight_layout()
            plot_filename = os.path.join(vis_dir, "graph_analysis_distributions.png") # New filename
            plt.savefig(plot_filename, dpi=150)
            print(f"Combined graph property distribution plots saved to: {plot_filename}") # Update print message
            plt.close(fig) # Close figure to free memory

        except Exception as e:
            print(f"Error: Error during statistics processing or plotting: {e}")
            import traceback
            traceback.print_exc() # Print detailed error trace
        
    def execute(self, dir):
        """Execute the corresponding operation based on args.t."""
        args = self.args
        task_folder = args.task_folder
        # if args.api == 1:
        #     task_folder = os.path.join(task_folder, "api")
        # if args.use_agent == 1:
        #     task_folder = os.path.join(task_folder, "agent")
        if args.task == "judge_is_motif" or args.task == "modify_motif" or args.task == "judge_contain_motif":
            task_folder = os.path.join(task_folder, args.motif_name)
            # task_folder = os.path.join(task_folder, f"M{args.p}")
        print(f"\nExecuting action '{args.t}' for task '{args.task}' in folder '{task_folder}'")
        if args.t == "clear":
            remove_dir(task_folder)
        elif args.t == "gen":
            self.gen(task_folder) # gen method now automatically calls analyze_data
        elif args.t == "run":
            self.run(task_folder)
        elif args.t == "eval":
            self.evaluate(task_folder)
        elif args.t == "check":
            self.check(task_folder)
        elif args.t == "show":
            self.show(task_folder)
        # --- Added: Explicit analyze call ---
        elif args.t == "analyze":
            print(f"\n--- Explicitly running Data Analysis ---")
            self.analyze_data(task_folder)
        # --- End of addition ---
        else:
            print(f"Error: Action '{args.t}' not implemented.")
            raise NotImplementedError

