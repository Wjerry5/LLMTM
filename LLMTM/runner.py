import os
from .utils import  remove_dir, load_task
import json
from openai import OpenAI
# from .utils import send_prompt, DyGraphPrompt, DyGraphGenERCon, load_task 
from .utils import DyGraphGenControlMotif,DyGraphGenEdge, DyGraphPrompt, DyGraphGenERCon, DyGraphGenMotifCon,load_task, DyGraphGenMorMotifCon 
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import openai 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer 
from collections import defaultdict, Counter 
import langchain
from langchain.agents import initialize_agent, AgentType   

import networkx as nx
from scipy.stats import entropy
import joblib
from collections import Counter, defaultdict
try:
    from .utils import visualization
except ImportError:
    visualization = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plotting_available = True
except ImportError:
    plotting_available = False
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

try:
    from openai import RateLimitError, AuthenticationError, APIConnectionError
    OPENAI_RATE_LIMIT_ERROR = RateLimitError
    OPENAI_AUTH_ERROR = AuthenticationError
    OPENAI_CONNECTION_ERROR = APIConnectionError
except ImportError:
    try:
        from openai.error import RateLimitError, AuthenticationError, APIConnectionError
        OPENAI_RATE_LIMIT_ERROR = RateLimitError
        OPENAI_AUTH_ERROR = AuthenticationError
        OPENAI_CONNECTION_ERROR = APIConnectionError
    except ImportError:
        class FallbackOpenAIError(Exception):
            pass
        OPENAI_RATE_LIMIT_ERROR = FallbackOpenAIError
        OPENAI_AUTH_ERROR = FallbackOpenAIError
        OPENAI_CONNECTION_ERROR = FallbackOpenAIError

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
    "3-star":       [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')],             # k=4, l=3 
    "4-path":       [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')],             # k=4, l=3
    "4-cycle":      [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')], # k=4, l=4
    "4-tailedtriangle": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')], # k=4, l=4
    "butterfly":    [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')], # k=4, l=4 
    "4-chordalcycle": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')], # k=4, l=5
    "4-clique":     [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')], # k=4, l=6
    "bitriangle":   [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')], # k=6, l=6
}


MODEL_PATHS = {
    "Qwen_14B": "LLMTM/DeepSeek-R1-Distill-Qwen-14B",
    "Llama_8B": "LLMTM/Llama-3.1-Nemotron-Nano-8B-v1",
    "Deepseek_7B": "LLMTM/DeepSeek-R1-Distill-Qwen-7B",
    "Deepseek_7B_chat": "/LLMTM/llama_2_7B_chat",
    "Llama_7B_chat": "/LLMTM/llama_2_7B_chat",
    "Llama_13B_chat": "/LLMTM/llama_2_13B_chat",
    "Qwen-32B-Chat": "/LLMTM/Qwen1.5-32B-Chat-AWQ",
    "gpt-4o-mini": "",
    "deepseek-r1-250528": "",
    "QwQ": "/LLMTM/QwQ-32B",
    "Qwen2.5_32B": "/LLMTM/Qwen2.5-32B-Instruct",
    "Qwen_32B": "/LLMTM/DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-32B":"",
    "DeepSeek-R1-Distill-Qwen-14B":"",
    "o3-2025-04-16": "",
    "pangu_auto": "/home/ma-user/work/LLMTM/Pangu",
}

def initialize_model(model):
    model_path = MODEL_PATHS[model]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model_obj.eval()
    return model_obj, tokenizer

def chat_with_model(model, prompt, temperature, max_tokens):

    torch.cuda.empty_cache()
    try:
        debug_prompt_file = "prompt_debug_runner.txt"
        with open(debug_prompt_file, "w", encoding="utf-8") as f_debug:
            f_debug.write(prompt)
        print(f"[Debug Info] Prompt saved to: {debug_prompt_file}")
    except Exception as e_debug:
        print(f"[Debug Info] Error saving prompt to file: {e_debug}")

    if model == "pangu_auto":
        client = OpenAI(
            api_key="sk-xxx",  # Any string will work
            base_url="http://127.0.0.1:8888/v1", 
            default_headers={
                "Connection": "close",  # Avoid keep-alive
                "Keep-Alive": "timeout=0"  # Disable keep-alive
            }
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
        # do_sample, pad_token_id, repetition_penalty are local model parameters, not supported in API
    )
    response_text = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    result = {
        "content": response_text.strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    return result



def analyze_graph_features(edges):
    """
    Analyze a series of general structural and cognitive complexity metrics of a graph.

    Metrics include:
    - num_nodes: Number of nodes
    - num_edges: Number of edges
    - edge_density: Graph density
    - num_nodes_ge_3: Number of nodes with degree >= 3 (measuring core complexity)
    - num_nodes_eq_2: Number of nodes with degree = 2 (measuring path complexity)
    - cyclomatic_complexity: Cyclomatic complexity (measuring graph "entanglement")
    - edge_locality_score: Edge locality score (measuring LLM's sequential cognitive load)
    - temporal_clash_count: Temporal clash count (measuring temporal dimension ambiguity)
    
    Args:
        edges (list): List of edges, each in the form (u, v, t, op).
    
    Returns:
        dict: A dictionary containing various general features of the graph.
    """
    # --- 1. Data preprocessing and graph construction ---
    G = nx.Graph()
    active_edges = [edge for edge in edges if edge[3] == 'a']
    
    # If there are no active edges, return a dictionary with default values
    if not active_edges:
        return {
            "num_nodes": 0, "num_edges": 0, "edge_density": 0,
            "num_nodes_ge_3": 0, "num_nodes_eq_2": 0, "cyclomatic_complexity": 1,
            "edge_locality_score": 0, "temporal_clash_count": 0, "ratio_nodes_ge_3": 0, "ratio_nodes_eq_2": 0
        }
    
    # Extract node and edge information from active edges to build the graph
    edge_tuples = [(u, v) for u, v, t, op in active_edges]
    G.add_edges_from(edge_tuples)
    
    # --- 2. Basic and structural complexity metrics ---
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Graph density
    if num_nodes > 1:
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        # Ensure the denominator is not 0
        edge_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    else:
        edge_density = 0
        
    # Cyclomatic complexity (E - N + P)
    # P = number of connected components
    num_connected_components = nx.number_connected_components(G)
    cyclomatic_complexity = num_edges - num_nodes + num_connected_components

    # Node degree related metrics
    degrees = dict(G.degree())
    num_nodes_ge_3 = sum(1 for degree in degrees.values() if degree >= 3)
    num_nodes_eq_2 = sum(1 for degree in degrees.values() if degree == 2)
    ratio_nodes_ge_3 = num_nodes_ge_3/num_nodes
    ratio_nodes_eq_2 = num_nodes_eq_2/num_nodes
    
    # Filter core nodes for calculating locality and temporal metrics
    core_nodes = [node for node, degree in degrees.items() if degree >= 3]

    # --- 3. Cognitive and temporal complexity metrics ---
    # Create a mapping from nodes to their related edge information (timestamps and indices)
    node_to_edge_info = defaultdict(list)
    for i, (u, v, t, op) in enumerate(edges):
        if op == 'a':
            info = {'t': t, 'idx': i}
            node_to_edge_info[u].append(info)
            if u != v:
                node_to_edge_info[v].append(info)

    # Edge locality score (calculated only on core nodes)
    center_locality_scores = []
    for node in core_nodes:
        indices = [info['idx'] for info in node_to_edge_info[node]]
        if len(indices) >= 2:
            center_locality_scores.append(np.std(indices))
    
    edge_locality_score = np.mean(center_locality_scores) if center_locality_scores else 0
    
    # Temporal clash count (calculated only on core nodes, then summed)
    total_clashes = 0
    for node in core_nodes:
        timestamps = [info['t'] for info in node_to_edge_info[node]]
        if not timestamps:
            continue
        time_counts = Counter(timestamps)
        for count in time_counts.values():
            if count > 1:
                total_clashes += count * (count - 1) / 2

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "edge_density": edge_density,
        "num_nodes_ge_3": num_nodes_ge_3,
        "num_nodes_eq_2": num_nodes_eq_2,
        "cyclomatic_complexity": cyclomatic_complexity,
        "edge_locality_score": edge_locality_score,
        "temporal_clash_count": int(total_clashes),
        "ratio_nodes_ge_3": ratio_nodes_ge_3,
        "ratio_nodes_eq_2": ratio_nodes_eq_2,
    }

def predict_with_xgboost(features_dict, model_path=None, model_info=None):
    """
    Make predictions using a trained XGBoost model.
    
    Parameters:
    features_dict: Dictionary containing feature values, e.g.:
                  {'num_nodes': 10, 'edge_density': 0.5, ...}
    model_path: Path to the model file (must be provided if model_info is None)
    model_info: Dictionary containing model information (must be provided if model_path is None)
    
    Returns:
    prediction: 0 indicates using the Agent, 1 indicates using the LLM
    probability: Probability of using the LLM
    """
    if model_info is None and model_path is None:
        raise ValueError("Either model_path or model_info must be provided")
    
    if model_info is None:
        try:
            # model_info = joblib.load(model_path)
            
            model = xgb.XGBClassifier()
            model.load_model(model_path) 
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, None
    
    # model = model_info['model']
    # features = model_info['features']
    
    # Build feature array
    features = ["num_edges","cyclomatic_complexity","edge_locality_score","ratio_nodes_ge_3","ratio_nodes_eq_2"]
    X = np.array([[features_dict[f] for f in features]])
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Get probability of using the LLM
    
    return prediction, probability


def _generate_prompt_suffix(args):
    """Generates a unique suffix based on prompt configuration arguments."""
    
    k = args.__dict__.get('k', args.__dict__.get('num_examplars', 0)) 
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
            

# Defines the Runner class, responsible for managing and executing the entire experiment workflow, including:
# 1. Generate data and questions (gen) - only generates graph.json and qa.json
# 2. Call LLM to get answers (run) - dynamically generates prompts, saves answer with suffix
# 3. Evaluate model performance (evaluate) - loads answer with suffix, saves results with suffix
# 4. Check run status (check) - checks answer file with suffix
# 5. Summarize and display results (show)
class Runner:
    def __init__(self, args, try_all = False) -> None:
        self.args = args
        self.try_all = try_all
        
    def check(self, task_folder):
        args = self.args
        model = args.model
        prompt_suffix = _generate_prompt_suffix(args)
        instance_list_path = os.path.join(task_folder, "instance_list.json")
        try:
            with open(instance_list_path, "r") as f:
                 instance_info = json.load(f)
            instance_folders = instance_info["instance_list"] 
            print(instance_folders)
        except FileNotFoundError:
             print(f"Error: instance_list.json not found in {task_folder}. Please run '-t gen' first.")
             return -1
        except json.JSONDecodeError:
             print(f"Error: instance_list.json in {task_folder} is corrupted.")
             return -1


        finish = []
        torun = []
        sdict = {"num_edges": [], "num_nodes": [], "num_time": []}

        for i, folder_name in enumerate(instance_folders):
            folder_path = os.path.join(task_folder, folder_name)
            try:
                graph = json.load(open(os.path.join(folder_path, "graph.json")))
                for k, v in sdict.items():
                    v.append(graph.get(k, 0)) 
            except FileNotFoundError:
                print(f"Warning: graph.json not found in {folder_path}")
                torun.append(i) 
                continue
            except json.JSONDecodeError:
                 print(f"Warning: graph.json in {folder_path} is corrupted.")
                 torun.append(i) 
                 continue

       
            model_subfolder_path = os.path.join(folder_path, model)
            answer_path = os.path.join(model_subfolder_path, f"answer_{model}{prompt_suffix}.json")

            if os.path.exists(answer_path):
                try:
                    json.load(open(answer_path, "r"))
                    finish.append(i)
                except json.JSONDecodeError:
                    print(f"Warning: Answer file {answer_path} is corrupted.")
                    torun.append(i) 
            else:
                torun.append(i)
                
        print(f"--- Checking Status ---")
        print(f"Task Folder: {task_folder}")
        print(f"Model: {model}")
        print(f"Prompt Suffix: {prompt_suffix}")
        print(f"Finish {len(finish)}, ToRun {len(torun)} (Total {len(instance_folders)})")
        if sdict["num_edges"]: 
             print("Graph Stats (Avg±Std): " + "".join(f"{k}:{np.mean(v):.2f}±{np.std(v):.2f} " for k,v in sdict.items() if v))
        else:
             print("Graph Stats: No valid graph data found to compute statistics.")
        return len(torun)
        
    
    def generate_random(self, dir, T, N, p, seed, *targs ,**kwargs):
        """
        """
        args = self.args
        task = args.task

        if args.dataset != "random": # real-world dataset
            folder_setting = f"dyg{seed}"
        else:
            folder_setting = f"T{T}_N{N}_p{p}_seed{seed}"
        
        
        info = None
        if args.dataset != "random": # real-world dataset
            infos = json.load(open(f"./LLMTM/dataset/{args.dataset}/{args.dataset}_subgraphs.json"))
            info = infos[args.motif_name].get(f"subgraphs", {}).get(f"{seed}", None)

        elif args.task == "judge_contain_motif":
            p = int(p)
            folder_setting = f"T{T}_N{N}_M{p}_seed{seed}"
            dygen = DyGraphGenControlMotif()
            info = dygen.generate_graph_with_motif_control(M = p, seed = seed, motif_name = args.motif_name, T = T)
        elif args.motif == 1 and task == "judge_motif":
            folder_setting = f"T{T}_N{N}_p{p}_seed{seed}"
            p = int(p)
            M = PREDEFINED_MOTIFS[args.motif_name]
            dygen = DyGraphGenMotifCon()
            info = dygen.sample_dynamic_graph(T + T, predefined_motif = M, motif_time_window = T, seed=seed)
        elif args.motif == 1 and task == "modify_dyg":
            folder_setting = f"T{T}_N{N}_p{p}_seed{seed}"
            target_motif = PREDEFINED_MOTIFS[args.motif_name]
            p = int(p)
            dygen = DyGraphGenMorMotifCon()
            info = dygen.sample_dynamic_graph(T_total_time = T, N_total_nodes = N , M_target_edges = p, W_motif_window = args.w[0], target_motif_definition = target_motif, seed = seed)
            if info is None:
                print(f"Warning (generate_random): Data generation failed(DyGraphGenMorMotifCon) for task '{task}', T={T}, N={N}, M={p}, W={args.w[0] if isinstance(args.w, list) else args.w}, seed={seed}. Skipping this instance.")
                return None # Return None to indicate failure for this seed
        else:
            dygen = DyGraphGenERCon()
            info = dygen.sample_dynamic_graph(T = T, N = N , p = p, seed = seed)
        obj_task = load_task(task, args)
        dygprompt = DyGraphPrompt(obj_task, args = args)
        
  
        # 2. Generate QA pairs based on graph info and task
        if info is None: # General check, covers cases if info is not set in other branches or if the specific check was missed
            print(f"Warning (generate_random): 'info' is None before QA generation for task '{task}', T={T}, N={N}, p/M={p}, seed={seed}. Skipping this instance.")
            return None
            
        qa = obj_task.generate_qa(info, *targs, **kwargs)

        # If generate_qa returns None (e.g., no valid data to generate QA), return directly
        if qa is None:
             print(f"Warning (generate_random): Task {task} failed to generate valid QA data for T={T}, N={N}, p={p}, seed={seed}. Skipping this instance.")
             return None

        # --- save ---
        instance_folder = os.path.join(dir, folder_setting) 
        os.makedirs(instance_folder, exist_ok=True) 
        info_file = os.path.join(instance_folder, f"graph.json") 
        qa_file = os.path.join(instance_folder, f"qa.json") 
        
        # write files
        json.dump(info, open(info_file, "w"))

  
        if isinstance(qa.get("_original_context_set"), set):
            qa["_original_context_set"] = list(qa["_original_context_set"])
        if isinstance(qa.get("_final_edges_set"), set):
             qa["_final_edges_set"] = list(qa["_final_edges_set"])


        json.dump(qa, open(qa_file, "w"))

        
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
                    visualization.visualize_graph(num_nodes, temporal_edges, snapshot_path)
                    visualization.create_colored_animation(num_nodes, temporal_edges, gif_path)
                else:
                    print(f"Warning (generate_random): Missing data required for visualization (edge_index or N) for {folder_setting}")

            except Exception as e: 
                print(f"Error (generate_random): Error calling visualization function for {folder_setting}: {e}")
                import traceback
                traceback.print_exc()

        return folder_setting

    

    # run
    def gen(self, dir):
        """
        Generates all problem instance data (`graph.json`, `qa.json`) and visualizations for the specified task(s).
        After generation is complete, automatically performs timestamp data analysis.
        """
        print(f'--- Generating Base Data Files & Visualizations ---')
        args = self.args
        os.makedirs(dir, exist_ok=True)
        # 保存 gen args
        try:
            with open(os.path.join(dir, 'gen_args.json'), "w") as f:
                 json.dump(args.__dict__, f, indent=4)
        except Exception as e:
             print(f"Warning: Could not save gen_args.json: {e}")
        
        if args.dataset != "random": # real-world dataset
            instance_list = []
            task = args.task
            for i in range(20):
                folder_setting = self.generate_random(dir, args.T, args.N, args.p, seed=i)
                instance_list.append(folder_setting)
        else:

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
                                if args.task == "judge_multi_motif":
                                    current_seed += 1
                                folder_setting = self.generate_random(dir, T_val, N_val, p_val, current_seed, label=label)
                                if folder_setting:
                                    instance_list.append(folder_setting) 
                                    successful_seeds_for_combo += 1
                                    pbar.update(1)
                                    label = 1 - label
                                else:
                                    print("folder_setting is empty")
                                seed += 1
                                attempts += 1

                            if successful_seeds_for_combo < args.num_seed:
                                print(f"\nWarning: Generated {successful_seeds_for_combo}/{args.num_seed} instances for T={T_val}, N={N_val}, p={p_val} after {attempts} attempts.")

        instance_list_path = os.path.join(dir, f"instance_list.json")
        try:
            with open(instance_list_path, "w") as f:
                json.dump({"instance_list": instance_list}, f)
            print(f"\nSuccessfully generated data for {len(instance_list)} instances.")
            print(f"instance_list.json saved in {dir}")
            if visualization:
                 print(f"Visualizations saved in {os.path.join(dir, 'visualizations')}")

            if instance_list:
                 print("\n--- Starting Post-Generation Data Analysis ---")
                 self.analyze_data(dir)
            else:
                 print("\nSkipping data analysis as no instances were successfully generated.")

        except Exception as e:
             print(f"Error saving instance_list.json: {e}")
             print("\nSkipping data analysis due to error saving instance_list.json.")
        # --- 修改结束 ---
    
    def run_one(self, task_folder):
        """
        Runs a single batch of LLM calls, processing all instance questions.
        
        Batch processing independence guarantee:
        1. Set disable_memory=True during Agent initialization to disable memory function
        2. Set clear_history=True on every get_response call to clear history
        3. Ensures there is absolutely no memory association between the 100 questions
        
        Saves both the prompt_qa and answer files to the model subdirectory.
        """
        args = self.args
        model = args.model
        use_agent = args.use_agent
        use_api = args.api
        
        if use_api == 1:
            from .api.api import OpenAIAPI
            args.key = "123"
            api = OpenAIAPI(key=args.key)
        elif use_agent == 1:
            if args.model == "pangu_auto":
                from .agent_pangu.agent_manager import AgentManager
            else:
                from .agent_4o.agent_manager import AgentManager

            agent_manager = AgentManager(
                model_name=model,
                temperature=0.1,
                max_new_tokens=10240,
                memory_k=5,
                verbose=True,
                max_iterations=5,  # Reduce maximum iterations to force Agent to finish quickly
                handle_parsing_errors=True,
                disable_memory=False
            )

            print("🤖 Initializing Agent Manager for batch processing...")
            
        # else:
            # model_obj, tokenizer = initialize_model(model)
        try:
            obj_task = load_task(args.task, args)
            dygprompt = DyGraphPrompt(obj_task, args=args)
        except Exception as e:
            print(f"Error initializing task or prompt generator: {e}")
            return

        prompt_suffix = _generate_prompt_suffix(args)

        instance_list_path = os.path.join(task_folder, "instance_list.json")
        try:
            with open(instance_list_path, "r") as f:
                 instance_info = json.load(f)
            instance_folders = instance_info["instance_list"] 
        except FileNotFoundError:
             print(f"Error: instance_list.json not found in {task_folder}. Cannot run.")
             return
        except json.JSONDecodeError:
             print(f"Error: instance_list.json in {task_folder} is corrupted. Cannot run.")
             return

        print(f"--- Running LLM Inference ---")
        print(f"Model: {model}, Prompt Suffix: {prompt_suffix}")

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


                    if os.path.exists(answer_path):
                        bar.set_postfix_str("Skipped (Answer Exists)")
                        continue

                    try:
                        with open(qa_file_path, "r") as f:
                            qa = json.load(f)
                    except FileNotFoundError:
                         print(f"\nSkipping {folder_name}: qa.json not found.")
                         continue
                    except json.JSONDecodeError:
                         print(f"\nSkipping {folder_name}: qa.json is corrupted.")
                         continue

                    prompt_qa = dygprompt.generate_prompt_qa(**qa)
                    prompt = prompt_qa['prompt']

                    try:
                        os.makedirs(model_subfolder_path, exist_ok=True) 
                    except OSError as e:
                         print(f"\nError creating model subfolder {model_subfolder_path}: {e}")
                         continue 

                    try:
                        with open(prompt_qa_path, "w") as f:
                             json.dump(prompt_qa, f, indent=4)
                    except Exception as e:
                         print(f"\nWarning: Could not save prompt_qa file to {prompt_qa_path}: {e}")
                    
                    if args.balance == 1:
                        graph = qa.get("context") 
                        features = analyze_graph_features(graph)
                        
                        model_path = model_path = os.path.join(os.path.dirname(__file__), "utils", "xgboost_Pangu.json")
                        prediction, probability = predict_with_xgboost(features, model_path=model_path)
                        threshold = 0.70
                        use_llm = prediction
                        if use_llm and probability > threshold:
                            use_api = 1
                            use_agent = 0
                            from .api.api import OpenAIAPI
                            api = OpenAIAPI(key=args.key)
                        else:
                            use_api = 0
                            use_agent = 1
                            from .agent_pangu.agent_manager import AgentManager
                            agent_manager = AgentManager(
                                model_name=model,
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
                        answer_content = agent_manager.get_response(
                            task=args.task,
                            prompt=prompt, 
                            agent_log_path=agent_log_path,
                            clear_history=True  
                        )
                        average_times += answer_content.get("time", 0)
                        average_tokens += float(answer_content.get("tokens", 0).get("total_tokens", 0))
                        print("📋 Agent Response Content:")
                        print(answer_content["content"])
                        if args.verbose and answer_content.get("intermediate_steps"):
                            print("🔧 Intermediate Steps:")
                            print(answer_content["intermediate_steps"]) 
                    
                    try:
                        with open(answer_path, "w") as f:
                             json.dump(answer_content, f) 
                    except Exception as e:
                         print(f"\nError saving answer file {answer_path}: {e}")

                except FileNotFoundError as e:
                    bar.set_postfix_str(f"Error: File not found - {e}")
                    print(f"\nSkipping {folder_name}: Required file not found - {e}")
                except json.JSONDecodeError as e:
                    failed_file = "unknown json"
                    if 'qa_file_path' in locals() and qa_file_path in str(e): failed_file = qa_file_path
                    bar.set_postfix_str(f"Error: JSON decode - {failed_file}")
                    print(f"\nSkipping {folder_name}: JSON decode error in {failed_file} - {e}")
                except OPENAI_RATE_LIMIT_ERROR as e:
                    bar.set_postfix_str("Error: Rate Limit")
                    print(f"\nOpenAI Rate Limit Error encountered: {e}. Waiting and will retry (if using try_all)...")
                    time.sleep(60) 
                except OPENAI_AUTH_ERROR as e:
                     bar.set_postfix_str("Error: Auth Failed")
                     print(f"\nOpenAI Authentication Error: {e}. Check API key.")
                     
                except OPENAI_CONNECTION_ERROR as e:
                     bar.set_postfix_str("Error: Connection")
                     print(f"\nOpenAI API Connection Error: {e}. Check network or server status. Waiting briefly...")
                     time.sleep(10) 
                except Exception as e:
                    bar.set_postfix_str(f"Error: {type(e).__name__}")
                    print(f"\nError processing {folder_name}: {type(e).__name__} - {e}")
                    
                
        average_filename = f"average{prompt_suffix}.json"
        average_path = os.path.join(task_folder, average_filename)            
        with open(average_path, "w") as f:
            json.dump({"average_times": average_times/len(instance_folders), "average_tokens": average_tokens/len(instance_folders)}, f)
        # print(f"Average times: {average_times/len(instance_folders)}, Average tokens: {average_tokens/len(instance_folders)}")

    def run(self, task_folder):
        """
        Executes LLM calls.
        If self.try_all is True, it will loop run_one until all tasks are completed.
        Otherwise, it only calls run_one once.

        Args:
            task_folder (str): The root directory for storing task data and results.
        """
        print(f"--- Starting LLM Run ---")
        print(f"Task: '{self.args.task}', Folder: {task_folder}")
        if self.try_all:
            while True:
                self.run_one(task_folder)
                print("\nChecking run status...")
                
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
        Evaluates model performance.
        """
        args = self.args
        model = args.model
        task = args.task
        prompt_suffix = _generate_prompt_suffix(args) # Suffix is needed to load correct answer file
        obj_task = load_task(task, args)


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


        metrics = []
        total_tokens = []
        prompt_tokens = []
        completion_tokens = []
        # Restore original error folder lists for simplicity as requested previously
        # temporal_wrong_folders = []
        # completion_wrong_folders = []
        wrong_folders = [] 
        fail_folders = [] 
        num_times = []
        num_edges = []
        num_nodes = []

        print(f"--- Evaluating Results ---")

        print(f"Task: {task}, Model: {model}, Suffix: {prompt_suffix}")

        for folder_name in tqdm(instance_folders, desc="Evaluating Instances"):
            folder_path = os.path.join(task_folder, folder_name)
            qa_file_path = os.path.join(folder_path, "qa.json")
            graph_path = os.path.join(folder_path, f"graph.json")
           
            model_subfolder_path = os.path.join(folder_path, model)
            answer_path = os.path.join(model_subfolder_path, f"answer_{model}{prompt_suffix}.json")
            with open(qa_file_path, "r") as f:
                qa = json.load(f)   
            with open(answer_path, "r") as f:
                answer = json.load(f) # answer is the dict with 'content', 'total_tokens' etc.
            with open(graph_path, "r") as f:
                graph = json.load(f)

          

            if args.task == "modify_dyg":
                metric, extracted_answer = obj_task.evaluate(qa, answer["content"], args.w[0])
            else:
                try:
                    metric, extracted_answer = obj_task.evaluate(qa, answer["content"], use_agent=args.use_agent)
                except TypeError:
                    metric, extracted_answer = obj_task.evaluate(qa, answer["content"])
            metrics.append(metric)
            
           
            if metric < 0:
                fail_folders.append(folder_name) 
            elif metric == 0 or metric == 2:
               wrong_folders.append(folder_name) 
           
            total_tokens.append(answer.get('total_tokens', 0))
            prompt_tokens.append(answer.get('prompt_tokens', 0))
            completion_tokens.append(answer.get("completion_tokens", 0))
            num_times.append(graph.get('num_time', 0))
            num_edges.append(graph.get('num_edges', 0))
            num_nodes.append(graph.get('num_nodes', 0))
       
        num_all = len(metrics)
        if num_all == 0:
            print("Error: Failed to evaluate any instances.")
            return
        print(num_all)
        print(metrics)
        right_rate = sum(metric for metric in metrics if metric > 0 and metric <= 1) / num_all if num_all > 0 else 0
        # Error rate: metric = 0, -1, -2 (Logic error, Sort error, Integrity error)
        wrong_rate = sum(1 for m in metrics if m == 0 or m == 2) / num_all if num_all > 0 else 0
       # Failure rate: metric = -3 (Parsing error, File not found, JSON error, Unexpected evaluation error)
        fail_rate = sum(1 for m in metrics if m == -1) / num_all if num_all > 0 else 0

        total_tokens_sum = sum(t for t in total_tokens if t is not None) 
        valid_token_counts = len([t for t in total_tokens if t is not None])
        average_tokens_calc = total_tokens_sum / valid_token_counts if valid_token_counts > 0 else 0

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
            "error_folders": wrong_folders, # Combines logic errors and integrity errors
            "failure_folders": fail_folders, # Lists failed folders separately
            # "temporal_wrong_folders": temporal_wrong_folders,
            # "completion_wrong_folders": completion_wrong_folders,
        }

        results_filename = f"results_{args.motif_name}_{model}{prompt_suffix}.json"
        results_save_path = os.path.join(task_folder, results_filename)
        try:
             with open(results_save_path, "w") as f:
                 json.dump(results, f, indent=4)
             print(f"\nEvaluation results saved to: {results_save_path}")
        except Exception as e:
            print(f"\nError saving evaluation results to {results_save_path}: {e}")

        print(f"\n--- Evaluation Summary ---")
        print(f"Task: {task}, Model: {model}, Suffix: {prompt_suffix}") # 
        print(f"Correct Rate: {right_rate:.4f}, Error Rate: {wrong_rate:.4f}, Failure Rate: {(1-right_rate-wrong_rate):.4f}")
        print(f"Average Tokens: {average_tokens_calc:.2f}, Total Tokens: {total_tokens_sum}")

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
        Displays the results for a single task. (Loads the result file with suffix)
        """
        args = self.args
        task = args.task
        task_folder = args.task_folder
        model = args.model
        prompt_suffix = _generate_prompt_suffix(args)
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

        print(f"\n--- Results Summary (from {os.path.basename(results_file)}) ---")
        print(f"Task: {results_data.get('task', 'N/A')}")
        print(f"Model: {results_data.get('model', 'N/A')}")
        print(f"Suffix: {results_data.get('prompt_config_suffix', 'N/A')}") 
        print(f"Average Accuracy (metric>0): {results_data.get('average_acc', 'N/A'):.4f}")
        print(f"Wrong Rate (metric<0): {results_data.get('wrong_rate', 'N/A'):.2f}")
        # print(f"Average Tokens: {results_data.get('average_tokens', 'N/A'):.2f}")
        # print(f"Total Tokens: {results_data.get('total_tokens', 'N/A')}")
        # print(f"Temporal Errors (metric=-1): {len(results_data.get('temporal_wrong_folders', []))}")
        # print(f"Completion Errors (metric=-2): {len(results_data.get('completion_wrong_folders', []))}")

        # print("\nNote: Detailed breakdown by T, N, p requires adapting the 'show' method or storing more data in results.")


    def analyze_data(self, task_folder):
        """
        Analyzes the distribution of timestamps, node counts, and edge counts
        in the generated data, and saves the statistics and visualizations.
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
        for folder_name in tqdm(folders_to_analyze, desc="Analyzing Graphs"): 
            graph_path = os.path.join(task_folder, folder_name, "graph.json")
            try:
                with open(graph_path, "r") as f:
                    graph_data = json.load(f)

                node_count = graph_data.get('N', None)
                edge_count = graph_data.get('num_edges', None)
                timestamps_in_file = [int(edge[2]) for edge in graph_data.get('edge_index', []) if len(edge) > 2] 

                if node_count is not None and edge_count is not None:
                    all_node_counts.append(node_count)
                    all_edge_counts.append(edge_count)
                    all_timestamps.extend(timestamps_in_file) 
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
        valid_timestamps = [t for t in all_timestamps if t is not None] 

        if not valid_nodes or not valid_edges or not valid_timestamps:
            print(f"Error: Failed to collect sufficient graph attribute data (requires nodes, edges, and timestamps) from any valid instances. Valid instances count: {valid_instances}")
            return

        print(f"Analysis complete. Data collected from {valid_instances}/{len(instance_folders)} valid instances.")
        print(f"Node counts: {len(valid_nodes)}, Edge counts: {len(valid_edges)}, Timestamp events: {len(valid_timestamps)}")

 
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
            print(f"Graph attribute statistics saved to: {stats_save_path}")

            fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 

            # Subplot 1: Node Count Distribution
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


            # Subplot 2: Edge Count Distribution
            sns.histplot(valid_edges, ax=axes[1], kde=True) # Using KDE shows a smoother distribution
            axes[1].set_title('Edge Count Distribution')
            axes[1].set_xlabel('Number of Edges')
            axes[1].set_ylabel('Frequency / Density')

            # Subplot 3: Timestamp Distribution (retaining original histogram logic)
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
            plot_filename = os.path.join(vis_dir, "graph_analysis_distributions.png")
            plt.savefig(plot_filename, dpi=150)
            print(f"Combined graph attribute distribution plot saved to: {plot_filename}")


        except Exception as e:
            print(f"Error: An error occurred during statistics processing or plotting: {e}")
            import traceback
            traceback.print_exc() 
        
    def execute(self, dir):
        """Executes the corresponding action based on args.t."""
        args = self.args
        task_folder = args.task_folder
        if args.dataset != "random":
            task_folder = os.path.join(task_folder, args.dataset)
        if args.task == "judge_motif" or args.task == "modify_dyg" or args.task == "judge_contain_motif":
            task_folder = os.path.join(task_folder, args.motif_name)
        # if args.task == "judge_contain_motif":    
        #     task_folder = os.path.join(task_folder, f"M{args.p}")
        if args.api == 1:
            task_folder = os.path.join(task_folder, "api")
        if args.use_agent == 1:
            task_folder = os.path.join(task_folder, "agent")
        if args.balance == 1:
            task_folder = os.path.join(task_folder, "balance")
        print(f"\nExecuting action '{args.t}' for task '{args.task}' in folder '{task_folder}'")
        if args.t == "clear":
            remove_dir(task_folder)
        elif args.t == "gen":
            self.gen(task_folder) 
        elif args.t == "run":
            self.run(task_folder)
        elif args.t == "eval":
            self.evaluate(task_folder)
        elif args.t == "check":
            self.check(task_folder)
        elif args.t == "show":
            self.show(task_folder)
        elif args.t == "analyze":
            print(f"\n--- Explicitly running Data Analysis ---")
            self.analyze_data(task_folder)
        else:
            print(f"Error: Action '{args.t}' not implemented.")
            raise NotImplementedError

