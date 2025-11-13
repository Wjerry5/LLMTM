import os
from .utils import  remove_dir, load_task
import json
from openai import OpenAI
# from .utils import send_prompt, DyGraphPrompt, DyGraphGenERCon, load_task # 确保 load_task 等导入
from .utils import DyGraphGenControlMotif,DyGraphGenEdge, DyGraphPrompt, DyGraphGenERCon, DyGraphGenMotifCon,load_task, DyGraphGenMorMotifCon # 确保 load_task 等导入
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import openai # 导入 openai 以便捕获其特定错误
import torch # 导入 torch
from transformers import AutoModelForCausalLM, AutoTokenizer # 导入 transformers 类
from collections import defaultdict, Counter # 导入 Counter 用于频数统计
import langchain
from langchain.agents import initialize_agent, AgentType   
# --- 添加可视化导入 ---
import networkx as nx
from scipy.stats import entropy
from collections import Counter, defaultdict
try:
    from .utils import visualization
except ImportError:
    print("警告: LLMDyG_Motif.utils.visualization 未找到或导入失败。可视化功能将不可用。")
    visualization = None
# --- 添加绘图导入 ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plotting_available = True
except ImportError:
    print("警告: matplotlib 或 seaborn 未安装。数据分析绘图功能将不可用。")
    plotting_available = False
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# --- 修复OpenAI异常导入兼容性 ---
try:
    # 新版本OpenAI库 (v1.0+)
    from openai import RateLimitError, AuthenticationError, APIConnectionError
    OPENAI_RATE_LIMIT_ERROR = RateLimitError
    OPENAI_AUTH_ERROR = AuthenticationError
    OPENAI_CONNECTION_ERROR = APIConnectionError
except ImportError:
    try:
        # 旧版本OpenAI库 (v0.x)
        from openai.error import RateLimitError, AuthenticationError, APIConnectionError
        OPENAI_RATE_LIMIT_ERROR = RateLimitError
        OPENAI_AUTH_ERROR = AuthenticationError
        OPENAI_CONNECTION_ERROR = APIConnectionError
    except ImportError:
        # 如果都导入失败，定义兼容的异常类
        class FallbackOpenAIError(Exception):
            pass
        OPENAI_RATE_LIMIT_ERROR = FallbackOpenAIError
        OPENAI_AUTH_ERROR = FallbackOpenAIError
        OPENAI_CONNECTION_ERROR = FallbackOpenAIError
        print("警告: 无法导入OpenAI异常类，使用兼容模式")

import sys
import torch
import transformers
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    # print("cuDNN Version:", torch.backends.cudnn.version()) # 可能需要 cudnn

PREDEFINED_MOTIFS = {
    "2-star":       [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a')],                             # k=3, l=2
    "triangle":     [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u0', 't2', 'a')],             # k=3, l=3
    "3-star":       [('u0', 'u1', 't0', 'a'), ('u0', 'u2', 't1', 'a'), ('u0', 'u3', 't2', 'a')],             # k=4, l=3 (中心节点为 0)
    "4-path":       [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a')],             # k=4, l=3
    "4-cycle":      [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u0', 't3', 'a')], # k=4, l=4
    "4-tailedtriangle": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a')], # k=4, l=4
    "butterfly":    [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u2', 't2', 'a'), ('u2', 'u0', 't3', 'a')], # k=4, l=4 (与 4-cycle 拓扑相同，时序不同？根据图像解析有歧义，暂不包含)
    "4-chordalcycle": [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u2', 'u3', 't2', 'a'), ('u3', 'u1', 't3', 'a'), ('u3', 'u0', 't4', 'a')], # k=4, l=5
    "4-clique":     [('u0', 'u1', 't0', 'a'), ('u1', 'u2', 't1', 'a'), ('u1', 'u3', 't2', 'a'), ('u2', 'u3', 't3', 'a'), ('u3', 'u0', 't4', 'a'), ('u0', 'u2', 't5', 'a')], # k=4, l=6
    "bitriangle":   [('u0', 'u1', 't0', 'a'), ('u1', 'u3', 't1', 'a'), ('u3', 'u5', 't2', 'a'), ('u5', 'u4', 't3', 'a'), ('u4', 'u2', 't4', 'a'), ('u2', 'u0', 't5', 'a')], # k=6, l=6
}


MODEL_PATHS = {
    "Qwen_14B": "LLMDyG_Motif/DeepSeek-R1-Distill-Qwen-14B",
    "Llama_8B": "LLMDyG_Motif/Llama-3.1-Nemotron-Nano-8B-v1",
    "Deepseek_7B": "LLMDyG_Motif/DeepSeek-R1-Distill-Qwen-7B",
    "Deepseek_7B_chat": "/LLMDyG_Motif/llama_2_7B_chat",
    "Llama_7B_chat": "/LLMDyG_Motif/llama_2_7B_chat",
    "Llama_13B_chat": "/LLMDyG_Motif/llama_2_13B_chat",
    "Qwen-32B-Chat": "/LLMDyG_Motif/Qwen1.5-32B-Chat-AWQ",
    "gpt-4o-mini": "",
    "deepseek-r1-250528": "",
    "QwQ": "/LLMDyG_Motif/QwQ-32B",
    "Qwen2.5_32B": "/LLMDyG_Motif/Qwen2.5-32B-Instruct",
    "Qwen_32B": "/LLMDyG_Motif/DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-32B":"",
    "DeepSeek-R1-Distill-Qwen-14B":"",
    "o3-2025-04-16": "",
    "pangu_auto": "/home/ma-user/work/LLMDyG_Motif/Pangu",
}

def initialize_model(model):
    """在主进程中初始化模型和分词器"""
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

def chat_with_model(model, prompt, temperature, max_tokens):
    """
    与加载的模型进行交互。
    """
    # 将 prompt 包装成 messages 格式 (如果模型需要)
    # 注意：Qwen 可能不需要这种包装，可以直接用 prompt 字符串。
    # 但为了与 send_prompt 保持某种程度的一致性，这里保留 messages 结构，
    # 并在 tokenizer 调用时直接使用 prompt。
    # messages = [{"role": "user", "content": prompt}]

    torch.cuda.empty_cache() # 清理缓存

    # inputs = tokenizer(
    #     prompt,
    #     return_tensors="pt",
    # )
    # prompt_tokens = inputs.input_ids.shape[1]


    try:
        debug_prompt_file = "prompt_debug_runner.txt"
        with open(debug_prompt_file, "w", encoding="utf-8") as f_debug:
            f_debug.write(prompt)
        print(f"[调试信息] Prompt 已保存到: {debug_prompt_file}")
    except Exception as e_debug:
        print(f"[调试信息] 保存 prompt 到文件时出错: {e_debug}")

    # with torch.no_grad():
    #     # 确保 inputs 在正确的设备上
    #     device_inputs = inputs.to(model.device)
    #     outputs = model.generate(
    #         **device_inputs,
    #         max_new_tokens=max_tokens,
    #         temperature=temperature,
    #         do_sample=False, # 如果 temperature > 0 才进行采样
    #         pad_token_id=tokenizer.eos_token_id, # <--- 添加显式设置
    #         repetition_penalty=1.0           # <--- 添加显式设置
    #     )
    # completion_token_ids = outputs[0][prompt_tokens:]
    # response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(response_text)
    # completion_tokens = len(completion_token_ids)
    # total_tokens = prompt_tokens + completion_tokens

    # # 清理 GPU 内存
    # del inputs, outputs, completion_token_ids, device_inputs # 清理 device_inputs
    # torch.cuda.empty_cache()
    if model == "pangu_auto":
        client = OpenAI(
            api_key="sk-xxx",  # 任意字符串即可
            base_url="http://127.0.0.1:8888/v1", 
            default_headers={
                "Connection": "close",  # 避免保持连接
                "Keep-Alive": "timeout=0"  # 禁用keep-alive
            }
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
        # do_sample, pad_token_id, repetition_penalty 都是本地模型的参数，在API中不支持
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


# 辅助函数：根据 args 生成唯一的 prompt 配置 ID
def _generate_prompt_suffix(args):
    """Generates a unique suffix based on prompt configuration arguments."""
    # 使用 args.__dict__.get 提供默认值，避免 args 中缺少某些属性时出错
    k = args.__dict__.get('k', args.__dict__.get('num_examplars', 0)) # 兼容 k 和 num_examplars
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
            

# 定义 Runner 类，负责管理和执行整个实验流程，包括：
# 1. 生成数据和问题 (gen) - 只生成 graph.json 和 qa.json
# 2. 调用 LLM 获取答案 (run) - 动态生成 prompt，保存带后缀的 answer
# 3. 评估模型性能 (evaluate) - 加载带后缀的 answer，保存带后缀的 results
# 4. 检查运行状态 (check) - 检查带后缀的 answer 文件
# 5. 汇总和展示结果 (show)
class Runner:
    """
    管理 LLM 在动态图任务上评估的整个流程，包括：
    1. 生成数据和问题 (gen) - 只生成 graph.json 和 qa.json
    2. 调用 LLM 获取答案 (run) - 动态生成 prompt，保存带后缀的 answer
    3. 评估模型性能 (evaluate) - 加载带后缀的 answer，保存带后缀的 results
    4. 检查运行状态 (check) - 检查带后缀的 answer 文件
    5. 汇总和展示结果 (show)
    """
    def __init__(self, args, try_all = False) -> None:
        """
        初始化 Runner。

        Args:
            args (object): 包含所有配置参数的对象 (通常来自 argparse)。
            try_all (bool, optional): 是否在 run 模式下持续尝试运行，直到所有任务完成。
                                      默认为 False。
        """
        self.args = args
        self.try_all = try_all
        
    def check(self, task_folder):
        """
        检查指定任务文件夹中特定配置的运行状态。

        Args:
            task_folder (str): 任务数据和结果存储的根目录。

        Returns:
            int: 待运行 (torun) 的实例数量。
        """
        args = self.args
        model = args.model
        prompt_suffix = _generate_prompt_suffix(args)

        # --- 修改: 加载 instance_list.json ---
        instance_list_path = os.path.join(task_folder, "instance_list.json")
        try:
            with open(instance_list_path, "r") as f:
                 instance_info = json.load(f)
            instance_folders = instance_info["instance_list"] # 使用新 key
            print(instance_folders)
        except FileNotFoundError:
             print(f"Error: instance_list.json not found in {task_folder}. Please run '-t gen' first.")
             return -1
        except json.JSONDecodeError:
             print(f"Error: instance_list.json in {task_folder} is corrupted.")
             return -1
        # --- 修改结束 ---

        finish = []
        torun = []
        sdict = {"num_edges": [], "num_nodes": [], "num_time": []}

        # --- 修改: 使用 instance_folders ---
        for i, folder_name in enumerate(instance_folders):
            folder_path = os.path.join(task_folder, folder_name)
            # 加载该实例的图信息
            try:
                graph = json.load(open(os.path.join(folder_path, "graph.json")))
                # 收集统计数据
                for k, v in sdict.items():
                    v.append(graph.get(k, 0)) # 使用 get 以防某些 key 不存在
            except FileNotFoundError:
                print(f"Warning: graph.json not found in {folder_path}")
                torun.append(i) # 如果图文件不存在，也视为待运行
                continue
            except json.JSONDecodeError:
                 print(f"Warning: graph.json in {folder_path} is corrupted.")
                 torun.append(i) # 如果图文件损坏，也视为待运行
                 continue

            # --- 修改: answer 文件现在在模型子目录中 ---
            model_subfolder_path = os.path.join(folder_path, model)
            answer_path = os.path.join(model_subfolder_path, f"answer_{model}{prompt_suffix}.json")
            # --- 修改结束 ---

            if os.path.exists(answer_path):
                try:
                    # 可选：检查答案文件是否有效
                    json.load(open(answer_path, "r"))
                    finish.append(i)
                except json.JSONDecodeError:
                    print(f"Warning: Answer file {answer_path} is corrupted.")
                    torun.append(i) # 答案文件损坏，视为待运行
            else:
                torun.append(i)
                
        print(f"--- Checking Status ---")
        print(f"Task Folder: {task_folder}")
        print(f"Model: {model}")
        print(f"Prompt Suffix: {prompt_suffix}")
        print(f"Finish {len(finish)}, ToRun {len(torun)} (Total {len(instance_folders)})")
        if sdict["num_edges"]: # 仅在收集到数据时打印
             print("Graph Stats (Avg±Std): " + "".join(f"{k}:{np.mean(v):.2f}±{np.std(v):.2f} " for k,v in sdict.items() if v))
        else:
             print("Graph Stats: No valid graph data found to compute statistics.")
        return len(torun)
        
    
    def generate_random(self, dir, T, N, p, seed, *targs ,**kwargs):
        """
        为单个实例生成动态图数据 (`graph.json`)、问答对 (`qa.json`)，
        并将它们保存到指定的子目录中。**不再保存 prompt_qa.json**。

        Args:
            dir (str): 保存该实例数据的父目录。
            T (int): 动态图的时间步数。
            N (int): 动态图的节点数。
            p (float): 动态图生成模型的相关概率参数。
            seed (int): 用于生成该实例的随机种子。
            *targs: 传递给 obj_task.generate_qa 的额外位置参数。
            **kwargs: 传递给 obj_task.generate_qa 的额外关键字参数 (例如 label)。

        Returns:
            str: 生成的实例的文件夹名称 (格式如 "T_N_p_seed")。
                 如果生成失败则返回 None。
        """
        args = self.args
        task = args.task

        if args.dataset != "random": # real-world dataset
            folder_setting = f"dyg{seed}"
        else:
            folder_setting = f"T{T}_N{N}_p{p}_seed{seed}"
        
        
        info = None
        if args.dataset != "random": # real-world dataset
            infos = json.load(open(f"/home/hb/LLMDyG_Motif/LLMDyG_Motif/dataset/{args.dataset}/{args.dataset}_subgraphs.json"))
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
                print(f"警告 (generate_random): 数据生成失败 (DyGraphGenMorMotifCon) for task '{task}', T={T}, N={N}, M={p}, W={args.w[0] if isinstance(args.w, list) else args.w}, seed={seed}. 跳过此实例.")
                return None # Return None to indicate failure for this seed
        else:
            dygen = DyGraphGenERCon()
            info = dygen.sample_dynamic_graph(T = T, N = N , p = p, seed = seed)
        obj_task = load_task(task, args)
        dygprompt = DyGraphPrompt(obj_task, args = args)
        
  
        # 2. 根据图信息和任务生成问答对
        if info is None: # General check, covers cases if info is not set in other branches or if the specific check was missed
            print(f"警告 (generate_random): 'info' is None before QA generation for task '{task}', T={T}, N={N}, p/M={p}, seed={seed}. 跳过此实例.")
            return None
            
        qa = obj_task.generate_qa(info, *targs, **kwargs)

        # 如果 generate_qa 返回 None (例如没有有效数据生成 QA)，则直接返回
        if qa is None:
             print(f"警告 (generate_random): 任务 {task} 未能为 T={T}, N={N}, p={p}, seed={seed} 生成有效的 QA 数据。跳过此实例。")
             return None

        # --- 保存文件 (恢复 try-except) ---
        instance_folder = os.path.join(dir, folder_setting) # 实例数据文件夹路径
        os.makedirs(instance_folder, exist_ok=True) # 创建实例文件夹
        info_file = os.path.join(instance_folder, f"graph.json") # 图信息文件路径
        qa_file = os.path.join(instance_folder, f"qa.json") # 问答对文件路径
        
        # write files
        json.dump(info, open(info_file, "w"))

  
        if isinstance(qa.get("_original_context_set"), set):
            # 假设集合元素是元组，可以直接转为列表
            qa["_original_context_set"] = list(qa["_original_context_set"])
        if isinstance(qa.get("_final_edges_set"), set):
             # 假设集合元素是元组，可以直接转为列表
             qa["_final_edges_set"] = list(qa["_final_edges_set"])


        json.dump(qa, open(qa_file, "w"))

        # --- 调用可视化 (保持不变，但包含在 generate_random 内) ---
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
                    # print(f"  正在生成可视化 for {folder_setting}...") # 可以取消注释以获取详细日志
                    visualization.visualize_graph(num_nodes, temporal_edges, snapshot_path)
                    visualization.create_colored_animation(num_nodes, temporal_edges, gif_path)
                else:
                    print(f"警告 (generate_random): 缺少可视化所需数据 (edge_index or N) for {folder_setting}")

            except Exception as e: # 保持对可视化错误的捕获
                print(f"错误 (generate_random): 调用可视化函数时出错 for {folder_setting}: {e}")
                import traceback
                traceback.print_exc()

        return folder_setting

    

    # run
    def gen(self, dir):
        """
        生成指定任务的所有问题实例数据 (`graph.json`, `qa.json`) 及可视化。
        生成完成后，自动进行时间戳数据分析。
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
                instance_list.append(folder_setting) # 使用新列表名
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
                                    print("folder_setting为空")
                                seed += 1
                                attempts += 1

                            if successful_seeds_for_combo < args.num_seed:
                                print(f"\nWarning: Generated {successful_seeds_for_combo}/{args.num_seed} instances for T={T_val}, N={N_val}, p={p_val} after {attempts} attempts.")

        instance_list_path = os.path.join(dir, f"instance_list.json")
        try:
            # 使用 key "instance_list"
            with open(instance_list_path, "w") as f:
                json.dump({"instance_list": instance_list}, f)
            print(f"\nSuccessfully generated data for {len(instance_list)} instances.")
            print(f"instance_list.json saved in {dir}")
            if visualization:
                 print(f"Visualizations saved in {os.path.join(dir, 'visualizations')}")

            # --- 添加分析步骤 (依赖 instance_list) ---
            if instance_list:
                 print("\n--- Starting Post-Generation Data Analysis ---")
                 self.analyze_data(dir)
            else:
                 print("\nSkipping data analysis as no instances were successfully generated.")
            # --- 分析步骤结束 ---

        except Exception as e:
             print(f"Error saving instance_list.json: {e}")
             print("\nSkipping data analysis due to error saving instance_list.json.")
        # --- 修改结束 ---
    
    def run_one(self, task_folder):
        """
        运行单个批次的 LLM 调用，处理所有实例问题。
        
        批量处理独立性保证：
        1. Agent初始化时设置 disable_memory=True 禁用内存功能
        2. 每次调用 get_response 时设置 clear_history=True 清空历史
        3. 确保100个问题之间完全没有记忆关联
        
        将 prompt_qa 和 answer 文件都保存到模型子目录。
        """
        args = self.args
        model = args.model
        use_agent = args.use_agent
        use_api = args.api
        
        if args.balance == 1:
            from .Balance.agent_manager import AgentManager
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
        elif use_api == 1:
            from .api.api import OpenAIAPI
            api = OpenAIAPI(key=args.key)
        elif use_agent == 1:
            if args.model == "pangu_auto":
                from .agent_pangu.agent_manager import AgentManager
            else:
                from .agent_4o.agent_manager import AgentManager

            agent_manager = AgentManager(
                key=args.key,
                model_name=model,
                temperature=0.1,
                max_new_tokens=10240,
                memory_k=5,
                verbose=True,
                max_iterations=5,  # Reduce maximum iterations to force Agent to finish quickly
                handle_parsing_errors=True,
                disable_memory=False
            )

            print("🤖 初始化Agent Manager用于批量处理...")
            
        # else:
            # model_obj, tokenizer = initialize_model(model)
        try:
            obj_task = load_task(args.task, args)
            dygprompt = DyGraphPrompt(obj_task, args=args)
        except Exception as e:
            print(f"Error initializing task or prompt generator: {e}")
            return

        prompt_suffix = _generate_prompt_suffix(args)

        # --- 加载 instance_list.json ---
        instance_list_path = os.path.join(task_folder, "instance_list.json")
        try:
            with open(instance_list_path, "r") as f:
                 instance_info = json.load(f)
            instance_folders = instance_info["instance_list"] # 使用新 key
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

                    # --- 修改: 定义 prompt_qa 和 answer 的路径 (都在模型子目录) ---
                    model_subfolder_path = os.path.join(folder_path, model)
                    prompt_qa_filename = f"prompt_qa{prompt_suffix}.json"
                    prompt_qa_path = os.path.join(model_subfolder_path, prompt_qa_filename)

                    answer_filename = f"answer_{model}{prompt_suffix}.json"
                    answer_path = os.path.join(model_subfolder_path, answer_filename) # <--- 修改 answer 路径
                    
                    agent_filename = f"agent_{prompt_suffix}.log"
                    agent_log_path = os.path.join(model_subfolder_path, agent_filename)

                    balance_agent_filename = f"balance_agent_{prompt_suffix}.log"
                    balance_agent_log_path = os.path.join(model_subfolder_path, balance_agent_filename)
                    
                    api_filename = f"api_{prompt_suffix}.log"
                    api_log_path = os.path.join(model_subfolder_path, api_filename)


                    # 检查 answer 文件是否存在
                    if os.path.exists(answer_path):
                        bar.set_postfix_str("Skipped (Answer Exists)")
                        continue

                    # 加载 qa 数据
                    try:
                        with open(qa_file_path, "r") as f:
                            qa = json.load(f)
                    except FileNotFoundError:
                         print(f"\nSkipping {folder_name}: qa.json not found.")
                         continue
                    except json.JSONDecodeError:
                         print(f"\nSkipping {folder_name}: qa.json is corrupted.")
                         continue

                    # 动态生成 prompt_qa 字典
                    prompt_qa = dygprompt.generate_prompt_qa(**qa)
                    prompt = prompt_qa['prompt']
                    # prompt = prompt.encode('utf-8').decode('unicode_escape')
                    # --- 修改: 保存 prompt_qa 和 answer 前确保模型子目录存在 ---
                    try:
                        os.makedirs(model_subfolder_path, exist_ok=True) # 创建模型子目录
                    except OSError as e:
                         print(f"\nError creating model subfolder {model_subfolder_path}: {e}")
                         continue # 无法保存，跳过此实例

                    # 保存 prompt_qa
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
                            from .api.api import OpenAIAPI
                            api = OpenAIAPI(key=args.key)
                        else:
                            use_api = 0
                            use_agent = 1
                            from .agent_pangu.agent_manager import AgentManager
                            agent_manager = AgentManager(
                                key=args.key,
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
                        # 🔑 批量处理：每个问题都是独立的，无记忆关联
                        # get_response方法会自动清空任何残留的记忆状态
                        print(f"🔄 处理独立问题: {folder_name}")
                        answer_content = agent_manager.get_response(
                            task=args.task,
                            prompt=prompt, 
                            agent_log_path=agent_log_path,
                            clear_history=True  # 明确指定清空历史，确保问题独立
                        )
                        average_times += answer_content.get("time", 0)
                        average_tokens += float(answer_content.get("tokens", 0).get("total_tokens", 0))
                        print("📋 Agent响应内容:")
                        print(answer_content["content"])
                        if args.verbose and answer_content.get("intermediate_steps"):
                            print("🔧 执行步骤:")
                            print(answer_content["intermediate_steps"]) 
                    # else:
                    #     # 调用 LLM
                    #     token_budget = con.get_token()
                    #     bar.set_postfix_str(f"Tokens Left: {int(token_budget)}")
                    #     # answer_content = chat_with_model(model_obj, tokenizer, prompt, temperature=args.temperature, max_tokens=args.max_tokens)
                    #     answer_content = chat_with_model(model, prompt, temperature=args.temperature, max_tokens=args.max_tokens)
                    #     con.time_token()
                    #     tokens_used = answer_content.get('total_tokens', 0)
                    #     con.use_token(tokens_used)
                    #     bar.set_postfix_str(f"Tokens Used: {tokens_used}")

                    # 保存答案 (到模型子目录)
                    try:
                        with open(answer_path, "w") as f:
                             json.dump(answer_content, f) # 保存包含 content 和 token 的完整字典
                    except Exception as e:
                         print(f"\nError saving answer file {answer_path}: {e}")

                except FileNotFoundError as e:
                    bar.set_postfix_str(f"Error: File not found - {e}")
                    print(f"\nSkipping {folder_name}: Required file not found - {e}")
                except json.JSONDecodeError as e:
                    # 定位具体哪个文件解析失败
                    failed_file = "unknown json"
                    if 'qa_file_path' in locals() and qa_file_path in str(e): failed_file = qa_file_path
                    bar.set_postfix_str(f"Error: JSON decode - {failed_file}")
                    print(f"\nSkipping {folder_name}: JSON decode error in {failed_file} - {e}")
                except OPENAI_RATE_LIMIT_ERROR as e:
                    bar.set_postfix_str("Error: Rate Limit")
                    print(f"\nOpenAI Rate Limit Error encountered: {e}. Waiting and will retry (if using try_all)...")
                    time.sleep(60) # 等待 60 秒
                    # 依赖外层 try_all 循环来重试当前实例
                    # 注意：如果不用 try_all，这里会导致实例被跳过
                except OPENAI_AUTH_ERROR as e:
                     bar.set_postfix_str("Error: Auth Failed")
                     print(f"\nOpenAI Authentication Error: {e}. Check API key.")
                     # 这种错误通常无法通过重试解决，可能需要停止
                     # break # 可以选择跳出循环
                except OPENAI_CONNECTION_ERROR as e:
                     bar.set_postfix_str("Error: Connection")
                     print(f"\nOpenAI API Connection Error: {e}. Check network or server status. Waiting briefly...")
                     time.sleep(10) # 短暂等待后可能恢复
                     # 依赖外层 try_all 重试
                except Exception as e:
                    bar.set_postfix_str(f"Error: {type(e).__name__}")
                    print(f"\nError processing {folder_name}: {type(e).__name__} - {e}")
                    # 可以在这里添加更详细的错误处理或日志记录
                
        average_filename = f"average{prompt_suffix}.json"
        average_path = os.path.join(task_folder, average_filename)            
        with open(average_path, "w") as f:
            json.dump({"average_times": average_times/len(instance_folders), "average_tokens": average_tokens/len(instance_folders)}, f)
        print(f"Average times: {average_times/len(instance_folders)}, Average tokens: {average_tokens/len(instance_folders)}")

    def run(self, task_folder):
        """
        执行 LLM 调用。
        如果 self.try_all 为 True，则会循环调用 run_one 直到所有任务完成。
        否则只调用一次 run_one。

        Args:
            task_folder (str): 任务数据和结果存储的根目录。
        """
        print(f"--- Starting LLM Run ---")
        print(f"Task: '{self.args.task}', Folder: {task_folder}")
        if self.try_all:
            while True:
                self.run_one(task_folder)
                print("\nChecking run status...")
                # 检查特定配置的完成情况
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
        评估模型性能。
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
        wrong_folders = [] # 添加用于记录 metric = 0 (错误) 的列表
        fail_folders = [] # 添加用于记录 metric = -1 (失败/解析错误) 的列表
        num_times = []
        num_edges = []
        num_nodes = []

        print(f"--- Evaluating Results ---")
        # --- 修改: 使用后缀打印 ---
        print(f"Task: {task}, Model: {model}, Suffix: {prompt_suffix}")
        # instance_folders = instance_folders[:68]
        # --- 修改: 使用 instance_folders ---
        for folder_name in tqdm(instance_folders, desc="Evaluating Instances"):
            folder_path = os.path.join(task_folder, folder_name)
            qa_file_path = os.path.join(folder_path, "qa.json")
            graph_path = os.path.join(folder_path, f"graph.json")
            # --- 修改: 从模型子目录加载 answer 文件 ---
            model_subfolder_path = os.path.join(folder_path, model)
            answer_path = os.path.join(model_subfolder_path, f"answer_{model}{prompt_suffix}.json")
            with open(qa_file_path, "r") as f:
                qa = json.load(f)   
            with open(answer_path, "r") as f:
                answer = json.load(f) # answer is the dict with 'content', 'total_tokens' etc.
            with open(graph_path, "r") as f:
                graph = json.load(f)

            # 现在 qa 中的集合已经被恢复，可以安全传递给 obj_task.evaluate

            if args.task == "modify_dyg":
                metric, extracted_answer = obj_task.evaluate(qa, answer["content"], args.w[0])
            else:
                # 安全地传递use_agent参数，避免不支持的任务类出错
                try:
                    metric, extracted_answer = obj_task.evaluate(qa, answer["content"], use_agent=args.use_agent)
                except TypeError:
                    # 如果任务类不支持use_agent参数，则不传递该参数
                    metric, extracted_answer = obj_task.evaluate(qa, answer["content"])
            metrics.append(metric)
            
            # --- 恢复: 原始错误分类 (包含 fail_folders) ---
            if metric < 0:
                fail_folders.append(folder_name) # 解析/格式错误
            elif metric == 0 or metric == 2:
               wrong_folders.append(folder_name) # 回复错误
            # --- 恢复: 记录 token 和 stats ---
            total_tokens.append(answer.get('total_tokens', 0))
            prompt_tokens.append(answer.get('prompt_tokens', 0))
            completion_tokens.append(answer.get("completion_tokens", 0))
            num_times.append(graph.get('num_time', 0))
            num_edges.append(graph.get('num_edges', 0))
            num_nodes.append(graph.get('num_nodes', 0))
        # --- 恢复: 原始统计计算 (修改为包含 fail_rate) ---
        num_all = len(metrics)
        if num_all == 0:
            print("错误：未能成功评估任何实例。")
            return
        print(num_all)
        print(metrics)
        right_rate = sum(metric for metric in metrics if metric > 0 and metric <= 1) / num_all if num_all > 0 else 0
        # 错误率：metric = 0, -1, -2 (逻辑错误、排序错误、完整性错误)
        wrong_rate = sum(1 for m in metrics if m == 0 or m == 2) / num_all if num_all > 0 else 0
        # 失败率：metric = -3 (解析错误、文件未找到、JSON错误、评估中意外错误)
        fail_rate = sum(1 for m in metrics if m == -1) / num_all if num_all > 0 else 0

        total_tokens_sum = sum(t for t in total_tokens if t is not None) # 确保加总时忽略 None
        valid_token_counts = len([t for t in total_tokens if t is not None])
        average_tokens_calc = total_tokens_sum / valid_token_counts if valid_token_counts > 0 else 0

        # --- 恢复: 原始 results 结构 (更新错误/失败率) ---
        results = {
            "task": task,
            "model": model,
            "prompt_config_suffix": prompt_suffix,
            "correct_rate": right_rate, # 正确率 (metric=1)
            "error_rate": wrong_rate,   # 错误率 (metric=0, -1, -2)
            "failure_rate": fail_rate,  # 失败率 (metric=-3)
            "average_tokens": average_tokens_calc,
            "total_tokens": total_tokens_sum,
            "average_num_times": np.mean(num_times) if num_times else 0,
            "average_num_edges": np.mean(num_edges) if num_edges else 0,
            "average_num_nodes": np.mean(num_nodes) if num_nodes else 0,
            "metrics": metrics, # 原始 metric 列表
            "total_tokens_list": total_tokens,
            "prompt_tokens_list": prompt_tokens,
            "completion_tokens_list": completion_tokens,
            "error_folders": wrong_folders, # 合并逻辑错误和完整性错误
            "failure_folders": fail_folders, # 单独列出失败的文件夹
            # 保留旧的分类，如果需要向后兼容
            # "temporal_wrong_folders": temporal_wrong_folders,
            # "completion_wrong_folders": completion_wrong_folders,
        }

        # --- 修改: 保存带后缀的 results 文件 ---
        results_filename = f"results_{args.motif_name}_{model}{prompt_suffix}.json"
        results_save_path = os.path.join(task_folder, results_filename)
        try:
             with open(results_save_path, "w") as f:
                 json.dump(results, f, indent=4)
             print(f"\nEvaluation results saved to: {results_save_path}")
        except Exception as e:
            print(f"\nError saving evaluation results to {results_save_path}: {e}")

        # --- 恢复: 原始打印格式 (更新错误/失败率) ---
        print(f"\n--- Evaluation Summary ---")
        print(f"Task: {task}, Model: {model}, Suffix: {prompt_suffix}") # 添加 Suffix
        # 更新打印输出以反映新的率
        print(f"Correct Rate: {right_rate:.4f}, Error Rate: {wrong_rate:.4f}, Failure Rate: {fail_rate:.4f}")
        print(f"Average Tokens: {average_tokens_calc:.2f}, Total Tokens: {total_tokens_sum}")
        # 恢复 np.std 计算
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
        展示单个任务的结果。(加载带后缀的结果文件)
        """
        args = self.args
        task = args.task
        task_folder = args.task_folder
        model = args.model
        prompt_suffix = _generate_prompt_suffix(args)
        # --- 修改: 加载带后缀的结果文件 ---
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

        # --- 恢复: 打印更多信息 ---
        print(f"\n--- Results Summary (from {os.path.basename(results_file)}) ---")
        print(f"Task: {results_data.get('task', 'N/A')}")
        print(f"Model: {results_data.get('model', 'N/A')}")
        print(f"Suffix: {results_data.get('prompt_config_suffix', 'N/A')}") # 显示后缀
        print(f"Average Accuracy (metric>0): {results_data.get('average_acc', 'N/A'):.4f}")
        print(f"Wrong Rate (metric<0): {results_data.get('wrong_rate', 'N/A'):.2f}")
        print(f"Average Tokens: {results_data.get('average_tokens', 'N/A'):.2f}")
        print(f"Total Tokens: {results_data.get('total_tokens', 'N/A')}")
        # print(f"Temporal Errors (metric=-1): {len(results_data.get('temporal_wrong_folders', []))}")
        # print(f"Completion Errors (metric=-2): {len(results_data.get('completion_wrong_folders', []))}")

        # --- 恢复: 移除分组逻辑说明 ---
        # print("\nNote: Detailed breakdown by T, N, p requires adapting the 'show' method or storing more data in results.")


    def analyze_data(self, task_folder):
        """
        分析生成数据的时间戳、节点数、边数分布，并将统计结果和可视化图保存。
        """
        if not plotting_available:
            print("错误: 绘图库 (matplotlib/seaborn) 不可用。无法执行分析。")
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

        # --- 修改: 初始化所有列表 ---
        all_timestamps = []
        all_node_counts = []
        all_edge_counts = []
        # --- 修改结束 ---
        valid_instances = 0
        folders_to_analyze = instance_folders

        print(f"Analyzing graph properties from {len(folders_to_analyze)} instances...")
        for folder_name in tqdm(folders_to_analyze, desc="Analyzing Graphs"): # 更新描述
            graph_path = os.path.join(task_folder, folder_name, "graph.json")
            try:
                with open(graph_path, "r") as f:
                    graph_data = json.load(f)

                # --- 修改: 收集节点数和边数 ---
                node_count = graph_data.get('N', None)
                edge_count = graph_data.get('num_edges', None) # num_edges 通常等于 len(edge_index)
                timestamps_in_file = [int(edge[2]) for edge in graph_data.get('edge_index', []) if len(edge) > 2] # 确保边元组有效

                if node_count is not None and edge_count is not None:
                    all_node_counts.append(node_count)
                    all_edge_counts.append(edge_count)
                    all_timestamps.extend(timestamps_in_file) # 只有在节点和边有效时才添加时间戳？或者分开处理？分开处理更好
                    valid_instances += 1
                else:
                    print(f"\nWarning: Skipping instance {folder_name} due to missing 'N' or 'num_edges' in graph.json.")
                # --- 修改结束 ---

            except FileNotFoundError:
                 print(f"\nWarning: graph.json not found for instance {folder_name}. Skipping.")
            except json.JSONDecodeError:
                 print(f"\nWarning: graph.json for instance {folder_name} is corrupted. Skipping.")
            except Exception as e:
                 print(f"\nWarning: Error processing {folder_name} during analysis: {e}")

        # --- 修改: 过滤 None 值 (尽管上面的逻辑可能已经避免了，但为了安全) ---
        valid_nodes = [n for n in all_node_counts if n is not None]
        valid_edges = [e for e in all_edge_counts if e is not None]
        valid_timestamps = [t for t in all_timestamps if t is not None] # 确保时间戳列表也干净
        # --- 修改结束 ---

        if not valid_nodes or not valid_edges or not valid_timestamps:
            print(f"错误: 未能从任何有效实例中收集到足够的图属性数据 (需要节点、边和时间戳)。有效实例数: {valid_instances}")
            return

        print(f"分析完成。从 {valid_instances}/{len(instance_folders)} 个有效实例中收集到数据。")
        print(f"  节点数: {len(valid_nodes)}, 边数: {len(valid_edges)}, 时间戳事件: {len(valid_timestamps)}")

        # --- 修改: 计算所有统计信息 ---
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
        # --- 修改结束 ---

        try:
            vis_dir = os.path.join(task_folder, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            # --- 修改: 保存组合统计文件 ---
            stats_save_path = os.path.join(vis_dir, "graph_analysis_stats.json")
            with open(stats_save_path, "w") as f:
                json.dump(stats_data, f, indent=4)
            print(f"图属性统计信息已保存至: {stats_save_path}")
            # --- 修改结束 ---

            # --- 修改: 创建组合绘图 ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # 1 行 3 列

            # 子图 1: 节点数分布
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


            # 子图 2: 边数分布
            sns.histplot(valid_edges, ax=axes[1], kde=True) # 使用 KDE 可以看到更平滑的分布
            axes[1].set_title('Edge Count Distribution')
            axes[1].set_xlabel('Number of Edges')
            axes[1].set_ylabel('Frequency / Density')

            # 子图 3: 时间戳分布 (保留原始直方图逻辑)
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
            plot_filename = os.path.join(vis_dir, "graph_analysis_distributions.png") # 新文件名
            plt.savefig(plot_filename, dpi=150)
            print(f"组合图属性分布图已保存至: {plot_filename}") # 更新打印信息
            plt.close(fig) # 关闭图形，释放内存
            # --- 修改结束 ---

        except Exception as e:
            print(f"错误: 处理统计或绘图时出错: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误追踪
        
    def execute(self, dir):
        """根据 args.t 执行相应的操作。"""
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
            self.gen(task_folder) # gen 方法现在会自动调用 analyze_data
        elif args.t == "run":
            self.run(task_folder)
        elif args.t == "eval":
            self.evaluate(task_folder)
        elif args.t == "check":
            self.check(task_folder)
        elif args.t == "show":
            self.show(task_folder)
        # --- 添加: 显式调用 analyze ---
        elif args.t == "analyze":
            print(f"\n--- Explicitly running Data Analysis ---")
            self.analyze_data(task_folder)
        # --- 添加结束 ---
        else:
            print(f"Error: Action '{args.t}' not implemented.")
            raise NotImplementedError

