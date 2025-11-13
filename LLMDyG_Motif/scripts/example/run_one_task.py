import sys 
import os

_current_file_path = os.path.abspath(__file__)
_script_dir = os.path.dirname(_current_file_path)
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import LLMDyG_Motif 
from LLMDyG_Motif.utils import load_task
import json
from config import get_args
from tqdm import tqdm
import random
from LLMDyG_Motif.runner import Runner
import numpy as np
import pandas as pd
import time
import langchain
import logging

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
"""
此脚本用于执行 LLMTM 框架中针对 **单个特定任务** 的完整流程。
它利用 Runner 类来处理：
1. (-t gen)   数据生成：为指定的任务 (--task) 和参数组合 (--T, --N, --p) 生成多个
               问题实例 (数量由 --num_seed 控制) 并保存。
2. (-t run)   模型运行：加载生成的数据，为每个实例调用 LLM (--model) 获取答案并保存。
               包含 try_all=True 逻辑，会持续尝试直到所有实例处理完毕或出错。
3. (-t check) 状态检查：检查当前任务文件夹下，有多少实例已完成，多少待运行。
4. (-t clear) 清理数据：删除当前任务的整个数据和结果文件夹。
5. (-t eval)  结果评估：加载模型答案和标准答案，计算并保存评估指标。
6. (-t show)  结果展示：(在此脚本中调用 Runner.show，其基类实现可能不够完善，
               run_tasks.py 中的 MRun.show 功能更全) 展示评估结果。
7. (-t analyze) 数据分析：在 'gen' 步骤后自动执行，分析生成数据的时间戳分布。此动作不再单独调用。


Usage examples:
# 1. Generate data for the 'sort_edge' task (e.g., 10 instances, T=5, N=10, p=0.3)
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 --num_seed 100 -t gen
# 2. Check the status of the 'sort_edge' task
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t check
# 3. Run the gpt-4o-mini model for the 'sort_edge' task to get answers
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t run --model pangu_auto --k 1 --add_cot 1 
# 4. Evaluate the results of gpt-4o-mini on the 'sort_edge' task (using the same suffix parameters as run)
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t eval --model pangu_auto --k 1 --add_cot 1 
# 5. Display the evaluation results of gpt-4o-mini on the 'sort_edge' task (using the same suffix parameters as eval)
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t show --model pangu_auto --k 1 --add_cot 1 
# 6. (Optional) Clean up the data for the 'sort_edge' task
# path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t clear

# --- Another valid task example: when_direct_link ---
# python scripts/example/run_one_task.py --task when_direct_link --T 5 --N 10 --p 0.3 --num_seed 10 -t gen
# python scripts/example/run_one_task.py --task when_direct_link --T 5 --N 10 --p 0.3 -t run --model pangu_auto --k 1
# python scripts/example/run_one_task.py --task when_direct_link --T 5 --N 10 --p 0.3 -t eval --model pangu_auto --k 1
"""

# --- 1. 设置参数和路径 ---
args = get_args() # 解析命令行参数
log_dir = args.log_dir # 获取基础日志目录
file_name = os.path.splitext(os.path.split(__file__)[-1])[0]

def get_task_folder(task):
    """构建特定任务的日志/数据文件夹路径。"""
    return os.path.join(log_dir, f"{file_name}", f"{task}")

task_folder = args.task_folder = get_task_folder(args.task)
print(f"Task folder set to: {task_folder}")

# --- 2. 执行 Runner ---
MRun = Runner # 直接使用基础 Runner 类
runner = MRun(args, try_all = True)
runner.execute(task_folder) # 传递 task_folder

print(f"\n--- Run One Task Script Finished ({args.t} mode for task '{args.task}') ---")

langchain.debug = True

logging.basicConfig(level=logging.INFO) # 或者 logging.DEBUG
# 如果想专门看 LangChain 的日志
# logging.getLogger("langchain").setLevel(logging.DEBUG)