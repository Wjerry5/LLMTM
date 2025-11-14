import sys 
import os
_current_file_path = os.path.abspath(__file__)
_script_dir = os.path.dirname(_current_file_path)
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from config import get_args

from tqdm import tqdm
import random
from LLMTM.runner import Runner
import numpy as np
import pandas as pd
import time
import langchain
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

"""
This script is used to execute the complete process for a **single specific task** in the LLMTM framework.
It utilizes the Runner class to handle:
1. (-t gen)   Data generation: Generate multiple problem instances for the specified task (--task) 
               and parameter combinations (--T, --N, --p) (number controlled by --num_seed) and save them.
2. (-t run)   Model execution: Load the generated data, call LLM (--model) for each instance to get answers and save them.
               Includes try_all=True logic, which continues to attempt until all instances are processed or an error occurs.
3. (-t check) Status check: Check the current task folder to see how many instances are completed and how many are pending.
4. (-t clear) Data cleanup: Delete the entire data and results folder for the current task.
5. (-t eval)  Results evaluation: Load model answers and standard answers, calculate and save evaluation metrics.
6. (-t show)  Results display: (This script calls Runner.show, whose base implementation may not be fully complete,
               MRun.show in run_tasks.py has more functionality) Display evaluation results.


Usage examples:
# 1. Generate data for the 'sort_edge' task (e.g., 10 instances, T=5, N=10, p=0.3)
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 --num_seed 100 -t gen
# 2. Check the status of the 'sort_edge' task
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t check
# 3. Run the gpt-4o-mini model for the 'sort_edge' task to get answers
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t run --model gpt-4o-mini --k 1 --add_cot 1 
# 4. Evaluate the results of gpt-4o-mini on the 'sort_edge' task (using the same suffix parameters as run)
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t eval --model gpt-4o-mini --k 1 --add_cot 1 
# 5. Display the evaluation results of gpt-4o-mini on the 'sort_edge' task (using the same suffix parameters as eval)
#path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t show --model gpt-4o-mini --k 1 --add_cot 1 
# 6. (Optional) Clean up the data for the 'sort_edge' task
# path to your/scripts/example/run_one_task.py --task sort_edge --T 5 --N 10 --p 0.3 -t clear

# --- Another valid task example: when_direct_link ---
# python scripts/example/run_one_task.py --task when_direct_link --T 5 --N 10 --p 0.3 --num_seed 10 -t gen
# python scripts/example/run_one_task.py --task when_direct_link --T 5 --N 10 --p 0.3 -t run --model gpt-4o-mini --k 1
# python scripts/example/run_one_task.py --task when_direct_link --T 5 --N 10 --p 0.3 -t eval --model gpt-4o-mini --k 1
"""

# --- 1. Setting parameters and paths ---
args = get_args() # Parse command line arguments
log_dir = args.log_dir # Get base log directory
file_name = os.path.splitext(os.path.split(__file__)[-1])[0]

def get_task_folder(task):
    """Construct the log/data folder path for a specific task."""
    return os.path.join(log_dir, f"{file_name}", f"{task}")

task_folder = args.task_folder = get_task_folder(args.task)
print(f"Task folder set to: {task_folder}")

# --- 2. Execute Runner ---
MRun = Runner # Directly use the base Runner class
runner = MRun(args, try_all = True)
runner.execute(task_folder) # Pass task_folder

print(f"\n--- Run One Task Script Finished ({args.t} mode for task '{args.task}') ---")

langchain.debug = True

logging.basicConfig(level=logging.INFO) # or logging.DEBUG
# If you want to specifically see LangChain logs
# logging.getLogger("langchain").setLevel(logging.DEBUG)