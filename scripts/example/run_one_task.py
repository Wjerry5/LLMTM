import sys 
import os

_current_file_path = os.path.abspath(__file__)
_script_dir = os.path.dirname(_current_file_path)
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import LLMTM 
from LLMTM.utils import load_task
import json
from config import get_args
from tqdm import tqdm
import random
from LLMTM.runner import Runner
import numpy as np
import pandas as pd
import time
import langchain
import logging

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
"""


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

args = get_args() 
log_dir = args.log_dir 
file_name = os.path.splitext(os.path.split(__file__)[-1])[0]

def get_task_folder(task):
    return os.path.join(log_dir, f"{file_name}", f"{task}")

task_folder = args.task_folder = get_task_folder(args.task)
print(f"Task folder set to: {task_folder}")


MRun = Runner 
runner = MRun(args, try_all = True)
runner.execute(task_folder) 

print(f"\n--- Run One Task Script Finished ({args.t} mode for task '{args.task}') ---")

langchain.debug = True

logging.basicConfig(level=logging.INFO) 