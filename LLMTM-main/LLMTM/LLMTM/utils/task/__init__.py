import os
import shutil
from .sort_edge import DyGraphTaskSortEdge
from .when_link_and_dislink import DyGraphTaskWhenDirectLink
from .what_edges import DyGraphTaskWhatEdgeAtTime
from .reverse_graph import DyGraphTaskReverseGraph
from .judge_is_motif import DyGraphTaskJudgeMotif
from .modify_motif import DyGraphTaskModifyDyG
from .multi_motif_judge import DyGraphTaskJudgeMultiMotif
from .judge_contain_motif import DyGraphTaskJudgeContainMotif
from .when_multi_motif_exist import DyGraphTaskMultiMotifFirstTime
from .multi_motif_count import DyGraphTaskMultiMotifCount

def load_task(task, args):
    if task == "sort_edge":
        return DyGraphTaskSortEdge(task, args)
    elif task == "when_link_and_dislink":
        return DyGraphTaskWhenDirectLink(task, args)
    elif task == "what_edges":
        return DyGraphTaskWhatEdgeAtTime(task, args)
    elif task == "reverse_graph":
        return DyGraphTaskReverseGraph(task, args)
    elif task == "judge_is_motif":
        return DyGraphTaskJudgeMotif(task, args)
    elif task == "modify_motif":
        return DyGraphTaskModifyDyG(task, args) 
    elif task == "multi_motif_judge":
        return DyGraphTaskJudgeMultiMotif(task, args)
    elif task == "judge_contain_motif":
        return DyGraphTaskJudgeContainMotif(task, args)
    elif task == "when_multi_motif_exist":
        return DyGraphTaskMultiMotifFirstTime(task, args)
    elif task == "multi_motif_count":
        return DyGraphTaskMultiMotifCount(task, args)
    else:
        raise NotImplementedError(f"Task '{task}' not implemented or removed.")

def remove_dir(log_dir):
    """
    Remove the specified directory and all its contents after user confirmation.

    Args:
        log_dir (str): Path of the directory to remove.
    """
    # Keep relative path output; avoid forcing absolute path conversion
    print(f"Removing {log_dir}? y or n ")  # Ask for user confirmation
    a = input()  # Read user input
    if a.strip().lower() == "y":  # If user inputs 'y' (case-insensitive, trim spaces)
        print(f"Removing {log_dir}")
        shutil.rmtree(log_dir)  # Recursively remove directory using shutil