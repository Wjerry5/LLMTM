from .sort_edge import DyGraphTaskSortEdge
from .when_direct_link import DyGraphTaskWhenDirectLink
from .what_edge_at_time import DyGraphTaskWhatEdgeAtTime
from .reverse_graph import DyGraphTaskReverseGraph
from .judge_motif import DyGraphTaskJudgeMotif
from .modify_dgy import DyGraphTaskModifyDyG
# from .when_indirect_link import DyGraphTaskWhenIndirectLink
from .judge_multi_motif import DyGraphTaskJudgeMultiMotif
from .judge_contain_motif import DyGraphTaskJudgeContainMotif
from .multi_motif_first_time import DyGraphTaskMultiMotifFirstTime
from .multi_motif_count import DyGraphTaskMultiMotifCount
# from .copy import DyGraphTaskCopyEdge
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
def load_task(task, args):
    if task == "sort_edge":
        return DyGraphTaskSortEdge(task, args)
    elif task == "when_direct_link":
        return DyGraphTaskWhenDirectLink(task, args)
    elif task == "what_edge_at_time":
        return DyGraphTaskWhatEdgeAtTime(task, args)
    elif task == "reverse_graph":
        return DyGraphTaskReverseGraph(task, args)
    elif task == "judge_motif":
        return DyGraphTaskJudgeMotif(task, args)
    elif task == "modify_dyg":
        return DyGraphTaskModifyDyG(task, args) 
    elif task == "when_indirect_link":
        return DyGraphTaskWhenIndirectLink(task, args)
    elif task == "judge_multi_motif":
        return DyGraphTaskJudgeMultiMotif(task, args)
    elif task == "judge_contain_motif":
        return DyGraphTaskJudgeContainMotif(task, args)
    elif task == "copy_edge":
        return DyGraphTaskCopyEdge(task, args)
    elif task == "multi_motif_first_time":
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