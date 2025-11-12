# args
from argparse import ArgumentParser
import os.path as osp
import string
def get_args(args=None):
    parser = ArgumentParser()
    # execute
    parser.add_argument("-t", type=str, choices="gen run check clear eval show".split())
    
    # task
    parser.add_argument("--task", type=str, default='judge_motif')
    # Use relative path by default for logs directory
    parser.add_argument("--log_dir", type=str, default=f"logs/{osp.split(osp.dirname(__file__))[-1]}", help="log directory to put generated data and logs")
    parser.add_argument("--use_agent", type=int, default=0, help = "decide whether to use agent")
    parser.add_argument("--api", type=int, default=0, help="decide whether to use api")
    parser.add_argument("--balance", type=int, default=0, help="decide whether to use balance")
    
    # data
    parser.add_argument("--num_seed", type=int, default=10, help="number of problem instances")
    parser.add_argument("--k", type=int, default=1, help="number of examples")
    parser.add_argument("--T", type=int, nargs='+', default=[5], help="number of time steps")
    parser.add_argument("--N", type=int, nargs='+', default=[10], help="number of nodes")
    parser.add_argument("--p", type=float, nargs='+', default=[0.3], help="probability of edges, when p > 1, p means the number of edges")
    parser.add_argument("--w", type=int, nargs='+', default=[0], help="motif time window")
    parser.add_argument('--motif_name', type=str, default='3-star', help='Name of the predefined motif to embed (e.g., triangle, 4-cycle)')

    parser.add_argument("--model", type=str, default='gpt-4o-mini', help="model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--verbose", type=bool, default=False, help="verbose level")
  
    # prompt
    parser.add_argument("--add_cot", type=int, default=False, help="1 add chain of thoughts, 0 not add")
    parser.add_argument("--add_role", type=int, default=False, help="1 add role instruction prompts, 0 not add")
    parser.add_argument("--dyg_type", type=int, default=1, help = "different prompt types for dynamic graphs, see utils/prompt.py")
    parser.add_argument("--edge_type", type=int, default=0, help= "different prompt types for edges, see utils/task/base.py")
    parser.add_argument("--imp", type=int, default=0, help= "additional prompt types for improving, see utils/task/prompt.py")
    parser.add_argument("--short", type=int, default=0, help= "additional prompt types for short answers, see utils/task/prompt.py")
    parser.add_argument("--gen", type=int, default=0, help= "0: random graph, 1: motif graph, 2, 3 ...")
    parser.add_argument("--motif", type=int, default=0, help = "different prompt types for motif, see utils/prompt.py")
    parser.add_argument("--multi_motif", type=int, default=0, help = "all motif definitions")
    # misc
    parser.add_argument("--change", type=int, default=0, help="")
    parser.add_argument("--key", type=str, default=0, help="")
    
    args = parser.parse_args()
    args.num_examplars = args.k
    return args