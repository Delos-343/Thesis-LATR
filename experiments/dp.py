import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import subprocess
import numpy as np
import random

def setup_dist_launch(args):
    # Data Parallel doesn't require distributed setup, so no need for proc_id, world_size, or local_rank.
    print("Running on a single GPU setup.")
    args.proc_id = 0  # Always 0 for single GPU setup
    args.world_size = 1  # Single GPU world size
    args.local_rank = 0  # Local rank is always 0

def setup_slurm(args):
    # SLURM logic for DDP is not needed, as we're focusing on single-GPU usage.
    print("Running on a single GPU setup with SLURM.")
    args.proc_id = 0  # Always 0 for single GPU
    args.local_rank = 0  # Local rank is always 0
    args.world_size = 1  # Single GPU world size

def setup_distributed(args):
    # Remove distributed setup, since we're using DataParallel (DP) for single GPU
    print("No distributed setup required. Using a single GPU.")
    args.gpu = 0  # Only using one GPU
    torch.cuda.set_device(args.gpu)

    # Use DataParallel directly when running on a single GPU
    # DataParallel will handle a single GPU seamlessly without distributed mode.

def dp_init(args):
    # Simplified for single GPU usage with DataParallel
    args.proc_id, args.gpu, args.world_size = 0, 0, 1

    # Single GPU mode: No need for distributed setup
    print("Running on a single GPU with DataParallel.")
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(args.proc_id)
    np.random.seed(args.proc_id)
    random.seed(args.proc_id)
    
    # If DataParallel was needed in the future:
    # model = torch.nn.DataParallel(model).cuda()

    return  # No distributed training setup needed here

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def reduce_tensor(tensor, world_size):
    # No need to use dist.all_reduce in DataParallel mode, since it handles this internally
    return tensor

def reduce_tensors(*tensors, world_size):
    # In DataParallel mode, reduce is handled automatically so no need to do any manual reduction
    return tensors  # Return the original tensors as they do not need to be reduced in DP