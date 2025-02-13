import torch

def get_rank() -> int:
    # Always return 0 since there is no distributed setup.
    return 0

def is_main_process() -> bool:
    # Since we have only one GPU, it will always be the main process.
    return True

def gpu_available() -> bool:
    # Check if a GPU is available for use.
    return torch.cuda.is_available()