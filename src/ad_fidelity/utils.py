import numpy as np
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)