import torch
from torch.nn import Module
from manage_data import BCDataset

class MLP(Module):
    def __init__(self, vector):
        