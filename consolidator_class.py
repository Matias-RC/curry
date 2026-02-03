import torch as th
import torch.nn as nn

class Consolidator(nn.Module):
    def __init__(self, *args, **kwargs):
        self.optimizer = None
        pass
    def set_training_mode(self, mode: bool) -> None:
        pass
    def _init_empty(self):
        pass