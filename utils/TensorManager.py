import torch
import logging


class TensorManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load(self, path):
        self.logger.debug(f"Loading tensor from {path}")
        return torch.load(path)

    def save(self, tensor, path):
        self.logger.debug(f"Saving tensor to {path}")
        torch.save(tensor, path)

