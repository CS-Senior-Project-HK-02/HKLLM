import torch
import torch.nn
import os

class InvalidModelPathError(Exception):
    """Exception raised when the model path is invalid"""
    def __init__(self,path, message = "The model path is invalid or the file does not exist: "):
        self.path = path
        self.message = f"{message} {path}"
        super().__init__(self.message)