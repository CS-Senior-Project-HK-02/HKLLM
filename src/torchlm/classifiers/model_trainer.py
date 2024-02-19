import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
from torchlm.models import *
from torchlm.utils.Exceptions import InvalidModelPathError
from tqdm import tqdm
from torch import torchmetrics
from accelerate import Accelerator


class Classifier:
    def __init__(self, model, device, use_mixed_precision = False, log_dir = None, model_checkpoint = None):
        if model == None:
            raise Exception
        self.model = model.to(device)
        
    def train_step(model,train_dataloader,num_epochs):
        
        
        accel = Accelerator()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                outputs = model()
                
                loss = outputs.loss
                
                accel.backward(loss)
            
            
        
        
        
        
        accel.backward()
        pass
    def val_step():
        pass
    def test_step():
        pass
    
    
    
    
def main(mode="full"):
    
    
    if mode in ["train", "full"]:
        pass
    pass

    if mode in ["test", "full"]:
        pass
        


if __name__ =="__main__":
    main(mode="full") #Can also be train or test
    