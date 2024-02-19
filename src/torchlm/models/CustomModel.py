import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_Model(nn.Module):
    def __init__(self, **kwargs):
        super(Custom_Model, self).__init__()
        from_checkpoint = kwargs.get('model_dict',None)
        
        
        
    def forward():
        pass
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate the model
model = Custom_Model()

# Calculate total and trainable parameters
total_params = count_parameters(model)
trainable_params = count_trainable_parameters(model)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
            