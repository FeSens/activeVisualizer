import torch
from torch.functional import F
from visualizer import visualize

separate 
def capture_shape(name, tensor, target_dict):
    if isinstance(tensor, torch.Tensor):
        target_dict[f'{name}.shape'] = tensor.shape
    # if out is a list of tensors, store the shape of each tensor
    elif isinstance(tensor, list) and all(isinstance(o, torch.Tensor) for o in tensor):
        for i, o in enumerate(tensor):
              target_dict[f'{name}[{i}].shape'] = o.shape

class MyMatMul(torch.nn.Module):
    def __init__(self):
        super(MyMatMul, self).__init__()
        
        self.layer1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        return torch.matmul(x, x.transpose(0, 1))

# Example model and operation
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.mt = MyMatMul()
    
    def forward(self, x):
        x = self.mt(x)  # Operation to track
        x = F.dropout(x, 0.5)
        return F.softmax(x, dim=-1)

target_dict = {}
def capture_activation(name, tensor, target_dict, laye_name):
    if name == laye_name:
        target_dict[name] = tensor

def capture_targets(name, tensor, target_dict):
    capture_shape(name, tensor, target_dict)
    capture_activation(name, tensor, target_dict, 'mt.matmul.0')

model = MyModel()
x = torch.randn(10, 10)

with visualize(model, target_dict, capture_targets):
  model(x)

model(x)
print(target_dict)