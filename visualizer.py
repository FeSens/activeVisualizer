import torch
import torch.nn.functional as F
import functools
from contextlib import contextmanager


def capture_shape(name, tensor, target_dict):
    if isinstance(tensor, torch.Tensor):
        target_dict[f'{name}.shape'] = tensor.shape
    # if out is a list of tensors, store the shape of each tensor
    elif isinstance(tensor, list) and all(isinstance(o, torch.Tensor) for o in tensor):
        for i, o in enumerate(tensor):
              target_dict[f'{name}[{i}].shape'] = o.shape

# Function to patch another function
def patch_pytorch_internal_function(functions_dict, target_dict, func_to_patch, patch_name, module, capture_function):
    original_func = getattr(module, func_to_patch)
    # Initialize a counter for each operation type
    if func_to_patch not in functions_dict:
        functions_dict[func_to_patch] = {'count': 0}

    @functools.wraps(original_func)
    def patched_func(*args, **kwargs):
        op_counter = functions_dict[func_to_patch]['count']
        result = original_func(*args, **kwargs)
        # Generate a unique key for this operation invocation
        key = f"{patch_name}.{func_to_patch}.{op_counter}"
        capture_function(key, result, target_dict)
        # Increment the counter after the operation is tracked
        functions_dict[func_to_patch]['count'] += 1
        return result
    
    return patched_func

@contextmanager
def capture_pytorch_internal_functions(target_dict, name, patch_info, functions_dict, capture_function):
    original_funcs = {torch: {}, F: {}}
    try:
        for func_name in patch_info:
            for module in original_funcs.keys():
              original_func = getattr(module, func_name, None)
              if original_func is None:
                  continue
              patched_func = patch_pytorch_internal_function(functions_dict, target_dict, func_name, name, module, capture_function=capture_function)
              original_funcs[module][func_name] = original_func
              setattr(module, func_name, patched_func)
              
        yield
    finally:
        # Restore original functions
        for module, func in original_funcs.items():
            for func_name, original_func in func.items():
                setattr(module, func_name, original_func)


target_dict = {}
patch_info = {'matmul': 'layer.0.attention.matmul', 'softmax': 'layer.0.attention.softmax', 'dropout': 'layer.0.attention.dropout'}
capture_internal_functions = ['matmul', 'softmax', 'dropout']


@contextmanager
def visualize(model, target_dict, capture_function):
    original_funcs = {}
    try:
        for name, module in model.named_modules():
            # Skip wrapping the root module to avoid double-counting
            function_dict = {}
            if module == model:
                # name = 'model'
                continue
                
            original_forward = module.forward
            
            @functools.wraps(original_forward)
            def wrap_forward(*args, name=name, original_forward=original_forward, **kwargs):
                if name not in target_dict:
                    target_dict[name] = 0
                target_dict[name] += 1
                with capture_pytorch_internal_functions(target_dict, name, capture_internal_functions, function_dict, capture_function):
                  out = original_forward(*args, **kwargs)
                capture_function(name, out, target_dict)
                return out
            
            # Assign the wrapped function to the module's forward
            original_funcs[name] = original_forward
            module.forward = wrap_forward

        yield
    finally:
        # Restore original functions
        for name, module in model.named_modules():
            if module == model:
                continue
            module.forward = original_funcs[name]
    
    return model

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
