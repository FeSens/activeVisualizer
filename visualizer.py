import torch
import torch.nn.functional as F
import functools
from contextlib import contextmanager

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
def capture_pytorch_internal_functions(target_dict, name, patch_info, capture_function):
    functions_dict = {}
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


capture_internal_functions = ['matmul', 'softmax', 'dropout']

@contextmanager
def visualize(model, target_dict, capture_function):
    original_funcs = {}
    try:
        for name, module in model.named_modules():
            # Skip wrapping the root module to avoid double-counting
            if module == model:
                # name = 'model'
                continue
                
            original_forward = module.forward
            
            @functools.wraps(original_forward)
            def wrap_forward(*args, name=name, original_forward=original_forward, **kwargs):
                if name not in target_dict:
                    target_dict[name] = 0
                target_dict[name] += 1
                with capture_pytorch_internal_functions(target_dict, name, capture_internal_functions, capture_function):
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
