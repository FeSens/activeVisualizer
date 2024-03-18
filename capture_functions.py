import torch
from einops import rearrange

def capture_layers_builder(layers_name: list, target_dict, capture_shape=True, capture_activation=True, capture_distribution=True):
    def capture_layers(name, tensor):
        if capture_shape:
            _capture_shape(name, tensor, target_dict, layers_name)
        if capture_activation:
            _capture_activation(name, tensor, target_dict, layers_name)
        if capture_distribution:
            _capture_distribution(name, tensor, target_dict, layers_name)
    return capture_layers

def _capture_shape(name, tensor, target_dict, layer_name='all'):
    if layer_name == 'all' or name in layer_name:
        if isinstance(tensor, torch.Tensor):
            target_dict[f'{name}.shape'] = tensor.shape
      # if out is a list of tensors, store the shape of each tensor
    elif isinstance(tensor, list) and all(isinstance(o, torch.Tensor) for o in tensor):
        for i, o in enumerate(tensor):
            target_dict[f'{name}[{i}].shape'] = o.shape

def _capture_activation(name, tensor, target_dict, layer_name=[]):
    # Only capture if this if its shape is (B, H, T, T)
    
    # if len(tensor.shape) == 3:
    #     return
    if layer_name == 'all' or name in layer_name:
        #  normalize the tensor to be between 0 and 1 in the H dimension
        min = tensor.min(dim=2, keepdim=True)[0]
        max = tensor.max(dim=2, keepdim=True)[0]
        norm = (tensor - min) / (max - min + 1e-6)
        target_dict[f"{name}.activ"] = tensor.cpu().data.numpy().tolist()
        target_dict[f"{name}.activ_norm"] = norm.cpu().data.numpy().tolist()

def _capture_distribution(name, tensor: torch.Tensor, target_dict, layer_name=[]):
    if layer_name == 'all' or name in layer_name:
        # Store the distribution of the tensor
        tensor = tensor.squeeze(0).cpu().float()
          # Only capture if this if its shape is (B, T, H)
        if len(tensor.shape) == 3:
            return
            # tensor = rearrange(tensor, 'h t_h t_w -> t_h (h t_w)')
        # Normalize hist by whole tensor max and min
        # Create a vizualization for residual streams [hist, hist, ... N_layers] x Tokens
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        hist = [torch.histogram(t, bins=32, density=True, range=(min_val, max_val))[0].tolist() for t in tensor]
        dist = {
            "mean": tensor.mean(dim=1).tolist(),
            "std": tensor.std(dim=1).tolist(),
            "max": tensor.max(dim=1)[0].tolist(),
            "min": tensor.min(dim=1)[0].tolist(),
            "hist": hist
        }
        target_dict[f"{name}.dist"] = dist