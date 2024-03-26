import torch
from einops import rearrange
import re


def capture_layers_builder(layers_name: list, target_dict, capture_shape=False, capture_activation=True, capture_distribution=False, search_activation=False):
    def capture_layers(name, tensor):
        if capture_shape:
            _capture_shape(name, tensor, target_dict, layers_name)
        if capture_activation:
            _capture_activation(name, tensor, target_dict, layers_name)
        if capture_distribution:
            _capture_distribution(name, tensor, target_dict, layers_name)
        if False:
            _search_activation(name, tensor, target_dict)
        if True:
            _capture_attn_heads_that_only_activate_in_0(
                name, tensor, target_dict)
    return capture_layers


def ablate_attn_activation_builder(position: list):
    def ablate_attn_activation(name, tensor):
        for (n_head, value, layers_name) in position:
            _ablate_attn_activation(
                name, tensor, n_head, value, layers_name)

    return ablate_attn_activation


def _search_activation(name, tensor, target_dict, top_k=5):
    layer_type = 'self_attn.softmax'
    if not 'self_attn.softmax' in name:
        return

    target_tensor = torch.tensor(
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).to(tensor.device)
    target_tensor = target_tensor/torch.sum(target_tensor)
    #  Calculate the MSE between the target tensor (B, H, T, T) and the current tensor (T,)
    B, H, T, T = tensor.shape
    t = tensor[:, :, T-1, :]  # 0.07149197906255722 0.0714682936668396
    l1 = torch.sum(((t - target_tensor))**2, dim=2)
    l2 = 0  # torch.mean((t - target_tensor)*(1-target_tensor)**2, dim=2)
    top_k_values, top_k_indices = torch.topk(
        (l1 + l2), dim=1, k=top_k, largest=False)
    # get the top k values
    # merge with target_dict but only keep the top k values globally on layer_type
    for i in range(top_k):
        current_key = f"{layer_type}.top_{i}"
        mse, indices = top_k_values[:, i], top_k_indices[:, i]
        for b in range(B):  # loop through batch to update global top-k
            # Update logic to compare and keep global top-k; simplified for clarity
            global_mse_key = f"{current_key}.mse"
            if global_mse_key not in target_dict or mse[b] < target_dict[global_mse_key]:
                target_dict[global_mse_key] = mse[b].item()
                target_dict[f"{current_key}.index"] = indices[b].cpu(
                ).data.numpy().tolist()
                target_dict[f"{current_key}.name"] = name
                target_dict[f"{current_key}.activ"] = tensor[:,
                                                             indices[b], :, :].cpu().data.numpy().tolist()


def _capture_shape(name, tensor, target_dict, layer_name='all'):
    if True:  # layer_name == 'all' or name in layer_name:
        if isinstance(tensor, torch.Tensor):
            target_dict[f'{name}.shape'] = tensor.shape
      # if out is a list of tensors, store the shape of each tensor
        elif isinstance(tensor, list) and all(isinstance(o, torch.Tensor) for o in tensor):
            for i, o in enumerate(tensor):
                target_dict[f'{name}[{i}].shape'] = o.shape


def _ablate_attn_activation(name, tensor, n_head, value=0, layer_name=''):
    if name == layer_name:
        # (B, H, T, T) = tensor.shape
        # tensor[0][n_head, :, :] = value
        # desired_distribution = tensor[0][n_head, :, :]
        # tensor[0][n_head, :, :] = torch.distributions.Normal(
        #     desired_distribution.mean(), desired_distribution.std()).sample(desired_distribution.shape)
        tensor[0][n_head, :, :] = 0  # tensor[0][n_head, :, :].mean()


def _capture_activation(name, tensor, target_dict, layer_name=[]):
    # Only capture if this if its shape is (B, H, T, T)
    # if len(tensor.shape) == 3:
    #     return
    if layer_name == 'all' or name in layer_name:
        #  normalize the tensor to be between 0 and 1 in the H dimension
        min = tensor.min(dim=3, keepdim=True)[0]
        max = tensor.max(dim=3, keepdim=True)[0]
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
        hist = [torch.histogram(t, bins=32, density=True, range=(min_val, max_val))[
            0].tolist() for t in tensor]
        dist = {
            "mean": tensor.mean(dim=1).tolist(),
            "std": tensor.std(dim=1).tolist(),
            "max": tensor.max(dim=1)[0].tolist(),
            "min": tensor.min(dim=1)[0].tolist(),
            "hist": hist
        }
        target_dict[f"{name}.dist"] = dist


def _capture_attn_heads_that_only_activate_in_0(name, tensor, target_dict, attention_regex="transformer.h.(\d+).attn.softmax.0"):
    # if name matches the attention_regex
    if not re.match(attention_regex, name):
        return
    # B, H, T, T = tensor.shape
    t = tensor[0]  # H, T, T = t.shape
    x = ((t.argmax(dim=2) == 0) * t[:, :, 0] > 0.9).all(dim=1)
    indices = torch.nonzero(x)[:, 0]
    target_dict[f'{name}.heads_that_only_activate_in_0'] = indices.cpu(
    ).data.numpy().tolist()
    if not 'to_ablate' in target_dict:
        target_dict['to_ablate'] = ''

    target_dict['to_ablate'] += " ".join([f"[{i}, 0.0, '{name}']," for i in indices.cpu(
    ).data.numpy().tolist()])
