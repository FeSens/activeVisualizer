import torch
from torch.functional import F
from visualizer import visualize


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import functools

torch.set_default_device("mps")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

def capture_shape(name, tensor, target_dict):
    if isinstance(tensor, torch.Tensor):
        target_dict[f'{name}.shape'] = tensor.shape
    # if out is a list of tensors, store the shape of each tensor
    elif isinstance(tensor, list) and all(isinstance(o, torch.Tensor) for o in tensor):
        for i, o in enumerate(tensor):
              target_dict[f'{name}[{i}].shape'] = o.shape

def capture_activation(name, tensor, target_dict, laye_name):
    if name == laye_name:
        target_dict[name] = tensor.cpu().data.numpy().tolist()

def capture_targets(name, tensor, target_dict):
    capture_shape(name, tensor, target_dict)
    capture_activation(name, tensor, target_dict, 'model.layers.2.self_attn.softmax.0')

target_dict = {}

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

with visualize(model, target_dict, capture_targets):
  outputs = model(**inputs)


import json
# Save the captured shapes to a JSON file
with open('target_dict.json', 'w') as f:
    json.dump(target_dict, f, indent=2)
