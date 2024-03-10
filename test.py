import torch
from torch.functional import F
from visualizer import visualize
from capture_functions import capture_layers_builder
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("mps")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


target_dict = {}
capture_targets = capture_layers_builder(['model.layers.2.self_attn.softmax.0'], target_dict)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

with visualize(model, capture_targets):
  outputs = model(**inputs)


import json
# Save the captured shapes to a JSON file
with open('target_dict.json', 'w') as f:
    json.dump(target_dict, f, indent=2)
