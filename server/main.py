from fastapi import FastAPI, WebSocket, Query
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import generate
from visualizer import visualize
import torch
from capture_functions import capture_layers_builder
from pydantic import BaseModel
from typing import List, Annotated

torch.set_default_device("mps")
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)

app = FastAPI()


@app.websocket("/ws/tokenizer")
async def tokenizer_endpoint(websocket: WebSocket):
	await websocket.accept()
	while True:
		# using stateful connection we can implement a more complex logic, avoiding the need to send the whole text in every message if necessary
		data = await websocket.receive_text()
		tokenized_text = tokenize(data)
		await websocket.send_json(tokenized_text)

@app.websocket("/ws/model/forward")
async def model_endpoint(websocket: WebSocket):
	await websocket.accept()
	while True:
		# using stateful connection we can implement a more complex logic, maybe caching the model outputs for a given input or implementing streaming of generated text
		data = await websocket.receive_json()
		text = data["text"]
		layer_name = data["layer_name"]
		top_k = data["top_k"]
		inputs = tokenizer(text, return_tensors="pt", return_attention_mask=False)
		captured_targets = {}

		with visualize(model, capture_layers_builder(layer_name, captured_targets)):
			with torch.no_grad():
				logits = model(**inputs)['logits'].topk(top_k, dim=-1)
		
		logits_values = logits.values.squeeze(dim=0).cpu().data.numpy().tolist() # squeeze to remove the batch dimension
		logits_indices = logits.indices.squeeze(dim=0).cpu().data.numpy().tolist() # squeeze to remove the batch dimension
		outputs = {
			"logits_values": logits_values,
			"logits_indices": logits_indices,
			"captured_targets": captured_targets
		}
		await websocket.send_json(outputs)

@app.websocket("/ws/model/generate")
async def model_endpoint(websocket: WebSocket):
	await websocket.accept()
	while True:
		# using stateful connection we can implement a more complex logic, maybe caching the model outputs for a given input or implementing streaming of generated text
		data = await websocket.receive_json()
		text = data["text"]
		# layer_name = data["layer_name"]
		# top_k = data["top_k"]
		# inputs = tokenizer(text, return_tensors="pt", return_attention_mask=False)
		# captured_targets = {}
		await generate(model, tokenizer, text, websocket, temp=0.0, max_tokens=30, repetition_penalty=None, repetition_context_size=None)

# class Decode(BaseModel):
#     ids: List[int]

@app.get("/decode")
async def decode(tokens: Annotated[List[int], Query()]):
	decoded = tokenizer.decode(tokens)
	return { "decoded_text": decoded }

@app.get("/model_capturable_layers")
async def model_capturable_layers():
	layers = {}
	with visualize(model, capture_layers_builder('all', layers, capture_shape=True, capture_activation=False)):
		with torch.no_grad():
			model(input_ids=torch.tensor([[1, 2]]))

	return { "model_layers": layers }

def tokenize(text):
    tokens, offsets = tokenizer(text, return_attention_mask=False, return_offsets_mapping=True).values()
    return {
        "tokens": tokens,
        "offsets": offsets,
        "n_tokens": len(tokens),
    }
