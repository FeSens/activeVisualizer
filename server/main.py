from fastapi import FastAPI, WebSocket, Query
from anyio import create_task_group, Semaphore, sleep
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import generate
from visualizer import visualize
import torch
from capture_functions import capture_layers_builder, ablate_attn_activation_builder
from pydantic import BaseModel
from typing import List, Annotated
from fastapi.middleware.cors import CORSMiddleware


torch.set_default_device("mps")
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", trust_remote_code=True)

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)


@app.websocket("/ws/tokenizer")
async def tokenizer_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # using stateful connection we can implement a more complex logic, avoiding the need to send the whole text in every message if necessary
        data = await websocket.receive_text()
        tokenized_text = tokenize(data)

        text = [data[x:y] for (x, y) in tokenized_text["offsets"]]
        tokenized_text["text"] = text
        await websocket.send_json(tokenized_text)


@app.websocket("/ws/model/forward")
async def model_endpoint(websocket: WebSocket):
    await websocket.accept()
    latest_data_semaphore = Semaphore(0, max_value=1)
    latest_data = {}

    async def receiver():
        nonlocal latest_data
        while True:
            data = await websocket.receive_json()
            latest_data = data
            try:
                latest_data_semaphore.release()
            except:
                pass

    async def processor():
        nonlocal latest_data
        while True:
            # sleep for 0.1 seconds to avoid busy waiting
            await sleep(0.1)
            await latest_data_semaphore.acquire()
            data = latest_data  # Copy the latest data for processing

            text = data["text"]
            layer_name = [data["layer_name"]]
            top_k = data["top_k"]
            positions_to_ablate = data.get("positions_to_ablate", [])
            inputs = tokenizer(text, return_tensors="pt",
                               return_attention_mask=False)
            captured_targets = {}

            ablate_fn = ablate_attn_activation_builder(
                position=positions_to_ablate)
            capture_fn = capture_layers_builder(layer_name, captured_targets)

            def visualize_fn(name, tensor):
                ablate_fn(name, tensor)
                capture_fn(name, tensor)

            with visualize(model, visualize_fn):
                with torch.no_grad():
                    logits = model(
                        **inputs, return_dict=False)[0].softmax(dim=-1).topk(top_k, dim=-1)

            logits_values = logits.values.squeeze(
                dim=0).cpu().data.numpy().tolist()
            logits_indices = logits.indices.squeeze(
                dim=0).cpu().data.numpy().tolist()
            logits_tokens = [tokenizer.batch_decode(
                l_i) for l_i in logits_indices]
            outputs = {
                "logits_values": logits_values,
                "logits_tokens": logits_tokens,
                "logits_indices": logits_indices,
                "captured_targets": captured_targets
            }
            await websocket.send_json(outputs)

    async with create_task_group() as tg:
        tg.start_soon(receiver)
        tg.start_soon(processor)


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


@app.get("/v1/decode")
async def decode(tokens: Annotated[List[int], Query()]):
    decoded = tokenizer.decode(tokens)
    return {"decoded_text": decoded}


@app.get("/v1/model_capturable_layers")
async def model_capturable_layers():
    layers = {}
    with visualize(model, capture_layers_builder('all', layers, capture_shape=True, capture_activation=False)):
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2]]))

    return {"model_layers": layers}


def tokenize(text):
    tokens, offsets = tokenizer(
        text, return_attention_mask=False, return_offsets_mapping=True).values()
    return {
        "tokens": tokens,
        "offsets": offsets,
        "n_tokens": len(tokens),
    }
