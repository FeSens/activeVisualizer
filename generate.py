from typing import Any, Callable, Generator, Optional, Tuple
from fastapi import WebSocket
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio


async def listen_for_stop_signal(websocket: WebSocket, task_to_cancel: asyncio.Task):
	print("to esperando")
	data = await websocket.receive_json()
	print("EUE REELKRAKNRKAJSNFJAN")
	if data.get("stop", False) == True:
		task_to_cancel.cancel()

#This code is borrowed from mlx_lm
async def generate(
	model: nn.Module,
	tokenizer: AutoTokenizer,
	prompt: str,
    websocket: WebSocket,
	temp: float = 0.0,
	max_tokens: int = 100,
	repetition_penalty: Optional[float] = None,
	repetition_context_size: Optional[int] = None,
) -> str:
    
	prompt_tokens = tokenizer(prompt, return_tensors="pt", return_attention_mask=False, return_offsets_mapping=False)['input_ids']
	tokens = prompt_tokens.squeeze(0).tolist()
	for (token, prob), n in zip(
		generate_step(
			prompt_tokens,
			model,
			temp,
			repetition_penalty,
			repetition_context_size,
		), range(max_tokens),
	):
		if token == tokenizer.eos_token_id:
			await websocket.send_json({"text": tokenizer.decode(tokens), "finished": True})
			break
        
		tokens.append(token.item())
		await websocket.send_json({"text": tokenizer.decode(tokens), "finished": n == max_tokens - 1})

def generate_step(
    prompt: torch.Tensor,
    model: nn.Module,
    temp: 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (torch.Tensor): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to consider for repetition penalty (default 20).

    Yields:
        Generator[Tuple[torch.Tensor, torch.Tensor]]: A generator producing
        one token and probability per call.
    """

    def sample(logits: torch.Tensor) -> Tuple[torch.Tensor, float]:
        softmax_logits = torch.softmax(logits, dim=-1)

        if temp == 0:
            token = torch.argmax(logits, dim=-1)
        else:
            token = torch.distributions.categorical.Categorical(logits * (1 / temp))

        prob = softmax_logits[0, token]
        return token, prob

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y = prompt
    cache = None

    repetition_context = prompt.tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    while True:
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        logits, cache = model(input_ids=y, past_key_values=cache, use_cache=True, return_dict=False)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, prob = sample(logits)
            repetition_context.append(y.item())
        else:
            y, prob = sample(logits)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]

        yield y, prob

#This code is borrowed from mlx_lm
def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: Any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (torch.Tensor): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (torch.Tensor): Logits with repetition penalty applied to generated tokens.
    """
    if len(generated_tokens) > 0:
        indices = torch.Tensor([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = torch.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits