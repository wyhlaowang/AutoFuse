import torch
from transformers import pipeline


def init_llama(model_name="meta-llama/Llama-3.2-3B", device="cuda:0", torch_dtype=torch.bfloat16):
    model = pipeline("text-generation", model=model_name, torch_dtype=torch_dtype, device_map=device)
    return model


def llama_gen(model, messages, max_new_tokens=64):
    outputs = model(messages, max_new_tokens=max_new_tokens, pad_token_id=model.tokenizer.eos_token_id)
    return outputs

