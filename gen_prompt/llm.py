import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


def init_llama(model_name="meta-llama/Llama-3.2-3B", device="cuda:0", torch_dtype=torch.bfloat16):
    model = pipeline("text-generation", model=model_name, torch_dtype=torch_dtype, device_map=device)
    return model


def llama_gen(model, messages, max_new_tokens=64):
    outputs = model(messages, max_new_tokens=max_new_tokens, pad_token_id=model.tokenizer.eos_token_id)
    return outputs


if __name__ == "__main__":
    import time
    DEV = "cuda:0"
    messages = [{"role": "user", "content": "Who are you?"}]

    model = init_llama(device=DEV, torch_dtype=torch.float16)
    for i in range(50):
        begin = time.time_ns()
        output_text = llama_gen(model, messages)
        end = time.time_ns()
        print(output_text)
        print(f"Time using: {(end-begin)/1e6} ms.")




