from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def init_qwen(model_name="Qwen/Qwen2-VL-2B-Instruct", if_flash=True, min_pixels=256*28*28, max_pixels=1280*28*28, device="cuda:0", torch_dtype="auto"):
    model_kwargs = {}
    if if_flash == True:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        **model_kwargs)

    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs['min_pixels'] = min_pixels
    if max_pixels is not None:
        processor_kwargs['max_pixels'] = max_pixels
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    return model, processor


def qwen_gen(model, processor, messages, device, max_new_tokens=128, images=None):
    # Preparation for inference
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs = process_vision_info(messages)[0] if images is None else images
    inputs = processor(text=[texts], images=image_inputs, videos=None, padding=True, return_tensors="pt").to(device)

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)    

    return output_text


if __name__ == "__main__":  
    import time
    from PIL import Image

    DEV = "cuda:0"

    model, processor = init_qwen(if_flash=False, max_pixels=256*28*28, device=DEV)

    messages = [{"role": "user",                
                "content": [{"type": "image"},
                            {"type": "text", "text": "Please describe any visible degradation in this image. Reply with keywords."}]}]
    
    img = Image.open("./data/vi/00320.png")

    for i in range(50):
        begin = time.time_ns()
        output_text = qwen_gen(model, processor, messages, DEV, 16, img)
        end = time.time_ns()
        print(output_text)
        print(f"Time using: {(end-begin)/1e6} ms.")



