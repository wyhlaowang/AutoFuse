import os
import re
import cv2
import torch
import uuid
import json
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.io.image import read_image, ImageReadMode
from torchvision import transforms
from vlm import init_qwen, qwen_gen
from llm import init_llama, llama_gen
from tqdm import tqdm


def tensor2PIL(im):
    im = im.repeat(3, 1, 1) if im.shape[0] == 1 else im
    np_array = im.cpu().numpy().astype(np.uint8)
    PIL_image = Image.fromarray(rearrange(np_array, "c h w -> h w c"))
    return PIL_image


def augment(vi, ir, im_h=336, im_w=336):
    vi_c = vi.shape[0]
    ir_c = ir.shape[0]

    NS = torch.randint(im_h, im_h*4, [1]).item()
    transform = transforms.Compose([transforms.Resize(NS),
                                    transforms.RandomCrop([im_h, im_w],
                                                         pad_if_needed=True,
                                                         padding_mode='reflect')])

    vi_ir = torch.cat([vi, ir], dim=0)
    vi_ir_t = transform(vi_ir)
    vi_t, ir_t = torch.split(vi_ir_t, [vi_c, ir_c], dim=0)

    return vi_t, ir_t


def to_y(im):
    im_ra = rearrange(im.cpu(), 'c h w -> h w c').numpy()
    im_ra = np.repeat(im_ra, 3, axis=-1) if im_ra.shape[-1] == 1 else im_ra
    im_ycrcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)
    im_y = torch.from_numpy(im_ycrcb[:, :, 0]).unsqueeze(0)

    return im_y


def text_to_dict_llama3(text):
    matches = re.findall(r'^(?:[-]|\d+\.)\s*([a-z\s]+?)\s*[:]\s*([a-z\s]+)$', text.lower(), re.MULTILINE)
    result_dict = {key.strip(): value.strip() for key, value in matches}
    return result_dict


def preprocess_and_save(data_dir, vis_folder, ir_folder, output_dir, device='cuda:0'):
    vis_file_list = []
    ir_file_list = []
    for ind, ins in enumerate(data_dir):
        vis_dir = os.path.join(ins, vis_folder[ind])
        ir_dir = os.path.join(ins, ir_folder[ind])
        file_ls = os.listdir(vis_dir)
        vis_file = [os.path.join(vis_dir, i) for i in file_ls]
        ir_file = [os.path.join(ir_dir, i) for i in file_ls]
        vis_file_list = vis_file
        ir_file_list = ir_file

    # two gpus are used to accelerate inference speed (qwen in cuda:0, llama in cuda:1)
    vlm_model, vlm_processor = init_qwen(model_name="Qwen/Qwen2-VL-2B-Instruct", if_flash=False, min_pixels=256*28*28, max_pixels=1280*28*28, device=device)
    llm_model = init_llama(model_name="meta-llama/Llama-3.2-3B-Instruct", device='cuda:1', torch_dtype=torch.float16)

    vi_output_dir = os.path.join(output_dir, 'vi')
    ir_output_dir = os.path.join(output_dir, 'ir')
    text_output_dir = os.path.join(output_dir, 'text')

    vlm_msg_deg = [{"role": "user",                
                    "content": [{"type": "image"}, {"type": "text", "text": "Describe any degradation in this image. Reply with keywords."}]}]

    for dir_path in [vi_output_dir, ir_output_dir, text_output_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    for idx in tqdm(range(len(vis_file_list)), desc="Processing images"):
        vi_3 = read_image(vis_file_list[idx], ImageReadMode.RGB).to(device)
        ir_1 = read_image(ir_file_list[idx], ImageReadMode.GRAY).to(device)

        vi_3, ir_1 = augment(vi_3, ir_1)
        vi_1 = to_y(vi_3)

        vlm_deg_vi = qwen_gen(vlm_model, vlm_processor, vlm_msg_deg, device, 16, [tensor2PIL(vi_3)])[0]
        vlm_deg_ir = qwen_gen(vlm_model, vlm_processor, vlm_msg_deg, device, 16, [tensor2PIL(ir_1)])[0]

        llm_msg_deg_vi = [{"role": "user", "content": f"Provide the positive counterpart for each type of image degradation: {vlm_deg_vi}."}]
        llm_deg_vi = llama_gen(llm_model, llm_msg_deg_vi, 64)[0]["generated_text"][-1]["content"]
        llm_deg_vi = text_to_dict_llama3(llm_deg_vi)

        llm_msg_deg_ir = [{"role": "user", "content": f"Provide the positive counterpart for each type of image degradation: {vlm_deg_ir}."}]
        llm_deg_ir = llama_gen(llm_model, llm_msg_deg_ir, 64)[0]["generated_text"][-1]["content"]
        llm_deg_ir = text_to_dict_llama3(llm_deg_ir)

        llm_deg = {**llm_deg_vi, **llm_deg_ir}

        # save
        unique_id = uuid.uuid4().hex[:4]
        vi_y_path = os.path.join(vi_output_dir, f"{os.path.basename(vis_file_list[idx]).split('.')[0]}_{unique_id}.png")
        ir_y_path = os.path.join(ir_output_dir, f"{os.path.basename(ir_file_list[idx]).split('.')[0]}_{unique_id}.png")
        text_output_path = os.path.join(text_output_dir, f"{os.path.basename(vis_file_list[idx]).split('.')[0]}_{unique_id}.json")

        vi_y_image = Image.fromarray((vi_1.squeeze().cpu().numpy()).astype(np.uint8))
        ir_y_image = Image.fromarray((ir_1.squeeze().cpu().numpy()).astype(np.uint8))
        vi_y_image.save(vi_y_path)
        ir_y_image.save(ir_y_path)
        
        with open(text_output_path, 'w') as f:
            json.dump(llm_deg, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess data for FusonData.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for processing (e.g., cuda:0, cuda:1)")
    parser.add_argument("--output_dir", type=str, default="./your_output_path/", help="Used to save generated results")

    args = parser.parse_args()
    
    data_dir = ['./your_dataset_path/']    

    vis_folder = ['vi/']
    ir_folder = ['ir/']

    output_dir = './' + args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_and_save(data_dir, vis_folder, ir_folder, output_dir, device=args.device)

