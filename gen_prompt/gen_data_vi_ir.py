import os
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
from utils import if_text_in, entropy_t4d, text_to_dict_llama3


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


def preprocess_and_save(data_dir, vis_folder, ir_folder, output_dir, step, repeat, device='cuda:0'):
    vis_file_list = []
    ir_file_list = []
    for ind, ins in enumerate(data_dir):
        vis_dir = os.path.join(ins, vis_folder[ind])
        ir_dir = os.path.join(ins, ir_folder[ind])
        file_ls = os.listdir(vis_dir)
        vis_file = [os.path.join(vis_dir, i) for i in file_ls]
        ir_file = [os.path.join(ir_dir, i) for i in file_ls]
        vis_file_list.extend(vis_file[::step[ind]] * repeat[ind])
        ir_file_list.extend(ir_file[::step[ind]] * repeat[ind])

    vlm_model, vlm_processor = init_qwen(model_name="Qwen/Qwen2-VL-2B-Instruct", if_flash=False, min_pixels=256*28*28, max_pixels=1280*28*28, device=device)
    llm_model = init_llama(model_name="meta-llama/Llama-3.2-3B-Instruct", device='cuda:1', torch_dtype=torch.float16)

    vi_output_dir = os.path.join(output_dir, 'vi')
    ir_output_dir = os.path.join(output_dir, 'ir')
    text_output_dir = os.path.join(output_dir, 'text')

    vlm_msg_deg = [{"role": "user",                
                    "content": [{"type": "image"}, {"type": "text", "text": "Describe any visible degradation in this image. Reply with keywords."}]}]

    for dir_path in [vi_output_dir, ir_output_dir, text_output_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    for idx in tqdm(range(len(vis_file_list)), desc="Processing images"):
        vi_3 = read_image(vis_file_list[idx], ImageReadMode.RGB).to(device)
        ir_1 = read_image(ir_file_list[idx], ImageReadMode.GRAY).to(device)

        vi_3, ir_1 = augment(vi_3, ir_1)
        vi_1 = to_y(vi_3)

        # skip relatively flat image patch
        if entropy_t4d(vi_1.unsqueeze(0)/255) < 6 and entropy_t4d(ir_1.unsqueeze(0)/255) < 6:
            continue

        vlm_deg_vi = qwen_gen(vlm_model, vlm_processor, vlm_msg_deg, device, 16, [tensor2PIL(vi_3)])[0]
        vlm_deg_ir = qwen_gen(vlm_model, vlm_processor, vlm_msg_deg, device, 16, [tensor2PIL(ir_1)])[0]

        if if_text_in(vlm_deg_vi, "no"):
            llm_deg_vi = {}
        else:
            llm_msg_deg_vi = [{"role": "user", "content": f"Provide the positive counterpart word for each type of image degradation: {vlm_deg_vi}, within 64 words."}]
            llm_deg_vi = llama_gen(llm_model, llm_msg_deg_vi, 64)[0]["generated_text"][-1]["content"]
            llm_deg_vi = text_to_dict_llama3(llm_deg_vi)

        if if_text_in(vlm_deg_ir, "no"):
            llm_deg_ir = {}
        else:
            llm_msg_deg_ir = [{"role": "user", "content": f"Provide the positive counterpart word for each type of image degradation: {vlm_deg_ir}, within 64 words."}]
            llm_deg_ir = llama_gen(llm_model, llm_msg_deg_ir, 64)[0]["generated_text"][-1]["content"]
            llm_deg_ir = text_to_dict_llama3(llm_deg_ir)

        llm_deg = {**llm_deg_vi, **llm_deg_ir}
        print(llm_deg)

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
    parser.add_argument("--repeat", type=int, nargs='+', default=[1, 1, 1], help="Repeat values for different datasets, typically from 20 to 60")
    parser.add_argument("--step", type=int, nargs='+', default=[1, 1, 1], help="Step values for different datasets")
    parser.add_argument("--output_dir", type=str, default="training_samples", help="Used to save generated results")

    args = parser.parse_args()
    
    data_dir = ['/M3FD_path/', '/LLVIP_path/', '/MSRS_path/']    
    vis_folder = ['vi/', 'vi/', 'vi']
    ir_folder = ['ir/', 'vi/', 'ir']

    output_dir = './' + args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_and_save(data_dir, vis_folder, ir_folder, output_dir, args.step, args.repeat, device=args.device)

