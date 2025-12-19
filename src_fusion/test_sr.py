import os
import cv2
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision.io.image import read_image, ImageReadMode
from einops import rearrange
from model import Encoder, Decoder, Modal
from torchvision import transforms


def rgb_y(im):
    DEV = im.device
    im_ra = rearrange(im, 'c h w -> h w c').cpu().numpy()
    im_ycrcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)
    im_y = torch.from_numpy(im_ycrcb[:,:,0]).unsqueeze(0).to(device=DEV)
    return im_y


def to_rgb(im_3, im_1):
    DEV = im_1.device
    im_3 = rearrange(im_3, 'c h w -> h w c').cpu().numpy()
    im_1 = rearrange(im_1, 'c h w -> h w c').cpu().numpy()
    crcb = cv2.cvtColor(im_3, cv2.COLOR_RGB2YCrCb)[:,:,1:]
    ycrcb = np.concatenate((im_1, crcb), -1)
    rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return rearrange(torch.from_numpy(rgb), 'h w c -> c h w').to(device=DEV)


def load(device='cuda:0'):
    enc = Encoder(2, 32, 3, False).to(device=device).eval()
    mod = Modal(2, 64, 16, False).to(device=device).eval()
    dec = Decoder(1, 32*(2**4), 4, False, 16, "reflect").to(device=device).eval()

    enc_w = torch.load('./weight/enc.pt', map_location=device)
    enc.load_state_dict(enc_w)

    mod_w = torch.load('./weight/mod.pt', map_location=device)
    mod.load_state_dict(mod_w)

    dec_w = torch.load('./weight/dec.pt', map_location=device)
    dec.load_state_dict(dec_w)

    print('=== Pretrained models load done ===')
    
    return enc, mod, dec


def t4d_save(t4d, save_path, save_file_name, subfolder):
    C = t4d.shape[1]
    if C == 1:
        im = 255 * t4d.cpu().squeeze(0).squeeze(0).clamp(0,1)
        im = Image.fromarray(im.numpy().astype('uint8'))
    else:
        im = 255 * t4d.cpu().squeeze(0).clamp(0,1)
        im = Image.fromarray(rearrange(im, 'c h w -> h w c').numpy().astype('uint8'))

    im.save(os.path.join(save_path, subfolder, Path(save_file_name).with_suffix(".png")), quality=100)


def test(dev): 
    enc, mod, dec = load(dev)
    ir_folder = 'ir'
    vis_folder = 'vi'
    data_path = './test_imgs/'
    save_path = './results/'

    os.makedirs(os.path.join(save_path, 'sr'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'nr'), exist_ok=True)

    file_list = os.listdir(os.path.join(data_path, ir_folder))

    us = nn.Upsample(scale_factor=2)
    print(f'Testing ... ')

    with torch.no_grad():
        for i in file_list:
            ir = read_image(os.path.join(data_path, ir_folder, i), ImageReadMode.GRAY).to(device=dev) / 255.
            vi = read_image(os.path.join(data_path, vis_folder, i), ImageReadMode.RGB).to(device=dev) / 255.

            vi_re = us(vi.unsqueeze(0)).squeeze(0)
            _, H, W = vi_re.shape
            ds = transforms.Resize((H//2, W//2)) 

            vi_1 = rgb_y(vi)
            fu = dec(enc(vi_1.unsqueeze(0), ir.unsqueeze(0)), mod(vi_1.unsqueeze(0), ir.unsqueeze(0)))
            fu = fu[:,:,:H,:W].squeeze(0)
            print(fu.shape)

            fu_3 = to_rgb(vi_re, fu)

            t4d_save(fu_3.unsqueeze(0), save_path, i, subfolder='sr')
            t4d_save(ds(fu_3).unsqueeze(0), save_path, i, subfolder='nr')


if __name__ == '__main__':
    test(dev='cuda:0')


