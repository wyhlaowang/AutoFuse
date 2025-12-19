# open lib
import time
import os
import shutil
import itertools
import clip
import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.backends.cudnn as cudnn
from pathlib import Path
from tqdm.auto import tqdm
# self lib
from model import *
from config import args as args_config
from IVTData import IVTData
from vgg import VGG



os.environ['MASTER_PORT'] = '1233'  


random.seed(178)
torch.manual_seed(178)
torch.cuda.manual_seed(178)
torch.cuda.manual_seed_all(178)
np.random.seed(986)


def copy_files(src_dir, dest_dir, file_extension):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            # check file type
            if src_file_path.endswith(file_extension):
                dest_file_path = os.path.join(dest_dir, file)
                # copy
                shutil.copy(src_file_path, dest_file_path)
                print(f"copying {src_file_path} to {dest_file_path}")


def get_tv(im_tensor):
    diff1 = im_tensor[:, :, :, :-1] - im_tensor[:, :, :, 1:]
    diff2 = im_tensor[:, :, :-1, :] - im_tensor[:, :, 1:, :]
    diff3 = im_tensor[:, :, 1:, :-1] - im_tensor[:, :, :-1, 1:]
    diff4 = im_tensor[:, :, :-1, :-1] - im_tensor[:, :, 1:, 1:]
    tv = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return tv


def entropy_t4d(images):
    B, C, _, _ = images.shape
    entropies = torch.zeros(B)
    for i in range(B):
        image = images[i]
        image_flat = image.view(C, -1)
        hist = torch.histc(image_flat, bins=256, min=0, max=1)
        prob = hist / torch.sum(hist)
        prob = prob[prob > 0]
        entropies[i] = -torch.sum(prob * torch.log2(prob))
    
    return entropies


def get_text_feature(prompts, clip_model, device, input_token=False):
    with torch.no_grad():
        if input_token:
                text_feature = clip_model.encode_text(prompts)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        else:
            prompts = [prompts] if not isinstance(prompts, list) else prompts
            with torch.no_grad():
                text_feature = clip_model.encode_text(clip.tokenize(prompts).to(device))
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

    return text_feature    


class Trainer(object):
    def __init__(self,
                args):
        super().__init__()
        self.args = args
        self.if_resume = False

        # device
        self.cuda = torch.cuda.is_available()
        self.dev = 'cuda:'+str(GLOBAL_RANK) if self.cuda else 'cpu'

        # clip config
        self.clip_model, _ = clip.load("ViT-B/32")
        self.vgg = VGG()

        vi_text = ['a visible gray image']
        ir_text = ['an infrared image']

        self.prompt = ['high resolution, salient objects, rich details']
        self.antonym = ['low resolution, indistinct objects, minimal details']
        print(f'prompt: {self.prompt}')

        with torch.no_grad():
            self.vit_dir = get_text_feature(vi_text, self.clip_model, self.dev, False)
            self.irt_dir = get_text_feature(ir_text, self.clip_model, self.dev, False)
            self.prompt_dir = get_text_feature(self.prompt, self.clip_model, self.dev, False)
            self.antonym_dir = get_text_feature(self.antonym, self.clip_model, self.dev, False)

        # Initialize encoders, generators
        self.enc = Encoder(2, 32, 3, True)
        self.mod = Modal(2, 64, 16, True)
        self.dec = Decoder(1, 32*(2**4), 4, True, 16)

        # Initialize weights
        if self.if_resume:
            self.load(load_epoch=100, pt_path='./pretrained_weight_path/')

        if self.cuda:
            self.enc.cuda()
            self.mod.cuda()
            self.dec.cuda()
            self.clip_model.cuda()
            self.vgg.cuda()

        if MUL_GPU:
            self.enc = DDP(self.enc, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
            self.mod = DDP(self.mod, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
            self.dec = DDP(self.dec, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

        self.step = 2
        self.prompt_weight_g = 2
        self.prompt_weight_p = 20
        self.de_gain = 1
        self.pp_weight = 17
        self.patch_select = 5.9
        self.data_len = [40000]

        # Optimizers
        self.optimizer_G = Adam(itertools.chain(self.enc.parameters(),
                                                self.mod.parameters(),
                                                self.dec.parameters()),
                                lr=args.lr, 
                                betas=args.betas)

        # Learning rate update schedulers
        if self.args.if_warm_up:
            self.warm_lr_G = LambdaLR(self.optimizer_G, lr_lambda=lambda x:(x+1)/(sum(self.data_len)/self.args.train_batch/torch.cuda.device_count()))
        self.lr_scheduler_G = MultiStepLR(self.optimizer_G, 
                                            milestones=self.args.lr_mstone, 
                                            gamma=self.args.lr_decay_gamma)

        if GLOBAL_RANK == 0:                
            self.results_folder = Path(self.args.save_dir)
            if not os.path.exists(self.results_folder):
                os.makedirs(self.results_folder)
            copy_files("./src_fusion", args.save_dir+"/src_fusion", ".py")


    def dynamic_cropper(self, vi, ir, fusion, num_crops=8):
        transform = transforms.Compose([transforms.RandomCrop(int(random.uniform(48,80))),
                                        transforms.RandomAffine(degrees=[-3,3],
                                                                shear=[-3,3,-3,3]),
                                        transforms.Resize(224)])  
        vi = vi.repeat(1,3,1,1) if vi.shape[1] == 1 else vi
        ir = ir.repeat(1,3,1,1) if ir.shape[1] == 1 else ir
        fusion = fusion.repeat(1,3,1,1) if fusion.shape[1] == 1 else fusion

        vi_cropped = []
        ir_cropped = []
        fu_cropped = []

        for _ in range(num_crops):
            catted = torch.cat([vi, ir, fusion], dim=0)
            transed = transform(catted)
            chunked = torch.chunk(transed, 3, dim=0)
            vi_cropped.append(chunked[0])
            ir_cropped.append(chunked[1])
            fu_cropped.append(chunked[2])        

        return torch.cat(vi_cropped, dim=0), torch.cat(ir_cropped, dim=0), torch.cat(fu_cropped, dim=0)


    def clip_norm(self, im):      
        DEV = im.device   
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        im_re = F.interpolate(im.repeat(1,3,1,1) if im.shape[1]==1 else im, size=224, mode='bilinear', align_corners=False)
        im_norm = (im_re - mean) / std
        return im_norm


    def clip_prompt_loss(self, fu_feature, vi_feature, ir_feature, prompt_dir, select_vi=None, select_ir=None):
        BP = fu_feature.shape[0]
        BG = prompt_dir.shape[0]

        prompt_dir = prompt_dir.repeat(BP//BG, 1)
        vi_feature = vi_feature.repeat(BP//BG, 1)
        ir_feature = ir_feature.repeat(BP//BG, 1)

        # ----- im feature -----
        vi_len = vi_feature.norm(dim=-1, keepdim=True)
        vi_dir = vi_feature / vi_len
        
        ir_len = ir_feature.norm(dim=-1, keepdim=True)
        ir_dir = ir_feature / ir_len

        fusion_len = fu_feature.norm(dim=-1, keepdim=True)
        fusion_dir = fu_feature / fusion_len

        # ----- im direction -----
        vi_fu = fusion_dir - vi_dir
        vi_fu_dir = vi_fu / vi_fu.norm(dim=-1, keepdim=True)

        ir_fu = fusion_dir - ir_dir
        ir_fu_dir = ir_fu / ir_fu.norm(dim=-1, keepdim=True)

        # ----- text direction -----
        vit_pt = prompt_dir - self.vit_dir.repeat(BP, 1)
        vit_pt_dir = vit_pt / vit_pt.norm(dim=-1, keepdim=True)

        irt_pt = prompt_dir - self.irt_dir.repeat(BP, 1)
        irt_pt_dir = irt_pt / irt_pt.norm(dim=-1, keepdim=True)

        # ----- calc loss -----
        select_vi = select_vi if select_vi is not None else 1
        select_ir = select_ir if select_ir is not None else 1
        
        loss_vi = 0.5 * (1 - torch.cosine_similarity(vi_fu_dir, vit_pt_dir, dim=1)) * select_vi
        loss_ir = 0.5 * (1 - torch.cosine_similarity(ir_fu_dir, irt_pt_dir, dim=1)) * select_ir

        return ((loss_ir+loss_vi).exp()-1).mean()


    def clip_antonym_loss(self, fu_feature, vi_feature, ir_feature, prompt_dir, select_vi=None, select_ir=None):
        BP = fu_feature.shape[0]
        BG = prompt_dir.shape[0]

        prompt_dir = prompt_dir.repeat(BP//BG, 1)
        vi_feature = vi_feature.repeat(BP//BG, 1)
        ir_feature = ir_feature.repeat(BP//BG, 1)

        # ----- im feature -----
        vi_len = vi_feature.norm(dim=-1, keepdim=True)
        vi_dir = vi_feature / vi_len
        
        ir_len = ir_feature.norm(dim=-1, keepdim=True)
        ir_dir = ir_feature / ir_len

        fusion_len = fu_feature.norm(dim=-1, keepdim=True)
        fusion_dir = fu_feature / fusion_len

        # ----- im direction -----
        vi_fu = fusion_dir - vi_dir
        vi_fu_dir = vi_fu / vi_fu.norm(dim=-1, keepdim=True)

        ir_fu = fusion_dir - ir_dir
        ir_fu_dir = ir_fu / ir_fu.norm(dim=-1, keepdim=True)

        # ----- text direction -----
        vit_pt = prompt_dir - self.vit_dir.repeat(BP, 1)
        vit_pt_dir = vit_pt / vit_pt.norm(dim=-1, keepdim=True)

        irt_pt = prompt_dir - self.irt_dir.repeat(BP, 1)
        irt_pt_dir = irt_pt / irt_pt.norm(dim=-1, keepdim=True)

        # ----- calc loss -----
        select_vi = select_vi if select_vi is not None else 1
        select_ir = select_ir if select_ir is not None else 1

        loss_vi = 0.5 * (1 + torch.cosine_similarity(vi_fu_dir, vit_pt_dir, dim=1)) * select_vi
        loss_ir = 0.5 * (1 + torch.cosine_similarity(ir_fu_dir, irt_pt_dir, dim=1)) * select_ir

        return ((loss_vi+loss_ir).exp()-1).mean()


    def feat_loss(self, fu, vi, ir):
        fu_vgg = self.vgg.get_features(fu)
        vi_vgg = self.vgg.get_features(vi)
        ir_vgg = self.vgg.get_features(ir)

        feat1 = torch.cat([vi_vgg['conv1_1'].unsqueeze(2), ir_vgg['conv1_1'].unsqueeze(2)], dim=2)
        feat1 = torch.max(feat1, dim=2, keepdim=False)[0]

        feat2 = torch.cat([vi_vgg['conv2_1'].unsqueeze(2), ir_vgg['conv2_1'].unsqueeze(2)], dim=2)
        feat2 = torch.max(feat2, dim=2, keepdim=False)[0]

        feat3 = torch.cat([vi_vgg['conv3_1'].unsqueeze(2), ir_vgg['conv3_1'].unsqueeze(2)], dim=2)
        feat3 = torch.max(feat3, dim=2, keepdim=False)[0]

        feat4 = torch.cat([vi_vgg['conv4_2'].unsqueeze(2), ir_vgg['conv4_2'].unsqueeze(2)], dim=2)
        feat4 = torch.max(feat4, dim=2, keepdim=False)[0]

        loss_1 = F.mse_loss(fu_vgg['conv1_1'], feat1, reduction='mean')
        loss_2 = F.mse_loss(fu_vgg['conv2_1'], feat2, reduction='mean')
        loss_3 = F.mse_loss(fu_vgg['conv3_1'], feat3, reduction='mean')
        loss_4 = F.mse_loss(fu_vgg['conv4_2'], feat4, reduction='mean')

        return (loss_3 + loss_4) / 2


    def train(self):
        for self.epoch in range(self.args.epochs):
            train_dataset = IVTData(self.prompt[0], self.antonym[0], sample_n=self.data_len)            
            if MUL_GPU:
                train_sampler = DistributedSampler(dataset=train_dataset, 
                                                shuffle=True, 
                                                drop_last=True)
                train_dataloader = DataLoader(dataset=train_dataset, 
                                                batch_size=self.args.train_batch, 
                                                sampler=train_sampler,
                                                num_workers=self.args.num_workers,
                                                pin_memory=True,
                                                drop_last=True)
            else:
                train_dataloader = DataLoader(dataset=train_dataset, 
                                                batch_size=self.args.train_batch, 
                                                shuffle=True,
                                                num_workers=self.args.num_workers,
                                                pin_memory=True,
                                                drop_last=True)            

            current_time = time.strftime('%y%m%d@%H:%M:%S')
            if GLOBAL_RANK == 0:
                print('=== Epoch {:5d} / {:5d} | Lr : {:.4e} | {} | {} ==='
                    .format(self.epoch, self.args.epochs, self.optimizer_G.param_groups[0]['lr'], current_time, self.args.save_dir))
                
            tqdm_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if GLOBAL_RANK == 0 else enumerate(train_dataloader)

            for _, sample in tqdm_bar:   
                # Set model input
                X1 = sample['vi_y'].to(device=self.dev)
                X2 = sample['ir_y'].to(device=self.dev)
                X1_N = sample['vi_n'].to(device=self.dev)
                X2_N = sample['ir_n'].to(device=self.dev)

                ET = sample['tk_t'].to(device=self.dev)
                ED = get_text_feature(ET, self.clip_model, self.dev, True)
                AET = sample['tk_a'].to(device=self.dev)
                AED = get_text_feature(AET, self.clip_model, self.dev, True)
                PD = self.prompt_dir.repeat(self.args.train_batch, 1).to(device=self.dev)
                APD = self.antonym_dir.repeat(self.args.train_batch, 1).to(device=self.dev)
                
                _, _, H, W = X1.shape

                self.optimizer_G.zero_grad()

                fusion_y = self.dec(self.enc(X1_N, X2_N), self.mod(X1_N, X2_N))
                fusion_y = fusion_y[:,:,:H,:W]

                vi_patch, ir_patch, fu_patch = self.dynamic_cropper(X1, X2, fusion_y, int(32))

                valid_vi = 1 * (entropy_t4d(vi_patch) > self.patch_select).to(vi_patch.device)
                valid_ir = 1 * (entropy_t4d(ir_patch) > self.patch_select).to(ir_patch.device)

                vi_feature = self.clip_model.encode_image(self.clip_norm(X1))
                ir_feature = self.clip_model.encode_image(self.clip_norm(X2))
                fu_feature = self.clip_model.encode_image(self.clip_norm(fusion_y))
                fu_patch_feature = self.clip_model.encode_image(self.clip_norm(fu_patch))

                loss_cg = self.prompt_weight_g * self.clip_prompt_loss(fu_feature, vi_feature, ir_feature, PD) + \
                        self.de_gain * self.prompt_weight_g * self.clip_prompt_loss(fu_feature, vi_feature, ir_feature, ED)
                loss_cp = self.prompt_weight_p * self.clip_prompt_loss(fu_patch_feature, vi_feature, ir_feature, PD, valid_vi, valid_ir) + \
                        self.de_gain * self.prompt_weight_p * self.clip_prompt_loss(fu_patch_feature, vi_feature, ir_feature, ED, valid_vi, valid_ir)

                loss_cag = self.prompt_weight_g * self.clip_antonym_loss(fu_feature, vi_feature, ir_feature, APD) + \
                        self.de_gain * self.prompt_weight_g * self.clip_antonym_loss(fu_feature, vi_feature, ir_feature, AED)
                loss_cap = self.prompt_weight_p * self.clip_antonym_loss(fu_patch_feature, vi_feature, ir_feature, APD, valid_vi, valid_ir) + \
                        self.de_gain * self.prompt_weight_p * self.clip_antonym_loss(fu_patch_feature, vi_feature, ir_feature, AED, valid_vi, valid_ir)

                loss_ff = self.pp_weight * self.feat_loss(fusion_y, X1, X2)

                loss = loss_cg + loss_cp + loss_cag + loss_cap + loss_ff

                loss.backward()
                self.optimizer_G.step() 

                # tqdm update
                if GLOBAL_RANK == 0:
                    current_lr = self.optimizer_G.param_groups[0]['lr']
                    s = f'Train | Lr: {current_lr:.2e} | loss:{loss:.2f} | tv:{0.002*get_tv(fusion_y):.2f} | '
                    tqdm_bar.set_description(s)

                if self.args.if_warm_up and self.epoch == 0:
                    self.warm_lr_G.step() 
                    
            self.lr_scheduler_G.step()
            
            if GLOBAL_RANK == 0 and self.epoch % self.step == 0:
                self.save()


    def save(self):
        torch.save(self.enc.module.state_dict() if MUL_GPU else self.enc.state_dict(), '{}/enc_{:05d}.pt'.format(self.args.save_dir, self.epoch))
        torch.save(self.mod.module.state_dict() if MUL_GPU else self.mod.state_dict(), '{}/mod_{:05d}.pt'.format(self.args.save_dir, self.epoch))
        torch.save(self.dec.module.state_dict() if MUL_GPU else self.dec.state_dict(), '{}/dec_{:05d}.pt'.format(self.args.save_dir, self.epoch))         


    def load(self, load_epoch, pt_path):
        dev = 'cuda:'+str(GLOBAL_RANK) if torch.cuda.is_available() else 'cpu'

        enc_w = torch.load(str(pt_path + f'/enc_{load_epoch:05d}.pt'), map_location=dev)
        self.enc.load_state_dict(enc_w)

        mod_w = torch.load(str(pt_path + f'/mod_{load_epoch:05d}.pt'), map_location=dev)
        self.mod.load_state_dict(mod_w)

        dec_w = torch.load(str(pt_path + f'/dec_{load_epoch:05d}.pt'), map_location=dev)
        self.dec.load_state_dict(dec_w)

        print('Pretrained weight load done !')


if __name__ == '__main__':
    MUL_GPU = False if torch.cuda.device_count() <= 1 else True
    # MUL_GPU = False

    print('MUL_GPU == ', MUL_GPU)

    if MUL_GPU:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args_config.local_rank)
        GLOBAL_RANK = dist.get_rank()
    else:
        GLOBAL_RANK = 0

    # config
    args = args_config

    if GLOBAL_RANK == 0:
        print('\n\n=== Arguments ===')
        cnt = 0
        for key in sorted(vars(args)):
            print(key, ':',  getattr(args, key), end='  |  ')
            cnt += 1
            if (cnt + 1) % 5 == 0:
                print('')
        print('\n')

    trainer = Trainer(args)
    trainer.train()
