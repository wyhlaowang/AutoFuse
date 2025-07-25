import time
import os
import itertools
import clip
import random
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm
from model import *
from config import args as args_config
from IVTData import IVTData
from vgg import VGG


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
        self.cuda = torch.cuda.is_available()
        self.dev = 'cuda:'+str(GLOBAL_RANK) if self.cuda else 'cpu'

        self.clip_model, _ = clip.load("ViT-B/32")
        self.vgg = VGG()

        vi_text = ['a visible gray image']
        ir_text = ['an infrared image']

        self.prompt = ['high resolution, salient objects, rich details']
        self.antonym = ['low resolution, indistinct objects, minimal details']

        with torch.no_grad():
            self.vit_dir = get_text_feature(vi_text, self.clip_model, self.dev, False)
            self.irt_dir = get_text_feature(ir_text, self.clip_model, self.dev, False)
            self.prompt_dir = get_text_feature(self.prompt, self.clip_model, self.dev, False)
            self.antonym_dir = get_text_feature(self.antonym, self.clip_model, self.dev, False)

        # Initialize encoders, generators
        self.enc = Encoder(2, 32, 3, True)
        self.mod = Modal(2, 64, 16, True)
        self.dec = Decoder(1, 32*(2**4), 4, True, 16)

        if self.cuda:
            self.enc.cuda()
            self.mod.cuda()
            self.dec.cuda()
            self.clip_model.cuda()
            self.vgg.cuda()

        self.enc = DDP(self.enc, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        self.mod = DDP(self.mod, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        self.dec = DDP(self.dec, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

        # Optimizers
        self.optimizer_G = Adam(itertools.chain(self.enc.parameters(),
                                                self.mod.parameters(),
                                                self.dec.parameters()),
                                lr=args.lr, 
                                betas=args.betas)
        self.tp = 0.5
        self.lg = 0.2
        self.patch_select = 6
        self.data_len = [40000]

        # Learning rate update schedulers
        if self.args.if_warm_up:
            self.warm_lr_G = LambdaLR(self.optimizer_G, lr_lambda=lambda x:(x+1)/(sum(self.data_len)/self.args.train_batch/torch.cuda.device_count()))
        self.lr_scheduler_G = MultiStepLR(self.optimizer_G, 
                                            milestones=self.args.lr_mstone, 
                                            gamma=self.args.lr_decay_gamma)
                        
        self.results_folder = Path(self.args.save_dir)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

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


    def embed_sim(self, fu_feature, vi_feature, ir_feature, prompt_dir, select_vi=None, select_ir=None):
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

        sim_vi = torch.cosine_similarity(vi_fu_dir, vit_pt_dir, dim=1) * select_vi
        sim_ir = torch.cosine_similarity(ir_fu_dir, irt_pt_dir, dim=1) * select_ir

        return sim_vi + sim_ir

    def contrast_loss(self, pos_sim, neg_sim):
        core = (neg_sim - pos_sim) / self.tp
        loss = (core.exp() + 1).log()
        return loss.mean()


    def vgg_loss(self, fu, vi, ir):
        fu_vgg = self.vgg.get_features(fu)
        vi_vgg = self.vgg.get_features(vi)
        ir_vgg = self.vgg.get_features(ir)

        feat2 = torch.cat([vi_vgg['conv2_1'].unsqueeze(2), ir_vgg['conv2_1'].unsqueeze(2)], dim=2)
        feat2 = torch.max(feat2, dim=2, keepdim=False)[0]

        feat3 = torch.cat([vi_vgg['conv3_1'].unsqueeze(2), ir_vgg['conv3_1'].unsqueeze(2)], dim=2)
        feat3 = torch.max(feat3, dim=2, keepdim=False)[0]

        feat4 = torch.cat([vi_vgg['conv4_2'].unsqueeze(2), ir_vgg['conv4_2'].unsqueeze(2)], dim=2)
        feat4 = torch.max(feat4, dim=2, keepdim=False)[0]

        loss_2 = F.mse_loss(fu_vgg['conv2_1'], feat2, reduction='mean')
        loss_3 = F.mse_loss(fu_vgg['conv3_1'], feat3, reduction='mean')
        loss_4 = F.mse_loss(fu_vgg['conv4_2'], feat4, reduction='mean')

        return loss_2 + loss_3 + loss_4


    def train(self):
        for self.epoch in range(self.args.epochs):
            train_dataset = IVTData(self.prompt[0], self.antonym[0], sample_n=self.data_len)            
            train_sampler = DistributedSampler(dataset=train_dataset, 
                                            shuffle=True, 
                                            drop_last=True)
            train_dataloader = DataLoader(dataset=train_dataset, 
                                            batch_size=self.args.train_batch, 
                                            sampler=train_sampler,
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
                
                # image level
                loss_g_static = self.lg * self.contrast_loss(self.embed_sim(fu_feature, vi_feature, ir_feature, PD), self.embed_sim(fu_feature, vi_feature, ir_feature, APD))
                loss_g_dynamic = self.lg * self.contrast_loss(self.embed_sim(fu_feature, vi_feature, ir_feature, ED), self.embed_sim(fu_feature, vi_feature, ir_feature, AED))

                # patch level
                loss_p_static = self.contrast_loss(self.embed_sim(fu_patch_feature, vi_feature, ir_feature, PD, valid_vi, valid_ir), self.embed_sim(fu_patch_feature, vi_feature, ir_feature, APD, valid_vi, valid_ir))
                loss_p_dynamic = self.contrast_loss(self.embed_sim(fu_patch_feature, vi_feature, ir_feature, ED, valid_vi, valid_ir), self.embed_sim(fu_patch_feature, vi_feature, ir_feature, AED, valid_vi, valid_ir))

                loss_ff = self.vgg_loss(fusion_y, X1, X2)

                loss = loss_g_static + loss_g_dynamic + loss_p_static + loss_p_dynamic + loss_ff

                loss.backward()
                self.optimizer_G.step() 

                # tqdm update
                if GLOBAL_RANK == 0:
                    current_lr = self.optimizer_G.param_groups[0]['lr']
                    s = f'Train | Lr: {current_lr:.2e} | loss:{loss:.2f} |'
                    tqdm_bar.set_description(s)

                if self.args.if_warm_up and self.epoch == 0:
                    self.warm_lr_G.step() 
                    
            self.lr_scheduler_G.step()
            if GLOBAL_RANK == 0:
                self.save()


    def save(self):
        torch.save(self.enc.module.state_dict(), '{}/enc_{:05d}.pt'.format(self.args.save_dir, self.epoch))
        torch.save(self.mod.module.state_dict(), '{}/mod_{:05d}.pt'.format(self.args.save_dir, self.epoch))
        torch.save(self.dec.module.state_dict(), '{}/dec_{:05d}.pt'.format(self.args.save_dir, self.epoch))               


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
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args_config.local_rank)
    GLOBAL_RANK = dist.get_rank()
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

