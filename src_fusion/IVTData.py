import os
import json
import torch
import clip
import random
# import nltk
import warnings
import torch.nn as nn
# from nltk import pos_tag
# from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io.image import read_image, ImageReadMode


warnings.filterwarnings("ignore")

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('punkt_tab')


# def get_sentence(word_list):
#     word_dic = {i: pos_tag(word_tokenize(i.split()[-1]))[0][1] for i in word_list}

#     nouns = [word for word, tag in word_dic.items() if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
#     if len(nouns) >= 1:
#         nouns[0] = "with " + nouns[0]
#     adjectives = [word for word, tag in word_dic.items() if tag in ['JJ', 'JJR', 'JJS']]

#     gen_sentence = ','.join(adjectives) + ' image ' + ','.join(nouns)

#     return gen_sentence


class IVTData(Dataset):
    def __init__(self,
                text,
                antonym,
                im_h=336, 
                im_w=336, 
                data_dir=["./training_samples"],
                sample_n=[40000],
                dae=0.03):
        self.text = text
        self.antonym = antonym
        self.im_h = im_h
        self.im_w = im_w
        self.sample_n = sample_n
        self.dae = dae

        ir_folder = ["ir"]       
        vis_folder = ["vi"]
        text_folder = ["prompt"]

        self.ir_file_list = []
        self.vis_file_list = []
        self.text_file_list = []

        self.resample = transforms.Resize((self.im_h//2, self.im_w//2)) 

        for ind, ins in enumerate(data_dir):
            ir_dir = os.path.join(ins, ir_folder[ind])
            vis_dir = os.path.join(ins, vis_folder[ind])
            text_dir = os.path.join(ins, text_folder[ind])

            file_ls = os.listdir(ir_dir)
            sampled_files = random.sample(file_ls, self.sample_n[ind])

            self.ir_file_list.extend([os.path.join(ir_dir, i) for i in sampled_files])
            self.vis_file_list.extend([os.path.join(vis_dir, i) for i in sampled_files])
            self.text_file_list.extend([os.path.join(text_dir, os.path.splitext(i)[0]+'.json') for i in sampled_files])

    def __len__(self):
        return len(self.ir_file_list)
    
    def __getitem__(self, idx):  
        vi_1 = read_image(self.vis_file_list[idx], ImageReadMode.GRAY)
        ir_1 = read_image(self.ir_file_list[idx], ImageReadMode.GRAY) 
        vi_1, ir_1 = self.augment(vi_1, ir_1)
        
        vi_1 = vi_1 / 255.
        ir_1 = ir_1 / 255.

        with open(self.text_file_list[idx], 'r') as f:
            text = json.load(f)
        tk_t, tk_a = self.get_token(text)

        ir_n = self.resample(ir_1)
        vi_n = self.resample(vi_1)
        
        return {"ir_y": ir_1,
                "ir_n": ir_n+self.dae*torch.randn_like(ir_n),
                "vi_y": vi_1, 
                "vi_n": vi_n+self.dae*torch.randn_like(vi_n), 
                "tk_t": tk_t,
                "tk_a": tk_a}
    
    def augment(self, vi, ir):     
        transform = transforms.Compose([transforms.Resize((self.im_h, self.im_w)),
                                        transforms.RandomHorizontalFlip(0.5)])
        
        vi_ir = torch.cat([vi, ir], dim=0)        
        vi_ir_t = transform(vi_ir)
        vi_t, ir_t = torch.split(vi_ir_t, [1, 1], dim=0)

        return vi_t, ir_t

    
    # def get_token(self, text, if_sentence=False):
    #     if text == None or len(text) == 0:
    #         tk_t = clip.tokenize([self.text])
    #         tk_a = clip.tokenize([self.antonym])
    #     else:
    #         if if_sentence:
    #             t = list(text.values())
    #             at = list(text.keys())
    #             t = get_sentence(t)
    #             at = get_sentence(at)
    #         else:
    #             t = ",".join(text.values())
    #             at = ",".join(text.keys())

    #         tk_t = clip.tokenize([t])
    #         tk_a = clip.tokenize([at])

    #     return tk_t.squeeze(0), tk_a.squeeze(0)


    def get_token(self, text):
        if text == None or len(text) == 0:
            tk_t = clip.tokenize([self.text])
            tk_a = clip.tokenize([self.antonym])
        else:
            t = ",".join(text.values())
            tk_t = clip.tokenize([t])

            at = ",".join(text.keys())
            tk_a = clip.tokenize([at])

        return tk_t.squeeze(0), tk_a.squeeze(0)
    

if __name__ == '__main__':
    DEV = "cuda:0"
    dataset = IVTData('an image with salient objects and detailed background',
                    'an image without salient objects and detailed background.',
                    im_h=336,
                    im_w=336)
    for idx, ins in enumerate(dataset):   
        print(f"=== text ====: {ins['tk_t'].shape}")
        print("*"*20, idx, "*"*20)

        
