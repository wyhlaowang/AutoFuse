import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG = models.vgg19(pretrained=True).features
        for parameter in self.VGG.parameters():
            parameter.requires_grad_(False)

    def get_features(self, image):
        image_re = image.repeat(1,3,1,1) if image.shape[1] == 1 else image

        vgg_convs = {'0': 'conv1_1',
                     '5': 'conv2_1',
                     '10': 'conv3_1',
                     '19': 'conv4_1',
                     '21': 'conv4_2',
                     '28': 'conv5_1',
                     '31': 'conv5_2'}  
        
        features = {}
        x = image_re
        for name, layer in self.VGG._modules.items():
            x = layer(x)   
            if name in vgg_convs:
                features[vgg_convs[name]] = x
        
        return features



