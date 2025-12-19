import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class Encoder(nn.Module):
    def __init__(self, in_channels=2, dim=32, n_sample=3, if_init=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, dim, 7, 1, 3),
                nn.InstanceNorm2d(dim),
                nn.ReLU(inplace=True)]

        for _ in range(n_sample):
            layers += [nn.Conv2d(dim, dim*2, 3, 2, 1),
                    nn.InstanceNorm2d(dim*2),
                    nn.ReLU(inplace=True)]
            dim *= 2

        layers += [nn.Conv2d(dim, dim*2, 3, 1, 1),
                nn.InstanceNorm2d(dim*2),
                nn.ReLU(inplace=True),
                Residual(dim*2, 'in')]

        self.model = nn.Sequential(*layers)

        if if_init:
            self.apply(weights_init_normal)

    def forward(self, x1, x2):
        return self.model(torch.cat([x1, x2], dim=1))


class Modal(nn.Module):
    def __init__(self, in_channels=2, dim=32, modal_dim=16, if_init=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, in_channels, 5, 2, 2), # downsample 
                nn.Conv2d(in_channels, dim, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim*2, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(dim*2, dim*4, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(dim*4, dim*2, 3, 2, 1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d([1,1]), nn.Conv2d(dim*2, modal_dim, 1, 1, 0)]

        self.model = nn.Sequential(*layers)

        if if_init:
            self.apply(weights_init_normal)
            
    def forward(self, x1, x2):
        x1 = F.interpolate(x1, (160, 160))
        x2 = F.interpolate(x2, (160, 160))
        return self.model(torch.cat([x1, x2], dim=1))
    

class Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, n_sample=3, if_init=True, modal_dim=16, pm=None):
        super().__init__()
        layers = [Residual(dim, 'adain', pm)]

        for _ in range(n_sample):
            layers += [nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(dim, dim//2, 5, 1, 2, **({'padding_mode': pm} if pm is not None else {})),
                    AdaptiveInstanceNorm2d(dim//2),
                    nn.ReLU(inplace=True)]
            dim = dim // 2

        layers += [nn.Conv2d(dim, out_channels, 7, 1, 3, **({'padding_mode': pm} if pm is not None else {})), 
                nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(modal_dim, num_adain_params)

        if if_init:
            self.apply(weights_init_normal)

    def get_num_adain_params(self):
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]

                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)

                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, x, m):      
        self.assign_adain_params(self.mlp(m))
        return self.model(x)


class Residual(nn.Module):
    def __init__(self, features, norm="adain", pm=None):
        super().__init__()
        if norm == "adain":
            norm_layer = AdaptiveInstanceNorm2d
        elif norm == "ln":
            norm_layer = LayerNorm
        elif norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "in":
            norm_layer = nn.InstanceNorm2d
        else:
            print(f"no {norm}")

        self.block = nn.Sequential(nn.Conv2d(features, features, 5, 1, 2, **({'padding_mode': pm} if pm is not None else {})),
                                norm_layer(features),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        return x + self.block(x)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = None
        self.bias = None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (self.weight is not None and self.bias is not None), "Please assign weight and bias before calling AdaIN."
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, h, w)
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True)
        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"
    

class LayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer_norm = nn.GroupNorm(1, num_features)

    def forward(self, x):
        return self.layer_norm(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_layer=1):
        super().__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_layer):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


if __name__ == "__main__":
    e = Encoder(2, 32, 3, True)
    m = Modal(2, 64, 16, True)
    d = Decoder(1, 32*(2**4), 4, True, 16)

    shape = 333
    input_tensor = torch.randn(1, 1, shape, shape)
    fe = e(input_tensor, input_tensor)
    fm = m(input_tensor, input_tensor)
    fu = d(fe, fm)
    print("Output shape:", fu.shape)

    total_params = sum(p.numel() for p in e.parameters())
    print(f"Total number of parameters: {total_params}")

    total_params = sum(p.numel() for p in m.parameters())
    print(f"Total number of parameters: {total_params}")

    total_params = sum(p.numel() for p in d.parameters())
    print(f"Total number of parameters: {total_params}")

    # mod = Modal(2, 64, 8, if_init=True)
    # shape = 160
    # input_tensor = torch.randn(1, 2, shape, shape)
    # print("Output shape:", mod(input_tensor).shape)
