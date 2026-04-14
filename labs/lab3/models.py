import torch
import torch.nn as nn
import numpy as np
from skimage.color import lab2rgb


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False, activation="relu"):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        else:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout(0.5))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

    
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = UNetBlock(1,   64,  down=True, use_bn=False, activation="leaky")  # 128
        self.e2 = UNetBlock(64,  128, down=True, activation="leaky")                # 64
        self.e3 = UNetBlock(128, 256, down=True, activation="leaky")                # 32
        self.e4 = UNetBlock(256, 512, down=True, activation="leaky")                # 16
        self.e5 = UNetBlock(512, 512, down=True, activation="leaky")                # 8
        self.e6 = UNetBlock(512, 512, down=True, activation="leaky")                # 4
        self.e7 = UNetBlock(512, 512, down=True, activation="leaky")                # 2
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder — in_ch doubles because of skip connections
        self.d1 = UNetBlock(512,  512, down=False, dropout=True)   # 2
        self.d2 = UNetBlock(1024, 512, down=False, dropout=True)   # 4
        self.d3 = UNetBlock(1024, 512, down=False, dropout=True)   # 8
        self.d4 = UNetBlock(1024, 512, down=False)                 # 16
        self.d5 = UNetBlock(1024, 256, down=False)                 # 32
        self.d6 = UNetBlock(512,  128, down=False)                 # 64
        self.d7 = UNetBlock(256,  64,  down=False)                 # 128
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        e1 = self.e1(x)  
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        bn = self.bottleneck(e7)
        d1 = self.d1(bn)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        final = self.final(torch.cat([d7, e1], dim=1))
        
        return final

    
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # input = L (1ch) + ab (2ch) = 3ch
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),
        )
        
    def forward(self, L, ab):
        return self.model(torch.cat([L, ab], dim=1))

    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = PatchGANDiscriminator()  # full scale
        self.d2 = PatchGANDiscriminator()  # half scale
        self.d3 = PatchGANDiscriminator()  # quarter scale
        self.downsample = nn.AvgPool2d(2)

    def forward(self, L, ab):
        L2  = self.downsample(L)
        ab2 = self.downsample(ab)
        L4  = self.downsample(L2)
        ab4 = self.downsample(ab2)
        return [
            self.d1(L, ab),
            self.d2(L2, ab2),
            self.d3(L4, ab4)
        ]


def lab_to_rgb_numpy(L, ab):
    """Convert LAB tensors to RGB numpy array."""
    L_np  = (L.squeeze().cpu().numpy() + 1.0) * 50.0
    ab_np = ab.permute(1, 2, 0).cpu().numpy() * 110.0
    lab   = np.concatenate([L_np[:, :, None], ab_np], axis=2)
    return np.clip(lab2rgb(lab.astype(np.float64)), 0, 1)


def prepare_L(image_path, base_transform, device):
    """Prepare grayscale L channel from image for model input."""
    from skimage.color import rgb2lab
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = base_transform(img)

    img_np = np.array(img, dtype=np.float32) / 255.0
    lab = rgb2lab(img_np).astype(np.float32)

    L = torch.from_numpy(lab[:, :, 0] / 50.0 - 1.0)\
            .unsqueeze(0).unsqueeze(0).to(device)

    return (img, L)
