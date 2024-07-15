import torch
import torch.nn as nn
import torch.nn.functional as F
from ..components.comps import ConvBlock, C2f, Bottleneck, SPPF, Concat, Detect

class YOLO8Head(nn.Module):
    def __init__(self, nc=80):
        super(YOLO8Head, self).__init__()
        
        self.upsample_10 = nn.Upsample(None, 2, "nearest")
        self.concat_11 = Concat(1) 
        self.c2f_12 = C2f(1024, 512, 3)
        
        # p3
        self.upsample_13 = nn.Upsample(None, 2, "nearest")
        self.concat_14 = Concat(1)     
        self.c2f_15 = C2f(768, 256, 3)
        
        # p5 
        self.conv_16 = ConvBlock(256, 256, 3, 2)
        self.concat_17 = Concat(1)
        self.c2f_18 = C2f(768, 512, 3)
        
        # 
        self.conv_19 = ConvBlock(512, 512, 3, 2)
        self.concat_20 = Concat(1)
        self.c2f_21 = C2f(1024, 512, 3)
        
        self.detection = Detect(nc, [256, 512, 512])
        
    def forward(self, x1, x2, x3):
        i1 = self.c2f_12(self.concat_11((self.upsample_10(x3), x2))) # B, 512, H, W
        i2 = self.c2f_15(self.concat_14((self.upsample_13(i1), x1))) # B, 256, H, W
        i3 = self.c2f_18(self.concat_17((self.conv_16(i2), i1))) # B, 512, H, W
        i4 = self.c2f_21(self.concat_20((self.conv_19(i3), x3))) # B, 512, H, W
        
        return self.detection([i2, i3, i4]) # output tensors
        
        
    