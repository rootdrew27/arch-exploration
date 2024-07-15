import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.comps import ConvBlock, C2f, SPPF

class YOLO8Backbone(nn.Module):
    def __init__(self):
        super(YOLO8Backbone, self).__init__()
        self.conv_1 = ConvBlock(3, 64, 3, 2) 
        self.conv_2 = ConvBlock(64, 128, 3, 2)
        self.c2f_1 = C2f(128, 128, 3, shortcut=True) 
        self.conv_3 = ConvBlock(128, 256, 3, 2)
        self.c2f_2 = C2f(256, 256, 6, shortcut=True) # P3
        self.conv_4 = ConvBlock(256, 512, 3, 2)
        self.c2f_3 = C2f(512, 512, 6, shortcut=True) # p4
        self.conv_5 = ConvBlock(512, 512, 3, 2) 
        self.c2f_4 = C2f(512, 512, 3, shortcut=True)
        self.sppf = SPPF(512, 512, 5)                # p5
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.c2f_1(x)
        x = self.conv_3(x)
        p3 = self.c2f_2(x)
        x = self.conv_4(x)
        p4 = self.c2f_3(x)
        x = self.conv_5(x)
        x = self.c2f_4(x)
        p5 = self.sppf(x)
        return p3, p4, p5 # channel dim = (256, 512, 512)