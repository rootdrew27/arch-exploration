import torch
import torch.nn as nn
import torch.nn.functional as F
from ..heads.yolo import YOLO8Head
from ..backbones.yolo import YOLO8Backbone

class RefNet(nn.Module):
    def __init__(self, nc=80):
        super(RefNet, self).__init__()
        self.backbone = YOLO8Backbone()
        self.detect_head = YOLO8Head(nc)
        
    def forward(self, x, x_ref):
        p3, p4, p5 = self.backbone(x)
        p3_2, p4_2, p5_2 = self.backbone(x_ref)
        
        p3, p4, p5 = p3 - p3_2, p4 - p4_2, p5 - p5_2
        
        y = self.detect_head(p3, p4, p5)
        return y
    

class YOLOv8(nn.Module):
    def __init__(self, nc=1):
        super(YOLOv8, self).__init__()
        self.backbone = YOLO8Backbone()
        self.head = YOLO8Head(nc)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
        