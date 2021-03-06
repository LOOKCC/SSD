#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VGG16(nn.Module):
    def __init__(self, in_channels):
        super(VGG16, self).__init__()
        self.layers = self._make_layers(in_channels)

    def forward(self, x):
        return self.layers(x)
    
    def _make_layers(self, in_channels):
        # no bn? why?
        cfg = [64, 64, 'M', 128， 128，'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        temp_channels = in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(temp_channels, x, kernel_size=3, padding=1, nn.ReLU(True))]
            temp_channels = x
        return nn.Sequential(*layers)

class L2Norm(nn.Module):
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.parameter(torch.Tensor(in_features))
        self.reset_parameter(scale)
    
    def reset_parameter(self, scale):
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, : , None, None]
        return scale*x

class VGG16Extractor300(nn.Module):
    def __init__(self):
        super(VGG16Extractor300, self).__init__()

        self.features = VGG16(3)
        self.norm4 = L2Norm(512, 20)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

    def forward(self, x):
        hs = []
        h = self.features(x)
        hs.append(self.norm4(h)) # b 512 w h 

        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h) # b 1024 w h

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h) # b 512 w h 

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h) # b 256 w h

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h) # b 256 w h

        h = F.relu(self.conv11_1(h))
        h = F.relu(sel  f.conv11_2(h))
        hs.append(h) # b 256 w h 

        return hs

class SSD300(nn.Module):
    steps = (8, 16, 32, 64, 100, 300)
    box_sizes = (30, 60, 111, 162, 213, 264, 315)
    aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,))
    fm_sizes = (38, 19, 10, 5, 3, 1)

    def __init__(self, num_class):
        super(SSD300, self).__init__()
        self.num_classes = num_class
        self.num_anchors = (4, 4, 6, 6, 4, 4)
        self.in_channels = (512, 1024, 512, 256, 256, 256)

        self.extractor = VGG16Extractor300()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4, kernel_size=3, padding=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*self.num_classes, kernel_size=3, padding=1)]

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.shape[0], -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.shape[0], -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds

            


        


