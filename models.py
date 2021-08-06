import numpy as np
import pandas as pd
import os
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models

import torch
import torch.nn as nn
import efficientnet_pytorch
from transformers import AutoModel


class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth=7, num_classes=15, fine_tune=True):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        self.base = efficientnet_pytorch.EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_filter = self.base._fc.in_features
        self.set_fine_tune(fine_tune)

    def forward(self, x):
        x = self.base.extract_features(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        return x

    def set_fine_tune(self, fine_tune=True):
        for p in self.base.parameters():
            p.requires_grad = fine_tune


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.image_encoder = EfficientNetEncoderHead(depth=5)
        self.text_encoder =  AutoModel.from_pretrained('distilbert-base-multilingual-cased')
        self.classifier = nn.Linear(2048 + 768, 16)
    
    def forward(self, images, input_ids):
        image_vec = self.image_encoder(images)
        bert_output = self.text_encoder(input_ids, attention_mask=input_ids != 0)
        text_vec = bert_output[0][:, 0, :]
        feats = torch.cat([image_vec, text_vec], dim=1)
        logit = self.classifier(feats)
        return logit
    


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = F.softmax(self.fc(x.reshape(-1, 2048)), dim=1)
        return x


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobile = models.mobilenet_v2(pretrained=True)
        self.mobile.classifier[-1] = nn.Linear(self.mobile.classifier[-1].in_features, 2)
        
    def forward(self, x):
        x = F.softmax(self.mobile(x), 1)
        return x
