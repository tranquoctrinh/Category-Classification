import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import AutoTokenizer



class Image_Dataset(Dataset):
    def __init__(self, img_dir, img_data, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_data = img_data
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        self.maxlen = 350


    def convert_line_uncased(self, text):
        tokens_a = self.tokenizer.tokenize(text)[:self.maxlen]
        one_token = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokens_a + ["[SEP]"]
        )
        one_token += [0] * (self.maxlen - len(tokens_a))
        return one_token

        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        try:
            img_name = os.path.join(self.img_dir, str(self.img_data.loc[index, 'product_id']) + '.jpg')
            image = Image.open(img_name).convert('RGB')
            #image = Image.open(img_name).convert('L')
            label = self.img_data.loc[index, 'category_code']
            label = torch.tensor(label)
        except:
            img_name = os.path.join(self.img_dir, str(self.img_data.loc[0, 'product_id']) + '.jpg')
            #image = Image.open(img_name).convert('L')
            image = Image.open(img_name).convert('RGB')
            print("error")
            label = self.img_data.loc[0, 'category_code']
            label = torch.tensor(label)
        if self.transform is not None:
            image = self.transform(image)
        text = self.img_data['text'].values[index]
        tokens_text = self.convert_line_uncased(text)

        #image = image.repeat(3, 1, 1)
        return image, torch.LongTensor(tokens_text), label
