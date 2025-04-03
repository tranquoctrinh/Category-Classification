from utils import *
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset
import io
from transformers import AutoTokenizer
from s3 import get_s3
from database_connections import qry_tpl_pglq as qrt


def get_result_json(product_id, namshub):
    return qrt(f"""SELECT tr.result_json
            FROM ai.task_results tr
            WHERE tr.namshub = '{namshub}'
                and tr.product_id = {product_id}
                and tr.is_deleted=0::BIT
            LIMIT 1
            """)


class Image_Dataset_Infer(Dataset):
    def __init__(self, img_data, transform=None):
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
        img_path_s3 = f"labelquestai/images1000px/{product_id}_front"
        image_bytes = get_s3(img_path_s3)
        image = Image.frombytes('RGBA', (128,128), image_bytes, 'raw')

        image = Image.open(img_name).convert('RGB')
        #image = Image.open(img_name).convert('L')
        label = self.img_data.loc[index, 'category_code']
        label = torch.tensor(label)

        if self.transform is not None:
            image = self.transform(image)
        # text = self.img_data['text'].values[index]
        tokens_text = self.convert_line_uncased(text)

        #image = image.repeat(3, 1, 1)
        return image, torch.LongTensor(tokens_text), label





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(model, product_id, transform):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    image = image.unsqueeze(0).to(device)

    pred =  model(image)#.cpu().detach()
    pred = pred.cpu().detach()
    y_pred = torch.argmax(pred).item()
    return y_pred


def main():
    product_id = ['2125570', '2135570', '2132170']

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 8

    # model = Model()
    # model.to(device)
    # model.load_state_dict(torch.load('model_eff_category.pt'))
    # model.eval()

    img_path_s3 = f"labelquestai/images1000px/{product_id[0]}_front"
    image_byte = get_s3(img_path_s3)
    image = Image.open(io.BytesIO(image_byte))
    image.save('tmp.jpg')

    ocr = get_result_json(product_id, 'google_ocr')[0][0][image_type]["ocr"]
    print(ocr)
    import ipdb; ipdb.set_trace()
main()
