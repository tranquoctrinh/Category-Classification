from utils import *
from datasets import *
from models import *
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils import TensorboardAggregator
from rich.progress import track
from torch import optim
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split

SAMPLE_FRAC = 1
IMAGE_DIR = '/home/ubuntu/front/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, criterion, optimizer):
    model.train()
    logs_file = f"./tb_logs/final_raw/"
    writer = SummaryWriter(logs_file)
    agg = TensorboardAggregator(writer)
    model.train()
    running_loss = .0
    total = 0
    correct = 0
    bar = tqdm(train_loader, total=len(train_loader))
    for data_, input_ids, target_ in bar: 
        data_, input_ids, target_ = data_.to(device), input_ids.to(device), target_.to(device)# on GPU
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(data_, input_ids)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == target_).item()
        total += target_.size(0)

    accuracy = 100 * correct / total
    loss_score = running_loss/len(train_loader)
    return accuracy, loss_score


def eval(model, val_loader, criterion, optimizer):
    model.eval()
    batch_loss = 0
    total_t = 0
    correct_t = 0
    bar = tqdm(val_loader, total=len(val_loader))
    for data_, input_ids, target_ in bar: 
        data_, input_ids, target_ = data_.to(device), input_ids.to(device), target_.to(device)# on GPU
        outputs_t = model(data_, input_ids)
        loss_t = criterion(outputs_t, target_)
        batch_loss += loss_t.item()
        _,pred_t = torch.max(outputs_t, dim=1)
        correct_t += torch.sum(pred_t==target_).item()
        total_t += target_.size(0)
    accuracy = 100 * correct_t / total_t
    loss_score = batch_loss/len(val_loader)
    return accuracy, loss_score

def main():
    seed_everything(2323)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 8

    data = pd.read_csv('../samples_category_text.csv').sample(frac=SAMPLE_FRAC)
    #train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
    #train_df = train_df.reset_index(drop=True)
    #val_df = val_df.reset_index(drop=True)
    train_df = pd.read_csv("../train.csv")
    val_df = pd.read_csv("../val.csv")
    print('Train:', train_df.shape)
    print('Validation:', val_df.shape)

    train_dataset = Image_Dataset(img_dir=IMAGE_DIR, img_data=train_df, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = Image_Dataset(img_dir=IMAGE_DIR, img_data=val_df, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model = Model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()#.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    n_epochs = 5
    real_val_acc = -np.Inf

    for epoch in range(1, n_epochs+1):
        print('\nEpoch = [{}]/[{}]\n'.format(epoch, n_epochs))
        t_acc, t_loss = train(model, train_loader, criterion, optimizer)
        print(f'\ntrain loss: {t_loss:.4f}, train acc: {t_acc:.4f}')
        with torch.no_grad():
            v_acc, v_loss = eval(model, val_loader, criterion, optimizer)
            print(f'validation loss: {v_loss:.4f}, validation acc: {v_acc:.4f}\n')

            network_learned = v_acc > real_val_acc
            # Saving the best weight
            if network_learned:
                real_val_acc = v_acc
                torch.save(model.state_dict(), 'model_eff_category.pt')
                print('Detected network improvement, saving current model')

    print('Best accuracy is {}'.format(real_val_acc))

if __name__ == "__main__":
    main()
