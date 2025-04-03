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
import pandas as pd
from pathlib import Path

from config import TrainingConfig
from trainer import Trainer

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

def create_data_loaders(config: TrainingConfig):
    """Create training and validation data loaders"""
    # Load data
    train_df = pd.read_csv("../train.csv")
    val_df = pd.read_csv("../val.csv")
    print('Train:', train_df.shape)
    print('Validation:', val_df.shape)

    # Create datasets
    train_dataset = Image_Dataset(
        img_dir=config.image_dir,
        img_data=train_df,
        transform=config.transforms
    )
    
    val_dataset = Image_Dataset(
        img_dir=config.image_dir,
        img_data=val_df,
        transform=config.transforms
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    return train_loader, val_loader

def main():
    # Initialize configuration
    config = TrainingConfig()
    
    # Set random seed
    seed_everything(config.seed)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Initialize model
    model = Model()
    model.to(config.device)
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        config=config.__dict__
    )
    
    # Start training
    trainer.train(num_epochs=config.num_epochs)

if __name__ == "__main__":
    main()
