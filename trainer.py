import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Initialize logging
        self.log_dir = Path(config.get('log_dir', './tb_logs/final_raw'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.best_val_acc = -np.inf
        self.current_epoch = 0
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        
        bar = tqdm(self.train_loader, total=len(self.train_loader))
        for data_, input_ids, target_ in bar:
            data_ = data_.to(self.device)
            input_ids = input_ids.to(self.device)
            target_ = target_.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data_, input_ids)
            loss = self.criterion(outputs, target_)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            
        accuracy = 100 * correct / total
        loss_score = running_loss / len(self.train_loader)
        return accuracy, loss_score
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        batch_loss = 0
        total = 0
        correct = 0
        
        with torch.no_grad():
            bar = tqdm(self.val_loader, total=len(self.val_loader))
            for data_, input_ids, target_ in bar:
                data_ = data_.to(self.device)
                input_ids = input_ids.to(self.device)
                target_ = target_.to(self.device)
                
                outputs = self.model(data_, input_ids)
                loss = self.criterion(outputs, target_)
                batch_loss += loss.item()
                
                _, pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred == target_).item()
                total += target_.size(0)
                
        accuracy = 100 * correct / total
        loss_score = batch_loss / len(self.val_loader)
        return accuracy, loss_score
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.config['checkpoint_dir'] / 'latest.pt')
        
        # Save best model if needed
        if is_best:
            torch.save(checkpoint, self.config['checkpoint_dir'] / 'best.pt')
    
    def train(self, num_epochs: int):
        """Main training loop"""
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            print(f'\nEpoch [{epoch}/{num_epochs}]')
            
            # Training phase
            train_acc, train_loss = self.train_epoch()
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            
            # Validation phase
            val_acc, val_loss = self.validate()
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Save checkpoint if validation accuracy improves
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print('New best model detected!')
            
            self.save_checkpoint(epoch, val_acc, is_best)
        
        print(f'Training completed. Best validation accuracy: {self.best_val_acc:.4f}')
        self.writer.close() 