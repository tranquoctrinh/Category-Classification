from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import torchvision.transforms as transforms

@dataclass
class TrainingConfig:
    # Data parameters
    image_dir: str = '/home/ubuntu/front/'
    sample_fraction: float = 1.0
    batch_size: int = 8
    num_workers: int = 4
    
    # Model parameters
    model_name: str = 'efficientnet-b5'
    num_classes: int = 16
    learning_rate: float = 0.00001
    
    # Training parameters
    num_epochs: int = 5
    seed: int = 2323
    
    # Paths
    checkpoint_dir: Path = Path('checkpoints')
    log_dir: Path = Path('tb_logs/final_raw')
    
    # Device
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @property
    def transforms(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __post_init__(self):
        # Create necessary directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert string paths to Path objects if needed
        if isinstance(self.image_dir, str):
            self.image_dir = Path(self.image_dir) 