# Category Classification

A deep learning model that classifies products into categories using a combination of EfficientNet for image processing and BERT for text understanding. This multimodal approach leverages both visual and textual information to achieve accurate product categorization.

## Features

- Multimodal architecture combining EfficientNet and BERT
- Support for 16 product categories
- EfficientNet-B5 as the backbone for image feature extraction
- DistilBERT for text processing
- Configurable training parameters
- TensorBoard logging support
- Checkpoint saving and loading

## Project Structure

```
.
├── config.py          # Configuration settings and parameters
├── datasets.py        # Dataset loading and preprocessing
├── infer.py          # Inference script
├── models.py         # Model architectures
├── train.py          # Training script
├── trainer.py        # Training logic and utilities
└── utils.py          # Helper functions
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- efficientnet-pytorch
- transformers
- Pillow
- pandas
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/category-classification.git
cd category-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py
```

The training configuration can be modified in `config.py`. Key parameters include:
- `image_dir`: Directory containing product images
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `model_name`: Base model architecture

### Inference

To run inference on new products:

```bash
python infer.py --image_path /path/to/image --text "product description"
```

## Model Architecture

The model combines two main components:
1. **Image Encoder**: EfficientNet-B5 for extracting visual features
2. **Text Encoder**: DistilBERT for processing product descriptions

The features from both encoders are concatenated and passed through a classifier to predict the product category.

## Configuration

Key configuration parameters in `config.py`:
- Image preprocessing: Resize to 256x256, center crop to 224x224
- Normalization: ImageNet mean and standard deviation
- Device: Automatically uses CUDA if available
- Checkpoint and log directories are automatically created

## License

[Your License]

## Contributing

[Your Contributing Guidelines]
