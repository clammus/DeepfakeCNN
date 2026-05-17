# Deepfake Detection - CNN Based Classifier

A CNN-based deepfake image detection system built with PyTorch, developed as part of CMPE452 Advanced Deep Learning course.

## Overview
This project trains a deep CNN to classify face images as Real or Fake using the [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) dataset from Kaggle.

## Model Architecture
- 6 Convolutional layers with ReLU activation
- MaxPool2d for downsampling
- Fully connected layer with 1024 hidden units
- Dropout (0.5) for regularization
- Binary output: Real / Fake

## Best Configuration
| Hyperparameter | Value |
|---|---|
| conv_kernel_size | 3 |
| num_conv_layers | 6 |
| fc_hidden_units | 1024 |
| optimizer | RMSprop |
| batch_size | 128 |
| num_epochs | 40 |
| pool_kernel_size | 4 |

## Results
| Model | Test Accuracy |
|---|---|
| Baseline CNN | 80.6% |
| Best Config CNN | 81.8% |

## Interface
Built with Gradio. Run locally:
```bash
pip install -r requirements.txt
python app.py
```

## Dataset
- 30,000 images (15k Real, 15k Fake) subset from 190k original
- 256x256 RGB JPEG format
- Split: 70% train, 15% val, 15% test
