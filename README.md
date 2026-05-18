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

## Hyperparameter Experiments
11 hyperparameters were tested using controlled experiment methodology — only one parameter was changed at a time while all others remained fixed. After each experiment, the best-performing value was updated in the baseline configuration and used in all subsequent experiments:

- conv_kernel_size
- conv_dropout
- activation_fn
- batch_size
- optimizer_type
- num_conv_layers
- fc_hidden_units
- pool_kernel_size
- pool_stride
- fc_dropout
- num_epochs

For each parameter, at least 6 different values were tested and results were compared using validation accuracy and loss curves.

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

### Baseline vs Best Configuration

| Metric | Baseline | Best Config |
|---|---|---|
| Test Accuracy | 79.89% | 81.78% |
| Fake Precision | 0.88 | 0.74 |
| Fake Recall | 0.69 | 0.97 |
| Real Precision | 0.75 | 0.96 |
| Real Recall | 0.90 | 0.66 |
| F1 Macro | 0.80 | 0.81 |

### Key Findings
- Best config significantly improved Fake detection (recall: 0.69 → 0.97)
- Model is conservative — tends to predict Fake more often
- Vanilla CNN has inherent limitations for deepfake detection in the frequency domain

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
