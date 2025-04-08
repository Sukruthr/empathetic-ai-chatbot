# Emotion Classification with DeBERTa

This repository contains code for fine-tuning a DeBERTa model for emotion classification using the GoEmotions dataset. The model can classify text into multiple emotions and provides detailed metrics for evaluation.

## Table of Contents
- [Setup](#setup)
- [Requirements](#requirements)
- [Usage](#usage)
- [Features](#features)
- [Viewing Results](#viewing-results)
- [TensorBoard Logging](#tensorboard-logging)

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/sukru/emotion-classification.git
   cd emotion-classification
   ```

## Requirements
You can install the required packages using pip:
```bash
conda create -n emotion_classification python=3.9
conda activate emotion_classification
pip install -r requirements.txt
```


2. Ensure you have a Hugging Face account and set up your authentication token:
   ```bash
   huggingface-cli login
   ```

## Usage
To train the model, run the following command:
```bash
python finetune_goe.py --hub_model_id "your-username/deberta-goemotions"
```
For inference run:
```bash
python finetune_goe.py --hub_model_id "suku9/emotion_classifier"
```

### Command Line Arguments
- `--hub_model_id`: Your Hugging Face model ID (required).
- `--model_name`: Base model to use (default: `microsoft/deberta-base`).
- `--learning_rate`: Learning rate for training (default: `2e-5`).
- `--batch_size`: Batch size for training and evaluation (default: `256`).
- `--epochs`: Number of training epochs (default: `2`).
- `--seed`: Random seed for reproducibility (default: `42`).
- `--max_length`: Maximum sequence length for tokenization (default: `128`).
- `--confidence_threshold`: Confidence threshold for displaying emotions (default: `0.3`).
- `--force_train`: Force training even if the model exists in the Hub.
- `--view_results`: Path to results directory to view metrics without training.
- `--disable_tensorboard`: Disable TensorBoard logging.

## Features
- **Multi-label Emotion Classification**: Classifies text into multiple emotions using a multi-label classification approach.
- **Detailed Metrics**: Computes and displays precision, recall, F1 scores, and a classification report for each emotion.
- **TensorBoard Logging**: Logs training metrics for visualization with TensorBoard.
- **Results Viewing**: Allows viewing of saved metrics without retraining the model.

## Viewing Results
To view results from a previous training run without retraining, use:
```bash
python finetune_goe.py --hub_model_id "your-username/deberta-goemotions" --view_results "./results/YYYYMMDD_HHMMSS"
```

## TensorBoard Logging
If you want to enable TensorBoard logging, ensure you do not use the `--disable_tensorboard` flag. After training, you can view the logs with:
```bash
tensorboard --logdir=./results/YYYYMMDD_HHMMSS
tensorboard --logdir=results
```