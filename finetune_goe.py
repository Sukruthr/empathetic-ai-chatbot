import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import EarlyStoppingCallback
from sklearn.metrics import f1_score
import torch.nn.functional as F

import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    set_seed,
)
from sklearn.metrics import precision_recall_fscore_support, classification_report
from huggingface_hub import HfApi, HfFolder
from torch.utils.data import DataLoader, WeightedRandomSampler

from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Define emotion label mapping
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]
 
def read_metrics_file(filepath):
    """Read and parse metrics from a JSON file"""
    try:
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error reading metrics file {filepath}: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for emotion classification")
    parser.add_argument("--hub_model_id", type=str, required=True, help="Your Hugging Face model id")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base", help="Base model to use")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--confidence_threshold", type=float, default=0.3, 
                        help="Confidence threshold for displaying emotions")
    parser.add_argument("--force_train", action="store_true", help="Force training even if model exists in Hub")
    parser.add_argument("--view_results", type=str, help="Path to results directory to view metrics without training")
    parser.add_argument("--disable_tensorboard", action="store_true", help="Disable TensorBoard logging")
    return parser.parse_args()

def clean_text(text):
    # Remove leading/trailing whitespace (expand with regex if needed)
    return text.strip()

def preprocess_function(examples, tokenizer, max_length):
    # Clean the texts before tokenization
    cleaned_texts = [clean_text(txt) for txt in examples["text"]]
    return tokenizer(
        cleaned_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

def convert_labels(example, num_labels):
    multi_hot = [0.0] * num_labels  # Use float zeros instead of int
    # example["labels"] is assumed to be a list of label indices
    for label in example["labels"]:
        multi_hot[label] = 1.0  # Use float ones instead of int
    example["labels"] = multi_hot
    return example

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid and threshold at 0.5 for multi-label predictions
    sigmoid_logits = torch.sigmoid(torch.tensor(logits))
    preds = (sigmoid_logits > 0.5).float().cpu().numpy()  # Ensure we're on CPU
    
    # Convert labels to float numpy array (if not already)
    if isinstance(labels, torch.Tensor):
        labels = labels.float().cpu().numpy()
    else:
        labels = np.array(labels, dtype=np.float32)
    
    # Exact match accuracy: all labels must match
    exact_match = np.mean(np.all(np.abs(preds - labels) < 1e-6, axis=1))
    
    # Convert to binary for metrics calculation
    preds_binary = (preds > 0.5).astype(np.int32)
    labels_binary = (labels > 0.5).astype(np.int32)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_binary, preds_binary, average="micro", zero_division=0
    )
    
    # Compute macro-averaged metrics (across all emotion categories)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_binary, preds_binary, average="macro", zero_division=0
    )
    
    # Generate a detailed classification report for error analysis (per emotion category)
    report = classification_report(labels_binary, preds_binary, target_names=EMOTION_LABELS, 
                                  output_dict=True, zero_division=0)
    
    return {
        "exact_match_accuracy": exact_match,
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "classification_report": report,
    }

def save_metrics(metrics, filepath):
    """Save metrics to a JSON file with timestamp"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Add timestamp
    metrics["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")

def classify_text(text, classifier, threshold=0.3):
    try:
        results = classifier(text)
        print("\nInput Text:", text)
        print("Predicted Emotions:")
        
        # Sort emotions by score in descending order
        sorted_emotions = sorted(results[0], key=lambda x: x['score'], reverse=True)
        
        # Display only emotions above threshold
        found_emotions = False
        for emotion in sorted_emotions:
            # Map numerical label to emotion name
            try:
                label_id = int(emotion['label'].split('_')[-1]) if '_' in emotion['label'] else int(emotion['label'])
                if 0 <= label_id < len(EMOTION_LABELS):
                    emotion_name = EMOTION_LABELS[label_id]
                else:
                    emotion_name = emotion['label']
            except (ValueError, IndexError):
                # If there's an issue parsing the label, use the original label
                emotion_name = emotion['label']
                
            score = emotion['score']
            
            if score >= threshold:
                found_emotions = True
                print(f"  {emotion_name}: {score:.4f}")
        
        if not found_emotions:
            print(f"  No emotions detected above threshold ({threshold})")
    except Exception as e:
        print(f"Error during classification: {e}")
        print("Please try another sentence.")

def display_metrics(results, dataset_name="test"):
    """Display detailed metrics in a readable format"""
    print(f"\n=== {dataset_name.capitalize()} Results ===")
    
    # Display overall metrics
    print(f"Exact Match Accuracy: {results['eval_exact_match_accuracy']:.4f}")
    print(f"Micro Precision: {results['eval_precision_micro']:.4f}")
    print(f"Micro Recall: {results['eval_recall_micro']:.4f}")
    print(f"Micro F1: {results['eval_f1_micro']:.4f}")
    print(f"Macro F1: {results['eval_f1_macro']:.4f}")
    
    # Display per-emotion metrics
    print("\nPer-emotion performance:")
    report = results['eval_classification_report']
    
    # Create a sorted list of emotions by F1 score
    emotions_by_f1 = [(emotion, report[emotion]['f1-score']) 
                      for emotion in EMOTION_LABELS 
                      if emotion in report]
    emotions_by_f1.sort(key=lambda x: x[1], reverse=True)
    
    # Print top and bottom performing emotions
    print("\nTop 5 best performing emotions:")
    for emotion, f1 in emotions_by_f1[:5]:
        precision = report[emotion]['precision']
        recall = report[emotion]['recall']
        support = report[emotion]['support']
        print(f"  {emotion}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={support}")
    
    print("\nBottom 5 worst performing emotions:")
    for emotion, f1 in emotions_by_f1[-5:]:
        precision = report[emotion]['precision']
        recall = report[emotion]['recall']
        support = report[emotion]['support']
        print(f"  {emotion}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={support}")
    
    return

def view_results(results_dir):
    """View results from a previous training run"""
    print(f"Viewing results from {results_dir}")
    
    # Find the metrics directory
    metrics_dir = os.path.join(results_dir, "metrics")
    if not os.path.exists(metrics_dir):
        # Check if the path is already pointing to the metrics directory
        if os.path.exists(os.path.join(results_dir, "train_results.json")):
            metrics_dir = results_dir
        else:
            print(f"No metrics directory found at {metrics_dir}")
            return False
    
    # Find and read metrics files
    train_path = os.path.join(metrics_dir, "train_results.json")
    val_path = os.path.join(metrics_dir, "val_results.json")
    test_path = os.path.join(metrics_dir, "test_results.json")
    
    # Read all available metrics
    metrics_found = False
    
    if os.path.exists(train_path):
        metrics_found = True
        train_metrics = read_metrics_file(train_path)
        if train_metrics:
            display_metrics(train_metrics, "train")
    
    if os.path.exists(val_path):
        metrics_found = True
        val_metrics = read_metrics_file(val_path)
        if val_metrics:
            display_metrics(val_metrics, "validation")
    
    if os.path.exists(test_path):
        metrics_found = True
        test_metrics = read_metrics_file(test_path)
        if test_metrics:
            display_metrics(test_metrics, "test")
    
    # Check for summary file
    summary_path = os.path.join(metrics_dir, "summary.txt")
    if os.path.exists(summary_path):
        print("\nSummary:")
        with open(summary_path, 'r') as f:
            print(f.read())
    
    if not metrics_found:
        print("No metrics files found in the specified directory.")
        return False
    
    return True

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Check if we just want to view results without training
    if args.view_results:
        if view_results(args.view_results):
            return
        else:
            print("Continuing with normal operation...")
    
    # Define your Hugging Face Hub repository ID
    hub_model_id = args.hub_model_id
    # Check if the finetuned model exists on Hugging Face Hub
    api = HfApi()
    perform_inference = False
    
    try:
        if not args.force_train:
            # If the model exists, model_info will be retrieved successfully
            model_info = api.model_info(hub_model_id)
            print(f"Finetuned model found on Hugging Face Hub: {hub_model_id}")
            print("Loading model for inference. Use --force_train to retrain.")
            perform_inference = True
    except Exception as e:
        print(f"Finetuned model not found on Hugging Face Hub. Training will commence.")
    
    if perform_inference:
        try:
            # Load model and tokenizer from Hugging Face Hub for inference
            model = AutoModelForSequenceClassification.from_pretrained(hub_model_id)
            tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
        except Exception as e:
            print(f"Error loading model from Hub: {e}")
            print("Falling back to training a new model.")
            perform_inference = False
    
    if not perform_inference:
        print("Starting training process...")
        print(f"Using model: {args.model_name}")
        print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
        
        # Check if HF_TOKEN is set or if user is logged in
        if not HfFolder().get_token():
            print("WARNING: No Hugging Face token found. You might not be able to push to Hub.")
            print("Please login using `huggingface-cli login` or set the HF_TOKEN environment variable.")
        
        # -----------------------------
        # 1. Data Collection & Preprocessing
        # -----------------------------
        
        # Load the GoEmotions dataset
        print("Loading GoEmotions dataset...")
        dataset = load_dataset("go_emotions")
        
        # Print dataset statistics
        print(f"Dataset splits: {dataset.keys()}")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")
        print(f"Test examples: {len(dataset['test'])}")
        
        # Total number of labels (27 emotions + neutral)
        num_labels = 28
        print(f"Using {num_labels} emotion labels")
        
        # Sample a few examples to verify labels
        print("\nSample examples from dataset:")
        for i in range(min(3, len(dataset['train']))):
            print(f"Example {i}:")
            print(f"  Text: {dataset['train'][i]['text']}")
            print(f"  Labels: {[EMOTION_LABELS[idx] for idx in dataset['train'][i]['labels']]}")
        print()
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        print("Preprocessing dataset...")
        # Preprocessing function: clean and tokenize the "text" field
        def preprocess_batch(examples):
            return preprocess_function(examples, tokenizer, args.max_length)
        
        # Apply tokenization to the dataset (batched)
        encoded_dataset = dataset.map(preprocess_batch, batched=True, desc="Tokenizing")
        
        # Convert the original label lists to fixed-length multi-hot vectors
        def convert_batch(example):
            return convert_labels(example, num_labels)

        encoded_dataset = encoded_dataset.map(convert_batch, desc="Converting labels")

        # Create separate datasets for input features and labels
        # For input features (keep as integers)
        input_dataset = encoded_dataset.remove_columns(["labels"])
        input_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        # For labels (convert to float32)
        label_dataset = encoded_dataset.remove_columns(["input_ids", "attention_mask"])
        label_dataset.set_format(type="torch", columns=["labels"], dtype=torch.float32)
        
        # Create combined datasets for each split with correct typing
        class CombinedDataset(torch.utils.data.Dataset):
            def __init__(self, input_dataset, label_dataset):
                self.input_dataset = input_dataset
                self.label_dataset = label_dataset
            
            def __len__(self):
                return len(self.input_dataset)
            
            def __getitem__(self, idx):
                input_item = self.input_dataset[idx]
                label_item = self.label_dataset[idx]
                return {**input_item, **label_item}
        
        train_dataset = CombinedDataset(input_dataset["train"], label_dataset["train"])
        val_dataset = CombinedDataset(input_dataset["validation"], label_dataset["validation"])
        test_dataset = CombinedDataset(input_dataset["test"], label_dataset["test"])


        # Example: each training example gets a weight inversely proportional
        # to how many labels it has. You can adjust logic as you see fit.
        weights_for_samples = []
        for i in range(len(train_dataset)):
            labels_tensor = train_dataset[i]["labels"]
            # A simple approach: if the sample has any minority label, boost weight
            # Or just treat each sample equally, etc.
            # Here, for demonstration, we do 1 + sum_of_labels
            sample_weight = 1.0 + labels_tensor.sum().item()
            weights_for_samples.append(sample_weight)

        train_sampler = WeightedRandomSampler(
            weights_for_samples,
            num_samples=len(weights_for_samples),
            replacement=True
        )
        
        # -----------------------------
        # 2. Model Setup & Fine-Tuning
        # -----------------------------
        
        print("Initializing model...")
        # Load the model and set it up for multi-label classification
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )


        class WeightedFocalLoss(nn.Module):
            """
            Multi-label focal loss with optional alpha weighting.
            alpha: can be a list or tensor of length `num_labels`,
                or a single float (e.g., 1.0).
            gamma: focusing parameter, commonly 2.0 or 1.5, etc.
            """
            def __init__(self, alpha=None, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, logits, targets):
                """
                logits: shape (batch_size, num_labels)
                targets: shape (batch_size, num_labels)
                """
                # Binary Cross Entropy for multi-label
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                
                # Convert logits -> probabilities
                probas = torch.sigmoid(logits)
                
                # Apply focal loss formula: (1 - p)^gamma * BCE
                focal_term = (1.0 - probas) ** self.gamma
                loss = focal_term * bce
                
                # If alpha is set (per-class weighting), multiply it
                if self.alpha is not None:
                    # alpha can be shape (num_labels,) or just a float
                    # broadcasting will handle correct shapes:
                    loss = loss * self.alpha
                
                # Average over all labels & batch
                return loss.mean()

        
        # # Simple Weighted BCEWithLogitsLoss
        # class WeightedBCEWithLogitsLoss(nn.Module):
        #     def __init__(self, weights):
        #         super().__init__()
        #         self.weights = weights
        #         self.bce = nn.BCEWithLogitsLoss(reduction='none')
                
        #     def forward(self, logits, targets):
        #         # logits, targets => (batch, num_labels)
        #         loss_unreduced = self.bce(logits, targets)
        #         # Multiply each label by its class weight
        #         weighted_loss = loss_unreduced * self.weights
        #         return weighted_loss.mean()
        

        # Calculate class frequencies on the training set
        class_counts = [0]*num_labels
        for example in dataset['train']:
            for lbl_idx in example['labels']:
                class_counts[lbl_idx] += 1

        # E.g., weight = max_frequency / class_frequency
        max_count = max(class_counts) if len(class_counts) > 0 else 1
        class_weights = [max_count / (c if c>0 else 1) for c in class_counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        
        # Explicitly verify we're using the correct loss function (BCEWithLogitsLoss)
        if hasattr(model, 'config'):
            print(f"Model problem type: {model.config.problem_type}")
            model.config.problem_type = "multi_label_classification"
            print(f"Setting problem type to: {model.config.problem_type}")
        
        # Define training arguments
        output_dir = f"./results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            push_to_hub=True,
            hub_model_id=hub_model_id,
            hub_strategy="end",  # Push model at the end of training
            report_to="tensorboard" if not args.disable_tensorboard else None,
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
        )
        
        # Create a Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None, #  train_dataset SNIPPET
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        # trainer._compute_loss = lambda model, inputs: WeightedBCEWithLogitsLoss(class_weights.to(model.device))(
        # model(**inputs).logits, 
        # inputs["labels"]
        # )
        
        trainer._compute_loss = lambda model, inputs: WeightedFocalLoss(
            alpha=class_weights.to(model.device),  # pass your per-class weights
            gamma=2.0  # focusing parameter
        )(
            model(**inputs).logits,
            inputs["labels"]
        )


        train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler
        )
        # Let the trainer know to use our DataLoader
        trainer.get_train_dataloader = lambda: train_dataloader

        
        # Train the model
        print(f"Starting training for {args.epochs} epochs...")
        if not args.disable_tensorboard:
            print(f"TensorBoard logs will be saved to: {output_dir}")
            print(f"To view with TensorBoard run: tensorboard --logdir={output_dir}")
        else:
            print("TensorBoard logging is disabled")
        print(f"You can view results later with: python mode.py --hub_model_id {hub_model_id} --view_results {output_dir}")
        trainer.train()
        
        # -----------------------------
        # 3. Evaluation & Error Analysis
        # -----------------------------
        
        # Evaluate on each dataset split
        print("Evaluating model...")
        train_results = trainer.evaluate(train_dataset)
        val_results = trainer.evaluate(val_dataset)
        test_results = trainer.evaluate(test_dataset)

        # Save metrics to files
        metrics_dir = os.path.join(output_dir, "metrics")
        save_metrics(train_results, os.path.join(metrics_dir, "train_results.json"))
        save_metrics(val_results, os.path.join(metrics_dir, "val_results.json"))
        save_metrics(test_results, os.path.join(metrics_dir, "test_results.json"))
        
        # Display detailed metrics for each dataset split
        display_metrics(train_results, "train")
        display_metrics(val_results, "validation")
        display_metrics(test_results, "test")
        
        # Create a summary file with key metrics for easy reference
        summary_path = os.path.join(metrics_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Training completed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results directory: {output_dir}\n")
            if not args.disable_tensorboard:
                f.write(f"TensorBoard logs: tensorboard --logdir={output_dir}\n")
            else:
                f.write(f"TensorBoard logging was disabled\n")
            f.write(f"View results with: python mode.py --hub_model_id {hub_model_id} --view_results {output_dir}\n\n")
            
            f.write("=== Train Results ===\n")
            f.write(f"Micro F1: {train_results['eval_f1_micro']:.4f}\n")
            f.write(f"Macro F1: {train_results['eval_f1_macro']:.4f}\n\n")
            
            f.write("=== Validation Results ===\n")
            f.write(f"Micro F1: {val_results['eval_f1_micro']:.4f}\n")
            f.write(f"Macro F1: {val_results['eval_f1_macro']:.4f}\n\n")
            
            f.write("=== Test Results ===\n")
            f.write(f"Micro F1: {test_results['eval_f1_micro']:.4f}\n")
            f.write(f"Macro F1: {test_results['eval_f1_macro']:.4f}\n")
        
        print(f"\nSummary of results saved to {summary_path}")
        
        # -----------------------------
        # 4. Push the Model to Hugging Face Hub & Save Locally
        # -----------------------------
        
        print("Pushing model to Hugging Face Hub...")
        try:
            trainer.push_to_hub()
            print(f"Model successfully pushed to {hub_model_id}")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")
            print("Continuing with local save...")
        
        # Save the final model and tokenizer locally
        local_save_dir = "./deberta-goemotions"
        model.save_pretrained(local_save_dir)
        tokenizer.save_pretrained(local_save_dir)
        print(f"Model and tokenizer saved locally to {local_save_dir}")

    # -----------------------------
    # 5. Inference: Test the Model in the Terminal
    # -----------------------------

    print("\nInitializing sentiment classifier for inference...")
    # Create an inference pipeline using the loaded/trained model
    try:
        sentiment_classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None  # Returns scores for all labels
        )
        
        # Test if the model works with a simple example
        print("Testing model inference with a sample text...")
        test_result = sentiment_classifier("I am feeling happy today")
        # print(test_result)
        # print()
        print("Model is ready for inference!")
        
    except Exception as e:
        print(f"Error creating inference pipeline: {e}")
        print("Trying alternative approach...")
        
        # Alternative approach if the standard pipeline fails
        def manual_inference(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            with torch.no_grad():
                outputs = model(**inputs)
            scores = torch.sigmoid(outputs.logits)[0].cpu().numpy()
            return [{"label": str(i), "score": float(score)} for i, score in enumerate(scores)]
        
        # Replace the sentiment classifier with our manual function
        sentiment_classifier = lambda text: [manual_inference(text)]
        print("Using manual inference function instead of pipeline.")

    # Create a user-friendly interface for terminal-based inference
    print("\n" + "="*50)
    print("Emotion Classification Bot")
    print("="*50)
    print("Enter sentences to classify their emotions (type 'exit' to quit)")
    print(f"Only showing emotions with confidence >= {args.confidence_threshold}")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n>> ")
            if user_input.strip().lower() == "exit":
                break
            if not user_input.strip():
                continue
                
            classify_text(user_input, sentiment_classifier, args.confidence_threshold)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
