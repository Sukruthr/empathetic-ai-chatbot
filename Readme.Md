# EmpatheticAI: A Complete Guide to Building and Deploying an Emotional Support Chatbot

This comprehensive guide will walk you through the process of building and deploying an emotion-aware chatbot from scratch, including fine-tuning an emotion classification model and creating a responsive chat interface. By the end, you'll have a fully functional emotional support chatbot that can detect user emotions and respond appropriately.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Fine-tuning the Emotion Classification Model](#fine-tuning-the-emotion-classification-model)
4. [Building the Chatbot Application](#building-the-chatbot-application)
5. [Deploying the Chatbot](#deploying-the-chatbot)


## Project Overview

EmpatheticAI is a chatbot that can:
- Detect emotions in user messages using a fine-tuned DeBERTa model
- Respond with empathetic and contextually appropriate messages
- Track emotional changes throughout a conversation
- Provide a user-friendly web interface

The project consists of two main components:
1. An emotion classification model fine-tuned on the GoEmotions dataset
2. A Gradio-based chat application that uses the model to detect emotions and generate responses

## Environment Setup

### Prerequisites

- Python 3.9+ 
- CUDA-compatible GPU (recommended for faster training and inference)
- Hugging Face account (for model hosting)

### Installation

1. Clone the repository (or create a new directory for your project):

```bash
git git@github.com:Sukruthr/empathetic-ai-chatbot.git
cd empathetic-ai-chatbot
```

2. Create a virtual environment:

```bash
conda create -n empathetic_ai python=3.9
conda activate empathetic_ai
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

Here's what should be in your `requirements.txt` file:

```
transformers>=4.36.0
datasets 
torch>=2.0.0 
numpy>=1.20.0
pandas 
scikit-learn 
tensorboard 
transformers[torch]
matplotlib   
huggingface_hub
gradio>=4.0.0
accelerate>=0.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```

4. Set up Hugging Face authentication for model uploading:

```bash
huggingface-cli login
```

Enter your authentication token when prompted. You can find this token in your Hugging Face account settings.

## Fine-tuning the Emotion Classification Model

The emotion classification model is the core of our chatbot's ability to understand user emotions. We'll fine-tune a DeBERTa model on the GoEmotions dataset, which contains 28 emotion categories.

### Understanding the GoEmotions Dataset

The GoEmotions dataset includes text samples labeled with emotions like joy, sadness, anger, fear, etc. Each sample can have multiple emotion labels, making this a multi-label classification problem.

### Running the Fine-tuning Script

1. Check the `finetune_goe.py` file to understand the fine-tuning process.

2. Run the fine-tuning script with your Hugging Face model ID:

```bash
python finetune_goe.py --hub_model_id "yourusername/emotion-classifier" --epochs 8 --batch_size 16
```

You can adjust the hyperparameters:
- `--model_name`: Base model to use (default: `microsoft/deberta-base`)
- `--learning_rate`: Learning rate for training (default: `2e-5`)
- `--batch_size`: Batch size for training and evaluation (default: `16`)
- `--epochs`: Number of training epochs (default: `5`)
- `--max_length`: Maximum sequence length (default: `128`)

### Monitoring Training Progress

The script will:
1. Download and preprocess the GoEmotions dataset
2. Fine-tune the model
3. Evaluate performance on test and validation sets
4. Push the model to Hugging Face Hub
5. Save metrics to a results directory

You can monitor training progress by viewing TensorBoard logs:

```bash
tensorboard --logdir=./results/YYYYMMDD_HHMMSS
```

### Understanding Training Results

After training, check the summary stats (from the actual summary.txt in the training results):

```
Model: microsoft/deberta-base
Training completed on: 2025-04-06 11:02:41
Results directory: ./results/20250406_103803

=== Train Results ===
Micro F1: 0.7785
Macro F1: 0.6870

=== Validation Results ===
Micro F1: 0.5892
Macro F1: 0.4924

=== Test Results ===
Micro F1: 0.5851
Macro F1: 0.4783
```

The F1 score is a balance of precision and recall. Micro F1 considers all predictions equally, while Macro F1 gives equal weight to each emotion class.

## Building the Chatbot Application

Now that we have our emotion classification model, let's build the chatbot application.

### Understanding the Chatbot Components

1. **ChatbotContext Class**: Manages conversation history and emotional context
2. **GradioEmotionChatbot Class**: Handles emotion detection and response generation
3. **UI Interface**: Uses Gradio for the web interface

### Implementing the Chatbot

Create a file named `app.py` with the provided code. Here's an overview of the key components:

#### Emotion Classification

The chatbot uses our fine-tuned model to classify emotions in user messages:

```python
def classify_text(self, text):
    """Classify text and return emotion data"""
    results = self.emotion_classifier(text)
    sorted_emotions = sorted(results[0], key=lambda x: x['score'], reverse=True)
    detected_emotions = []
    
    for emotion in sorted_emotions:
        # Extract emotion name and score
        emotion_name = EMOTION_LABELS[int(emotion['label'])]
        score = emotion['score']
        
        if score >= self.confidence_threshold:
            detected_emotions.append({"emotion": emotion_name, "score": score})
    
    return detected_emotions
```

#### Response Generation

Based on detected emotions, the chatbot generates contextually appropriate responses:

```python
def generate_response(self, user_message, emotion_data):
    """Generate a response based on the user's message and detected emotions"""
    # Get the primary emotion with context awareness
    primary_emotion = emotion_data[0]["emotion"] if emotion_data else "neutral"
    
    # Create a conversational prompt based on emotion
    system_instruction = f"""You are {self.context.bot_name}, having a natural conversation with your friend. 
    Your friend seems to be feeling {primary_emotion}...
    """
    
    # Generate the response using a language model
    response_text = self.response_generator(system_instruction)
    
    return clean_response_text(response_text, self.context.user_name)
```

### Running the Chatbot Application

To run the chatbot application:

```bash
python app.py
```

This will start a local web server with the Gradio interface, typically on http://127.0.0.1:7860/.

## Deploying the Chatbot

There are several ways to deploy your chatbot for broader use.

### Option 1: Gradio Sharing

The simplest option is to use Gradio's sharing feature, which creates a temporary public URL:

```python
demo.launch(share=True)
```

Add `share=True` to the `demo.launch()` call in your `app.py` file. This will generate a public URL that will be valid for 72 hours.

### Option 2: Hugging Face Spaces

For a more permanent solution, you can deploy your application on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Upload your code to the Space repository
3. Add a requirements.txt file with all dependencies
4. Add an app.py file as the entry point

Happy building! 🤖❤️