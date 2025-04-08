import gradio as gr
import torch
import os
import json
import re
import random
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)
import datetime
import sys

# Define emotion label mapping
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Map similar emotions to our response categories
EMOTION_MAPPING = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "joy",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "neutral",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "neutral",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral"
}

class ChatbotContext:
    """Class to maintain conversation context and history"""
    def __init__(self):
        self.conversation_history = []
        self.detected_emotions = []
        self.user_feedback = []
        self.current_session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Track emotional progression for therapeutic conversation flow
        self.conversation_stage = "initial"  # initial, middle, advanced
        self.emotion_trajectory = []  # track emotion changes over time
        self.consecutive_positive_count = 0
        self.consecutive_negative_count = 0
        # Add user name tracking
        self.user_name = None
        self.bot_name = "Mira"  # Friendly, easy to remember name
        self.introduced = False
        self.waiting_for_name = False
    
    def add_message(self, role, text, emotions=None):
        """Add a message to the conversation history"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = {
            "role": role,
            "text": text,
            "timestamp": timestamp
        }
        if emotions and role == "user":
            message["emotions"] = emotions
            self.detected_emotions.append(emotions)
            self._update_emotional_trajectory(emotions)
        
        self.conversation_history.append(message)
        return message
    
    def _update_emotional_trajectory(self, emotions):
        """Update the emotional trajectory based on newly detected emotions"""
        # Get the primary emotion
        primary_emotion = emotions[0]["emotion"] if emotions else "neutral"
        
        # Add to trajectory
        self.emotion_trajectory.append(primary_emotion)
        
        # Classify as positive, negative, or neutral
        positive_emotions = ["joy", "admiration", "amusement", "excitement", 
                             "optimism", "gratitude", "pride", "love", "relief"]
        negative_emotions = ["sadness", "anger", "fear", "disgust", "disappointment", 
                             "annoyance", "disapproval", "embarrassment", "grief", 
                             "remorse", "nervousness"]
        
        if primary_emotion in positive_emotions:
            self.consecutive_positive_count += 1
            self.consecutive_negative_count = 0
        elif primary_emotion in negative_emotions:
            self.consecutive_negative_count += 1
            self.consecutive_positive_count = 0
        else:  # neutral or other
            # Don't reset counters for neutral emotions to maintain progress
            pass
        
        # Update conversation stage based on trajectory and message count
        msg_count = len(self.conversation_history) // 2  # Count actual exchanges (user/bot pairs)
        if msg_count <= 1:  # First real exchange
            self.conversation_stage = "initial"
        elif msg_count <= 3:  # First few exchanges
            self.conversation_stage = "middle"
        else:  # More established conversation
            self.conversation_stage = "advanced"
    
    def get_emotional_state(self):
        """Get the current emotional state of the conversation"""
        if len(self.emotion_trajectory) < 2:
            return "unknown"
        
        # Get the last few emotions (with 'neutral' having less weight)
        recent_emotions = self.emotion_trajectory[-3:]
        positive_emotions = ["joy", "admiration", "amusement", "excitement", 
                             "optimism", "gratitude", "pride", "love", "relief"]
        negative_emotions = ["sadness", "anger", "fear", "disgust", "disappointment", 
                             "annoyance", "disapproval", "embarrassment", "grief", 
                             "remorse", "nervousness"]
        
        # Count positive and negative emotions
        pos_count = sum(1 for e in recent_emotions if e in positive_emotions)
        neg_count = sum(1 for e in recent_emotions if e in negative_emotions)
        
        if self.consecutive_positive_count >= 2:
            return "positive"
        elif self.consecutive_negative_count >= 2:
            return "negative"
        elif pos_count > neg_count:
            return "improving"
        elif neg_count > pos_count:
            return "declining"
        else:
            return "neutral"
    
    def add_feedback(self, rating, comments=None):
        """Add user feedback about the chatbot's response"""
        feedback = {
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.user_feedback.append(feedback)
        return feedback
    
    def get_recent_messages(self, count=5):
        """Get the most recent messages from the conversation history"""
        return self.conversation_history[-count:] if len(self.conversation_history) >= count else self.conversation_history
    
    def save_conversation(self, filepath=None):
        """Save the conversation history to a JSON file"""
        if not filepath:
            os.makedirs("./conversations", exist_ok=True)
            filepath = f"./conversations/conversation_{self.current_session_id}.json"
        
        data = {
            "conversation_history": self.conversation_history,
            "user_feedback": self.user_feedback,
            "emotion_trajectory": self.emotion_trajectory,
            "session_id": self.current_session_id,
            "start_time": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Conversation saved to {filepath}")
        return filepath

def clean_response_text(response, user_name):
    """Clean up the response text to make it more natural"""
    # Remove repeated name mentions
    if user_name:
        # Replace patterns like "Hey user_name," or "Hi user_name,"
        response = re.sub(r'^(Hey|Hi|Hello)\s+' + re.escape(user_name) + r',?\s+', '', response, flags=re.IGNORECASE)
        
        # Replace duplicate name mentions
        pattern = re.escape(user_name) + r',?\s+.*' + re.escape(user_name)
        if re.search(pattern, response, re.IGNORECASE):
            response = re.sub(r',?\s+' + re.escape(user_name) + r'([,.!?])', r'\1', response, flags=re.IGNORECASE)
        
        # Remove name at the end of sentences if it appears earlier
        if response.count(user_name) > 1:
            response = re.sub(r',\s+' + re.escape(user_name) + r'([.!?])(\s|$)', r'\1\2', response, flags=re.IGNORECASE)
    
    # Remove phrases that feel repetitive or formulaic
    phrases_to_remove = [
        r"let me know what you'd prefer,?\s+",
        r"i'm here to listen,?\s+",
        r"let me know if there's anything else,?\s+",
        r"i'm all ears,?\s+",
        r"i'm here for you,?\s+"
    ]
    
    for phrase in phrases_to_remove:
        response = re.sub(phrase, "", response, flags=re.IGNORECASE)
    
    # Fix multiple punctuation
    response = re.sub(r'([.!?])\s+\1', r'\1', response)
    
    # Fix missing space after punctuation
    response = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', response)
    
    # Make sure first letter is capitalized
    if response and len(response) > 0:
        response = response[0].upper() + response[1:]
    
    return response.strip()

class GradioEmotionChatbot:
    def __init__(self, emotion_model_id, response_model_id=None, confidence_threshold=0.3):
        self.emotion_model_id = emotion_model_id
        self.response_model_id = response_model_id or "mistralai/Mistral-7B-Instruct-v0.2"
        self.confidence_threshold = confidence_threshold
        self.context = ChatbotContext()
        self.initialize_models()
        
    def initialize_models(self):
        # Initialize emotion classification model
        print(f"Loading emotion classification model: {self.emotion_model_id}")
        try:
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(self.emotion_model_id)
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.emotion_model_id)
            
            self.emotion_classifier = pipeline(
                "text-classification",
                model=self.emotion_model,
                tokenizer=self.emotion_tokenizer,
                top_k=None  # Returns scores for all labels
            )
            print("Emotion classification model loaded successfully!")
        except Exception as e:
            print(f"Error loading emotion classification model: {e}")
            # Fallback to a dummy classifier for demo purposes
            self.emotion_classifier = lambda text: [[{"label": "neutral", "score": 1.0}]]
        
        # Initialize response generation model (or use fallback)
        print(f"Loading response generation model: {self.response_model_id}")
        try:
            self.response_model = AutoModelForCausalLM.from_pretrained(
                self.response_model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.response_tokenizer = AutoTokenizer.from_pretrained(self.response_model_id)
            
            self.response_generator = pipeline(
                "text-generation",
                model=self.response_model,
                tokenizer=self.response_tokenizer,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                temperature=0.7,
                max_new_tokens=100
            )
            print("Response generation model loaded successfully!")
        except Exception as e:
            print(f"Using fallback response generation. Reason: {e}")
            self.response_generator = self.fallback_response_generator
    
    def fallback_response_generator(self, prompt, **kwargs):
        """Fallback response generator using templates"""
        # Try to extract emotion from the prompt
        emotion_match = re.search(r"emotion: (\w+)", prompt.lower())
        if emotion_match:
            emotion = emotion_match.group(1)
        else:
            emotion = "neutral"
            
        # Default user name
        user_name = "friend"
        name_match = re.search(r"Your friend \((.*?)\)", prompt.lower())
        if name_match:
            user_name = name_match.group(1)
        
        # Extract user message
        message_match = re.search(r"message: \"(.*?)\"", prompt)
        user_message = message_match.group(1) if message_match else ""
        
        # Generate response using fallback method
        response = self.natural_fallback_response(user_message, emotion, user_name)
        
        # Format as if coming from the pipeline
        return [{"generated_text": response}]
    
    def natural_fallback_response(self, user_message, primary_emotion, user_name):
        """Conversational fallback responses that sound like a supportive friend"""
        # Define emotion categories
        sad_emotions = ["sadness", "disappointment", "grief", "remorse"]
        fear_emotions = ["fear", "nervousness", "anxiety"]
        anger_emotions = ["anger", "annoyance", "disapproval", "disgust"] 
        joy_emotions = ["joy", "admiration", "amusement", "excitement", "optimism", 
                       "gratitude", "pride", "love", "relief"]
        
        # Multi-stage response templates - more natural and varied
        if primary_emotion in joy_emotions:
            responses = [
                f"That's awesome, {user_name}! What made you feel that way?",
                f"I'm so glad to hear that! Tell me more about it?",
                f"That's great news! What else is going on with you lately?"
            ]
        elif primary_emotion in sad_emotions:
            responses = [
                f"I'm sorry to hear that, {user_name}. Want to talk about what happened?",
                f"That sounds rough. What's been going on?",
                f"Ugh, that's tough. How are you handling it?"
            ]
        elif primary_emotion in anger_emotions:
            responses = [
                f"That sounds really frustrating. What happened?",
                f"Oh no, that would upset me too. Want to vent about it?",
                f"I can see why you'd be upset about that. What are you thinking of doing?"
            ]
        elif primary_emotion in fear_emotions:
            responses = [
                f"That sounds scary, {user_name}. What's got you worried?",
                f"I can imagine that would be stressful. What's on your mind about it?",
                f"I get feeling anxious about that. What's the biggest concern for you?"
            ]
        else:  # neutral emotions
            responses = [
                f"What's been on your mind lately, {user_name}?",
                f"How's everything else going with you?",
                f"Tell me more about what's going on in your life these days."
            ]
        
        return random.choice(responses)
    
    def classify_text(self, text):
        """Classify text and return emotion data"""
        try:
            results = self.emotion_classifier(text)
            
            # Sort emotions by score in descending order
            sorted_emotions = sorted(results[0], key=lambda x: x['score'], reverse=True)
            
            # Process emotions above threshold
            detected_emotions = []
            for emotion in sorted_emotions:
                # Map numerical label to emotion name
                try:
                    label_id = int(emotion['label'].split('_')[-1]) if '_' in emotion['label'] else int(emotion['label'])
                    if 0 <= label_id < len(EMOTION_LABELS):
                        emotion_name = EMOTION_LABELS[label_id]
                    else:
                        emotion_name = emotion['label']
                except (ValueError, IndexError):
                    emotion_name = emotion['label']
                    
                score = emotion['score']
                
                if score >= self.confidence_threshold:
                    detected_emotions.append({"emotion": emotion_name, "score": score})
            
            # If no emotions detected above threshold, add neutral
            if not detected_emotions:
                detected_emotions.append({"emotion": "neutral", "score": 1.0})
                
            return detected_emotions
        except Exception as e:
            print(f"Error during classification: {e}")
            # Return neutral as fallback
            return [{"emotion": "neutral", "score": 1.0}]
    
    def format_emotion_text(self, emotion_data):
        """Create a simple emotion text display"""
        if not emotion_data:
            return ""
            
        # Define emotion emojis
        emotion_emojis = {
            "joy": "😊", "admiration": "🤩", "amusement": "😄", "approval": "👍",
            "excitement": "🎉", "gratitude": "🙏", "love": "❤️", "optimism": "🌟",
            "pride": "🦚", "relief": "😌", "sadness": "😢", "disappointment": "😞",
            "grief": "💔", "remorse": "😔", "embarrassment": "😳", "anger": "😠",
            "annoyance": "😤", "disapproval": "👎", "disgust": "🤢", "fear": "😨",
            "nervousness": "😰", "surprise": "😲", "confusion": "😕", "curiosity": "🤔",
            "neutral": "😐", "realization": "💡", "desire": "✨"
        }
        
        # Format the primary emotion
        primary = emotion_data[0]["emotion"]
        emoji = emotion_emojis.get(primary, "😐")
        score = emotion_data[0]["score"]
        
        return f"Detected: {emoji} {primary.capitalize()} ({score:.2f})"
        
    def generate_response(self, user_message, emotion_data):
        """Generate a response based on the user's message and detected emotions"""
        # Get the primary emotion with context awareness
        primary_emotion = emotion_data[0]["emotion"] if emotion_data else "neutral"
        
        # Get recent conversation history for context
        recent_exchanges = self.context.get_recent_messages(6)
        conversation_history = ""
        for msg in recent_exchanges:
            role = "Friend" if msg["role"] == "user" else self.context.bot_name
            conversation_history += f"{role}: {msg['text']}\n"
        
        # Check if this is a greeting
        is_greeting = any(greeting in user_message.lower() for greeting in ["hi", "hello", "hey", "greetings"])
        is_question_about_bot = "how are you" in user_message.lower() or any(q in user_message.lower() for q in ["what can you do", "who are you", "what are you", "your purpose"])
        
        # Handle special cases
        if is_greeting:
            if len(self.context.conversation_history) <= 4:  # First greeting exchange
                return f"Hi! I'm {self.context.bot_name}. It's nice to meet you. How are you feeling today?"
            else:
                return f"Hey! Good to chat with you again. What's been going on with you?"
        
        elif is_question_about_bot:
            return f"I'm doing well, thanks for asking! I'm {self.context.bot_name}, here as a friend to chat whenever you need someone to talk to. What's on your mind today?"
        
        # Create a more conversational prompt based on emotion
        system_instruction = f"""You are {self.context.bot_name}, having a natural conversation with your friend. You should respond in a casual, warm way like a supportive friend would - not like a therapist or clinical chatbot.

Your friend seems to be feeling {primary_emotion}. In your response:
1. Be genuinely empathetic but natural - like how a real friend would respond
2. Keep your response short (1-3 sentences) and conversational
3. Don't use phrases like "I understand" or "I'm here for you" too much - vary your language
4. Use casual language, contractions (don't instead of do not), and occasional sentence fragments
5. Don't sound formulaic or overly positive - be authentic
6. Keep the same emotional tone throughout your response
7. Don't explain what you're doing or add meta-commentary
8. DON'T address them by name multiple times or at the end of sentences - it sounds unnatural
9. Don't end with "Let me know what you'd prefer" or similar phrases

Recent conversation:
{conversation_history}

Your friend's message: "{user_message}"
Current emotion: {primary_emotion}

Respond naturally as a supportive friend (without using their name more than once if at all):"""
        
        try:
            # Generate the response
            generated = self.response_generator(
                system_instruction,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
            )
            
            # Extract the generated text
            if isinstance(generated, list):
                response_text = generated[0].get('generated_text', '')
            else:
                response_text = generated.get('generated_text', '')
            
            # Clean up the response - extract only the actual response without system prompt
            if "[/INST]" in response_text:
                parts = response_text.split("[/INST]")
                if len(parts) > 1:
                    response_text = parts[1].strip()
            
            # If we're still getting the system instruction, try an alternative approach
            if "Your friend seems to be feeling" in response_text:
                # Try to extract just the bot's response using pattern matching
                match = re.search(r'Respond naturally as a supportive friend.*?:\s*(.*?)$', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
                else:
                    # If that fails, try another approach - take text after the last numbered instruction
                    match = re.search(r'9\.\s+[^\n]+\s*(.*?)$', response_text, re.DOTALL)
                    if match:
                        response_text = match.group(1).strip()
                    else:
                        # Last resort: pick a fallback response based on emotion
                        response_text = self.natural_fallback_response(user_message, primary_emotion, self.context.user_name or "friend")
            
            # Remove any model-specific markers
            response_text = response_text.replace("<s>", "").replace("</s>", "")
            
            # Remove any internal notes or debugging info that might appear
            if "Note:" in response_text:
                response_text = response_text.split("Note:")[0].strip()
            
            # Remove any metadata or system-like text
            response_text = response_text.replace("Assistant:", "").replace(f"{self.context.bot_name}:", "").strip()
            
            # Remove any quotation marks surrounding the response
            response_text = response_text.strip('"').strip()
            
            # Handle potential model halt mid-sentence
            if response_text.endswith((".", "!", "?")):
                pass  # Response ends with proper punctuation
            else:
                # Try to find the last complete sentence
                last_period = max(response_text.rfind("."), response_text.rfind("!"), response_text.rfind("?"))
                if last_period > len(response_text) * 0.5:  # If we've got at least half the response
                    response_text = response_text[:last_period+1]
            
            # FINAL CHECK: If we still have parts of the system prompt, use fallback response
            if any(phrase in response_text for phrase in ["Your friend seems to be feeling", "Keep your response short", "Be genuinely empathetic"]):
                response_text = self.natural_fallback_response(user_message, primary_emotion, self.context.user_name or "friend")
            
            return clean_response_text(response_text.strip(), self.context.user_name)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self.natural_fallback_response(user_message, primary_emotion, self.context.user_name or "friend")
    
    def process_message(self, user_message, chatbot_history):
        """Process a user message and return the chatbot response"""
        # Initialize context if first message
        if not self.context.conversation_history:
            initial_greeting = f"Hi! I'm {self.context.bot_name}, your friendly emotional support chatbot. Who am I talking to today?"
            self.context.add_message("bot", initial_greeting)
            self.context.waiting_for_name = True
            return [[None, initial_greeting]]
        
        # Handle name collection if this is the first user message
        if self.context.waiting_for_name and not self.context.introduced:
            common_greetings = ["hi", "hey", "hello", "greetings", "howdy", "hiya"]
            words = user_message.strip().split()
            potential_name = None
            
            if "i'm" in user_message.lower() or "im" in user_message.lower():
                parts = user_message.lower().replace("i'm", "im").split("im")
                if len(parts) > 1 and parts[1].strip():
                    potential_name = parts[1].strip().split()[0].capitalize()
            
            elif "my name is" in user_message.lower():
                parts = user_message.lower().split("my name is")
                if len(parts) > 1 and parts[1].strip():
                    potential_name = parts[1].strip().split()[0].capitalize()
            
            elif len(words) <= 3 and words[0].lower() not in common_greetings:
                potential_name = words[0].capitalize()
            
            if potential_name:
                potential_name = ''.join(c for c in potential_name if c.isalnum())
            
            if potential_name and len(potential_name) >= 2 and potential_name.lower() not in common_greetings:
                self.context.user_name = potential_name
                greeting_response = f"Nice to meet you, {self.context.user_name}! How are you feeling today?"
            else:
                self.context.user_name = "friend"
                greeting_response = "Nice to meet you! How are you feeling today?"
            
            self.context.introduced = True
            self.context.waiting_for_name = False
            self.context.add_message("user", user_message)
            self.context.add_message("bot", greeting_response)
            
            return chatbot_history + [[user_message, greeting_response]]
        
        # Regular message processing
        emotion_data = self.classify_text(user_message)
        self.context.add_message("user", user_message, emotion_data)
        
        # Generate the response
        bot_response = self.generate_response(user_message, emotion_data)
        self.context.add_message("bot", bot_response)
        
        # Create a simple emotion display text
        emotion_text = self.format_emotion_text(emotion_data)
        
        # Combine emotion text with bot response
        full_response = f"{emotion_text}\n\n{bot_response}" if emotion_text else bot_response
        
        # Return updated chat history in the expected tuple format
        return chatbot_history + [[user_message, full_response]]
    
    def reset_conversation(self):
        """Reset the conversation context"""
        self.context = ChatbotContext()
        return []

# Create the Gradio interface
def create_gradio_interface():
    # Initialize the chatbot with default models
    emotion_model_id = os.environ.get("EMOTION_MODEL_ID", "suku9/emotion-classifier")
    response_model_id = os.environ.get("RESPONSE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    
    chatbot = GradioEmotionChatbot(emotion_model_id, response_model_id)
    
    # Create the Gradio interface with dark mode styling
    custom_css = """
    /* Dark mode styling */
    body {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    
    .gradio-container {
        max-width: 750px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        border-radius: 12px !important;
        background: #2d2d2d !important;
        padding: 15px !important;
    }

    /* Chatbot header styling */
    .gradio-container h1, #header {
        color: #a29bfe !important;
        text-align: center !important;
        font-size: 1.8rem !important;
        margin-bottom: 5px !important;
        font-weight: 600 !important;
    }

    .gradio-container p, #subheader {
        text-align: center !important;
        color: #b0b0b0 !important;
        margin-bottom: 15px !important;
        font-size: 0.9rem !important;
    }

    /* Chatbot window styling */
    #chatbot {
        height: 380px !important;
        overflow: auto !important;
        border-radius: 10px !important;
        background-color: #1a1a1a !important;
        border: 1px solid #3d3d3d !important;
        padding: 10px !important;
        margin-bottom: 15px !important;
    }
    
    /* Force horizontal text orientation for ALL elements */
    * {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        direction: ltr !important;
    }
    
    /* Message styling */
    .message {
        border-radius: 12px !important;
        padding: 8px 12px !important;
        margin: 5px 0 !important;
        max-width: 85% !important;
        width: 250px !important;
        word-break: break-word !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        direction: ltr !important;
    }
    
    .user-message {
        background-color: #4a5568 !important;
        color: #e2e8f0 !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
    }
    
    .bot-message {
        background-color: #553c9a !important;
        color: #f8f9fa !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
    }

    /* User input styling - FIX FOR VERTICAL TEXT */
    #user-input, .gradio-container textarea, .gradio-container input[type="text"] {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border-radius: 20px !important;
        padding: 10px 15px !important;
        border: 1px solid #3d3d3d !important;
        margin-bottom: 10px !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        direction: ltr !important;
        width: 100% !important;
        min-height: 45px !important;
        height: auto !important;
        resize: none !important;
    }
    
    /* Force text orientation for any text inputs */
    .cm-editor, .cm-scroller, .cm-content, .cm-line {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
    }
    
    /* Ensure row is horizontal */
    .gradio-row {
        flex-direction: row !important;
    }

    /* Fix for chat bubbles */
    .chat, .chat > div, .chat > div > div, .chat-msg, .chat-msg > div, .chat-msg-content {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
    }

    /* Apply horizontal text to all text elements in chatbot */
    .prose, .prose p, .prose span, .text-input-with-enter {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        direction: ltr !important;
    }
    
    /* Target the specific user bubble on the right side */
    .gradio-chatbot > div > div {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        direction: ltr !important;
    }

    /* Target any text inside chatbot bubbles */
    .gradio-chatbot * {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        direction: ltr !important;
    }

    /* AVATAR AND USERNAME FIXES */
    .avatar, .avatar-container, .avatar-image, .user-avatar, .bot-avatar {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        direction: ltr !important;
    }

    /* Fix for specific containers that might be causing issues */
    [class*="message"], [class*="bubble"], [class*="avatar"], [class*="chat"] {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        direction: ltr !important;
    }

    /* Button styling */
    .send-btn, .clear-btn {
        background-color: #6c5ce7 !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    .send-btn:hover, .clear-btn:hover {
        background-color: #5649c1 !important;
        transform: translateY(-1px) !important;
    }

    .clear-btn {
        background-color: #e74c3c !important;
    }

    .clear-btn:hover {
        background-color: #c0392b !important;
    }

    /* Hide footer */
    footer {
        display: none !important;
    }
    
    /* Fix scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        background-color: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #4a4a4a;
        border-radius: 3px;
    }
    """
    
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# EmotionChat", elem_id="header")
        gr.Markdown("A supportive chatbot that understands how you feel", elem_id="subheader")
        
        # Chat interface with improved styling
        chatbot_interface = gr.Chatbot(
            elem_id="chatbot",
            show_label=False,
            height=380,
            avatar_images=["https://em-content.zobj.net/source/microsoft-teams/363/bust-in-silhouette_1f464.png", 
                           "https://em-content.zobj.net/source/microsoft-teams/363/robot_1f916.png"],
        )
        
        # Input and button row with better styling
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                container=False,
                scale=9,
                elem_id="user-input",
                lines=1,
                max_lines=1,
                rtl=False
            )
            submit_btn = gr.Button("Send", scale=1, elem_classes="send-btn")
        
        # New conversation button
        clear_btn = gr.Button("New Conversation", elem_classes="clear-btn")
        
        # Set up the event handlers
        submit_btn.click(
            chatbot.process_message,
            inputs=[user_input, chatbot_interface],
            outputs=[chatbot_interface],
        ).then(
            lambda: "",  # Clear the input box after sending
            None,
            [user_input],
        )
        
        user_input.submit(
            chatbot.process_message,
            inputs=[user_input, chatbot_interface],
            outputs=[chatbot_interface],
        ).then(
            lambda: "",  # Clear the input box after sending
            None,
            [user_input],
        )
        
        clear_btn.click(
            chatbot.reset_conversation,
            inputs=None,
            outputs=[chatbot_interface],
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(debug=True, share=True)