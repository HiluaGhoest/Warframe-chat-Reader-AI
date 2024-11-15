import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import time
import logging
import json
import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Device configuration
print(torch.cuda.is_available())  # Should print True if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model and tokenizer directory
model_dir = './results/final_model'

# Check if the model directory exists, and if so, load the model and tokenizer
if os.path.exists(model_dir):
    logging.info("Loading the fine-tuned model and tokenizer...")
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    logging.info("Model and tokenizer loaded successfully.")
else:
    logging.warning(f"Model directory {model_dir} not found. Training from scratch.")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5).to(device)

# Custom Dataset class for trade data
class TradeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move tensors to the same device as the model
        item = {
            'input_ids': encoding['input_ids'].flatten().to(device),
            'attention_mask': encoding['attention_mask'].flatten().to(device),
            'labels': torch.tensor(label, dtype=torch.long).to(device)
        }
        return item

# Function to load and preprocess your labeled dataset
def load_labeled_data(dataset_path):
    with open(dataset_path, 'r') as file:
        data = json.load(file)

    texts = []
    labels = []
    action_to_label = {'WTB': 0, 'WTS': 1, 'WTT': 2}  # Map your actions to labels

    for entry in data:
        username = entry.get('username', '')
        
        # Loop through each possible action (action1, action2, etc.)
        for action_key in ['action1', 'action2']:  # Extend this if you have more actions
            if action_key in entry:
                action = entry[action_key]
                
                # Get the items for this action
                items = entry.get(f'{action_key}items', [])
                
                # Process each item in the action's items
                for item_entry in items:
                    item_name = item_entry.get('item', '')
                    price = item_entry.get('price', '')
                    currency = item_entry.get('currency', '')
                    
                    # Construct text from available fields
                    text = f"{username} {action} {item_name} {price} {currency}"
                    texts.append(text)

                    # Assign label based on the action
                    label = action_to_label.get(action, 3)  # Default to 3 for unknown actions
                    labels.append(label)
    
    return texts, labels

# Fine-tuning function
def fine_tune_model(train_texts, train_labels, epochs=3, batch_size=8):
    # Create the dataset (no need to wrap it in DataLoader)
    train_dataset = TradeDataset(train_texts, train_labels, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',              # Output directory
        num_train_epochs=epochs,            # Number of training epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,                   # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                  # Strength of weight decay
        logging_dir='./logs',               # Directory for storing logs
        logging_steps=10,                   # Log every 10 steps
        evaluation_strategy="epoch",        # Evaluate every epoch
        save_steps=1000,                    # Save model every 1000 steps
        save_total_limit=2,                 # Keep only the last 2 saved models
        gradient_accumulation_steps=2,      # Accumulate gradients for 2 steps
        fp16=True if torch.cuda.is_available() else False,  # Enable FP16 only if CUDA is available
    )

    # Use the dataset directly, not the DataLoader
    trainer = Trainer(
        model=model,                        # The model to train
        args=training_args,                 # Training arguments
        train_dataset=train_dataset,        # Use the dataset directly
        eval_dataset=train_dataset          # Use a separate eval dataset if available
    )

    # Start training
    trainer.train()

    # Explicitly save the model and tokenizer
    logging.info("Saving the model...")
    model.save_pretrained('./results/final_model')
    tokenizer.save_pretrained('./results/final_model')

    logging.info("Model and tokenizer saved to './results/final_model'.")

# Load your labeled data from the messages_dataset.json
train_texts, train_labels = load_labeled_data('messages_dataset.json')

# Fine-tune the model with the labeled data (if not already fine-tuned)
if not os.path.exists(model_dir):
    fine_tune_model(train_texts, train_labels)

# Function to process the OCR result and extract trade data
def preprocess_image(image_path):
    logging.info("Preprocessing the image...")
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.autocontrast(image)  # Enhance contrast automatically
    image = ImageEnhance.Sharpness(image).enhance(2.0)  # Increase sharpness
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Apply binary thresholding
    logging.info("Image preprocessing complete.")
    return image

def extract_trade_data_with_ai(raw_text):
    results = []
    
    for line in raw_text.splitlines():
        clean_text = line.strip()
        structured_data = call_ai_model(clean_text)  # Call fine-tuned model here
        results.append(structured_data)
    
    return results

def call_ai_model(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():  # Turn off gradient calculation for inference
        outputs = model(**inputs)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()

    # Assuming you map the predicted label to a specific action
    action_map = {0: 'WTB', 1: 'WTS', 2: 'WTT', 3: 'Unknown'}
    action = action_map.get(prediction, 'Unknown')

    structured_data = {
        "username": "dummy_username",  # Extracted from text or external context
        "action": action,              # Action based on prediction index
        "item_name": "Prime Weapon",   # Extracted from text
        "price": "37",                 # Extracted from text or predicted
        "currency": "DUCAT"            # Extracted or predicted
    }
    
    return structured_data

def save_to_json(data, file_path='messages_data.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                logging.warning("JSON file is empty or malformed, starting fresh.")
                existing_data = []
    else:
        existing_data = []
    
    existing_data.extend(data)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

# Main flow
if __name__ == "__main__":
    start_time = time.time()

    logging.info("Script started.")
    
    try:
        # Once fine-tuned or loaded, preprocess the image and extract trade data
        image_path = 'screenshots/inner_area_screenshot.png'
        preprocessed_image = preprocess_image(image_path)
        
        logging.info("Performing OCR...")
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?\'\"-_: '
        raw_text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
        logging.info("OCR complete.")
        
        trade_data = extract_trade_data_with_ai(raw_text)
        
        logging.info(f"Extracted {len(trade_data)} trade data entries.")
        for entry in trade_data:
            print(f"Username: {entry['username']}, Action: {entry['action']}, Item: {entry['item_name']}, Price: {entry['price']} {entry['currency']}")
        
        save_to_json(trade_data)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    
    logging.info(f"Execution time: {time.time() - start_time:.2f} seconds.")
