import json
import re
import torch
import unicodedata
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, 
                          TrainingArguments, DataCollatorWithPadding)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Function to normalize text for consistent matching
def normalize_text(text):
    return unicodedata.normalize("NFKC", text).strip()

# Load the label_to_id mapping from JSON file
with open("label_to_id.json", "r", encoding="utf-8") as f:
    raw_label_to_id = json.load(f)

# Pre-clean label_to_id keys: remove parenthesized parts and normalize so that they match cleaned CSV labels
label_to_id = {}
for key, value in raw_label_to_id.items():
    # Remove parenthesized parts and normalize the key
    cleaned_key = normalize_text(re.sub(r"\s*\(.*?\)", "", key))
    label_to_id[cleaned_key] = int(value)   # convert value to integer

# Build reverse mapping (id_to_label) using the normalized label names
id_to_label = {str(v): k for k, v in label_to_id.items()}

number_of_genres = len(label_to_id)
model_name = "csebuetnlp/banglabert"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=number_of_genres).to(device)

# Load the dataset; CSV file columns: lyrics and category separated by $
dataset = load_dataset("csv", 
                       data_files={"train": "data/train.csv", "test": "data/test.csv"},
                       delimiter="$")

print("Train columns:", dataset["train"].column_names)  # Should show ['lyrics', 'category']

def preprocess_function(examples):
    # Tokenize the lyrics; output will include input_ids and attention_mask
    tokenized = tokenizer(
        examples["lyrics"],
        truncation=True,
        padding="max_length",   # Alternatively, padding=True can be used in collator
        max_length=256
    )
    
    # Clean the category labels from CSV; remove parenthesized parts and normalize for matching
    cleaned_labels = [normalize_text(re.sub(r"\s*\(.*?\)", "", label)) for label in examples["category"]]
    
    # Map cleaned labels to label IDs using the normalized mapping
    labels = []
    for lab in cleaned_labels:
        if lab in label_to_id:
            labels.append(label_to_id[lab])
        else:
            raise ValueError(f"Unknown label encountered: {lab}")
    
    tokenized["labels"] = labels
    return tokenized

# Remove original columns to ensure the dataset only contains tokenized fields
columns_to_remove = dataset["train"].column_names  # ['lyrics', 'category']

# Apply the mapping (batched) and remove unused columns
encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=columns_to_remove)

# Define training arguments with remove_unused_columns=False to preserve our processed fields
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=8,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    gradient_accumulation_steps=4,
    fp16=True,
    save_steps=500,
    remove_unused_columns=False
)

def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Data collator that pads the encoded inputs
data_collator = DataCollatorWithPadding(tokenizer)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model and tokenizer for later inference
model.save_pretrained("./results/csebuetnlp/banglabert-genre-classifier")
tokenizer.save_pretrained("./results/csebuetnlp/banglabert-genre-classifier")

# Optionally, save the reverse mapping
with open("id_to_label.json", "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Training complete. Model and tokenizer saved.")