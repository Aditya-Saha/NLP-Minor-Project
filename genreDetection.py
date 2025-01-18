from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import json

# Number of genres (ensure this matches the saved label-to-ID mappings)

# Load label mappings from JSON file
with open("label_to_id.json", "r") as f:
    label_to_id = json.load(f)

numberOfGenres = len(label_to_id)
model_name = "csebuetnlp/banglabert"

# Check for GPU or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=numberOfGenres).to(device)

# Load dataset with $ delimiter
dataset = load_dataset(
    "csv",
    data_files={"train": "data/train.csv", "test": "data/test.csv"},
    delimiter="$",
    column_names=["lyrics", "category"]
)


# Generate id_to_label mapping
id_to_label = {int(v): k for k, v in label_to_id.items()}

# Preprocess the dataset
def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["lyrics"],
        truncation=True,
        padding=True,
        max_length=256  # Reduce max_length
    )
    tokenized_inputs["labels"] = [label_to_id[label] for label in examples["category"]]
    return tokenized_inputs

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduce batch size
    per_device_eval_batch_size=2,
    num_train_epochs=8,  # Fewer epochs
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    gradient_accumulation_steps=4,
    fp16=True,
    save_steps=500
)

# Define compute metrics function
def compute_metrics(pred):
    predictions, labels = pred
    preds = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Data collator to handle padding and batching
data_collator = DataCollatorWithPadding(tokenizer)

# Trainer
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

# Save model and tokenizer
model.save_pretrained("./results/csebuetnlp/banglabert-genre-classifier")
tokenizer.save_pretrained("./results/csebuetnlp/banglabert-genre-classifier")

# Save the mappings again (optional, for consistency)
with open("id_to_label.json", "w") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=4)
