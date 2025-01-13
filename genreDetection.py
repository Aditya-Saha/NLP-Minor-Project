from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Number of genres
numberOfGenres = 23
model_name = "csebuetnlp/banglabert"

# Check for CPU
device = torch.device("cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=numberOfGenres)

# Load dataset with $ delimiter
dataset = load_dataset(
    "csv",
    data_files={"train": "data/train.csv", "test": "data/test.csv"},
    delimiter="$",
    column_names=["lyrics", "category"]
)

# Map labels to integers
label_list = dataset["train"].unique("category")
label_list.sort()
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

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

# Define training arguments with reduced batch size and fewer epochs
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduce batch size
    per_device_eval_batch_size=2,
    num_train_epochs=2,  # Fewer epochs
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    gradient_accumulation_steps=4,  # Gradient accumulation
    fp16=True,  # Enable mixed precision training (CPU benefits)
    save_steps=500  # Save model every 500 steps
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