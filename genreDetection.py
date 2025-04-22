from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import json

# -------------------- STEP 1: Load Label and Keyword Dictionary --------------------

with open("label_to_id.json", "r", encoding="utf-8") as f:
    label_to_id = json.load(f)

with open("id_to_label.json", "r", encoding="utf-8") as f:
    id_to_label = json.load(f)

with open("genre_keywords.json", "r", encoding="utf-8") as f:
    genre_keywords = json.load(f)

numberOfGenres = len(label_to_id)
model_name = "csebuetnlp/banglabert"

# -------------------- STEP 2: Load Tokenizer & Model --------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the model is loaded correctly
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=numberOfGenres).to(device)
except OSError as e:
    print(f"Error loading model: {e}")
    # Handle model loading error here

# -------------------- STEP 3: Load Dataset --------------------

dataset = load_dataset(
    "csv",
    data_files={"train": "data/train.csv", "test": "data/test.csv"},
    delimiter="$",
    column_names=["lyrics", "category"]
)

# Check column names to ensure 'category' exists
print(dataset["train"].column_names)
print(dataset["test"].column_names)

# -------------------- STEP 4: Keyword-based Classification --------------------

def detect_genre_by_keywords(lyrics):
    matched_genres = []
    for genre, keywords in genre_keywords.items():
        if any(word in lyrics for word in keywords):
            matched_genres.append(genre)
    return matched_genres[0] if matched_genres else None

def filter_and_annotate_dataset(dataset_split):
    lyrics_list = dataset_split["lyrics"]
    labels = []

    for lyrics in lyrics_list:
        genre = detect_genre_by_keywords(lyrics)
        if genre and genre in label_to_id:
            labels.append(label_to_id[genre])
        else:
            labels.append(None)  # Mark for model prediction

    dataset_split = dataset_split.add_column("initial_label", labels)
    return dataset_split

dataset["test"] = filter_and_annotate_dataset(dataset["test"])
dataset["train"] = filter_and_annotate_dataset(dataset["train"])

# -------------------- STEP 5: Tokenization & Label Filling --------------------

def preprocess_function(examples):
    inputs = tokenizer(
        examples["lyrics"],
        truncation=True,
        padding=True,
        max_length=256
    )
    
    # Ensure the labels are integers (not strings)
    inputs["labels"] = [
        int(label) if label is not None else int(label_to_id.get(examples["category"][i], -1))
        for i, label in enumerate(examples["initial_label"])
    ]
    
    return inputs


encoded_dataset = dataset.map(preprocess_function, batched=True)

# -------------------- STEP 6: Training Arguments --------------------

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
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
    save_steps=500
)

# -------------------- STEP 7: Metrics --------------------

def compute_metrics(pred):
    predictions, labels = pred
    preds = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# -------------------- STEP 8: Training --------------------

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# -------------------- STEP 9: Save Everything --------------------

model.save_pretrained("./results/csebuetnlp/banglabert-genre-classifier")
tokenizer.save_pretrained("./results/csebuetnlp/banglabert-genre-classifier")

with open("id_to_label.json", "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=4)
