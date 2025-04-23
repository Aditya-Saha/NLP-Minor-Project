from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import json
import os
import sys

# -------------------- STEP 1: Load Label and Keyword Dictionaries --------------------

with open("label_to_id.json", "r", encoding="utf-8") as f:
    label_to_id = json.load(f)

with open("id_to_label.json", "r", encoding="utf-8") as f:
    id_to_label = json.load(f)

with open("genre_keywords.json", "r", encoding="utf-8") as f:
    genre_keywords = json.load(f)

numberOfGenres = len(label_to_id)
model_name = "csebuetnlp/banglabert"  # You can use 'sagorsarker/bangla-bert-base' as fallback

# -------------------- STEP 2: Load Tokenizer & Model --------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=numberOfGenres
    ).to(device)
except OSError as e:
    print(f"[ERROR] Model could not be loaded: {e}")
    print("Check if there's a local folder conflict or you're offline. You can try 'sagorsarker/bangla-bert-base'.")
    sys.exit(1)

# -------------------- STEP 3: Load Dataset --------------------

dataset = load_dataset(
    "csv",
    data_files={"train": "data/train.csv", "test": "data/test.csv"},
    delimiter="$",
    column_names=["lyrics", "category"]
)

# -------------------- STEP 4: Keyword-based Genre Detection --------------------

def detect_genre_by_keywords(lyrics):
    matched_genres = []
    for genre, keywords in genre_keywords.items():
        if any(word in lyrics for word in keywords):
            matched_genres.append(genre)
    return matched_genres[0] if matched_genres else None

def filter_and_annotate_dataset(dataset_split):
    lyrics_list = dataset_split["lyrics"]
    categories = dataset_split["category"]

    lyrics_clean = []
    categories_clean = []
    initial_labels = []

    for lyrics, category in zip(lyrics_list, categories):
        detected_genre = detect_genre_by_keywords(lyrics)
        final_genre = detected_genre if detected_genre else category
        label_id = label_to_id.get(final_genre)

        if label_id is not None:
            lyrics_clean.append(lyrics)
            categories_clean.append(category)
            initial_labels.append(label_id)

    return Dataset.from_dict({
        "lyrics": lyrics_clean,
        "category": categories_clean,
        "initial_label": initial_labels
    })

dataset["train"] = filter_and_annotate_dataset(dataset["train"])
dataset["test"] = filter_and_annotate_dataset(dataset["test"])

# -------------------- STEP 5: Tokenization --------------------

def preprocess_function(examples):
    inputs = tokenizer(
        examples["lyrics"],
        truncation=True,
        padding=True,
        max_length=256
    )
    # Flatten the label list
    inputs["labels"] = [int(label) for label in examples["initial_label"]]
    return inputs

encoded_dataset = dataset.map(preprocess_function, batched=True)

# -------------------- STEP 6: Training Arguments --------------------

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Fixing deprecated 'evaluation_strategy'
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
    fp16=torch.cuda.is_available(),
    save_steps=500
)

# -------------------- STEP 7: Evaluation Metrics --------------------

def compute_metrics(pred):
    predictions, labels = pred
    preds = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# -------------------- STEP 8: Trainer & Training --------------------

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# -------------------- STEP 9: Save Model & Tokenizer --------------------

output_dir = "./results/csebuetnlp/banglabert-genre-classifier"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save label mapping again
with open(os.path.join(output_dir, "id_to_label.json"), "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=4)

print("Training complete and model saved!")
