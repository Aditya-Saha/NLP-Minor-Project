from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def main():
    # Number of genres
    numberOfGenres = 23
    model_name = "sagorsarker/bangla-bert-base"

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
            max_length=256  # Reduced max length for lower memory usage
        )
        tokenized_inputs["labels"] = [label_to_id[label] for label in examples["category"]]
        return tokenized_inputs

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Define training arguments with optimizations
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Ensure both eval and save strategies match
        save_strategy="epoch",  # Ensure both eval and save strategies match
        learning_rate=2e-5,  # Adjusted learning rate for better convergence
        per_device_train_batch_size=8,  # Reduced batch size for low RAM
        per_device_eval_batch_size=8,
        num_train_epochs=10,  # Increased number of epochs for better training
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,  # More frequent logging
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        gradient_accumulation_steps=1,  # Reduced gradient accumulation
        warmup_steps=500,  # Add warmup steps
        fp16=True,  # Use mixed precision for faster training
        dataloader_num_workers=2,  # Reduced workers for low RAM
        dataloader_pin_memory=True
    )

    # Define compute metrics function
    def compute_metrics(pred):
        predictions, labels = pred
        preds = np.argmax(predictions, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
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
    model.save_pretrained("./results/banglabert-genre-classifier")
    tokenizer.save_pretrained("./results/banglabert-genre-classifier")

if __name__ == "__main__":
    main()