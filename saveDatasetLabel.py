from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset(
    "csv",
    data_files={"train": "data/train.csv"},
    delimiter="$",
    column_names=["lyrics", "category"]
)

# Extract the unique labels from the training dataset
label_list = dataset["train"].unique("category")
label_list.sort()

# Generate the mappings
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Save the mappings to JSON files
with open("label_to_id.json", "w") as f:
    json.dump(label_to_id, f, ensure_ascii=False, indent=4)  # ensure_ascii=False for non-ASCII characters

with open("id_to_label.json", "w") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=4)

# Print confirmation
print("Mappings successfully regenerated and saved.")
