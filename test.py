from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import json

# Load model and tokenizer
model_path = "./results/csebuetnlp/banglabert-genre-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Function to predict the genre of a given text
def predict_genre(lyrics):
    inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, max_length=256)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs).item()
    return predicted_label

# Option to choose between interactive input or testing with a CSV file
choice = input("Enter '1' for user input or '2' to test with a CSV file: ").strip()

if choice == '1':
    # Interactive mode
    lyrics = input("Enter lyrics: ").strip()
    predicted_genre = predict_genre(lyrics)
    print(f"Predicted genre ID: {predicted_genre}")

elif choice == '2':
    # CSV testing mode
    file_path = input("Enter the path to the CSV file: ").strip()
    label_to_id_path = "label_to_id.json"

    try:
        # Load label-to-ID mapping
        with open(label_to_id_path, "r", encoding="utf-8") as f:
            label_to_id = json.load(f)
        id_to_label = {v: k for k, v in label_to_id.items()}  # Reverse mapping for display

        # Read CSV file
        data = pd.read_csv(file_path, delimiter="$", header=None, names=["lyrics", "category"])
        correct = 0
        incorrect = 0

        # Loop through each row in the dataset
        count = 0
        for idx, row in data.iterrows():
            count = count + 1
            lyrics = row['lyrics']
            true_category_name = row['category']
            true_category_id = label_to_id.get(true_category_name, None)

            if true_category_id is None:
                print(f"Warning: Category '{true_category_name}' not found in label_to_id mapping.")
                continue

            predicted_category_id = predict_genre(lyrics)

            if predicted_category_id == true_category_id:
                correct += 1
            else:
                incorrect += 1
            if(count > 600) :
                break

        # Print the results
        total = correct + incorrect
        print(f"\nTotal samples: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Incorrect predictions: {incorrect}")
        print(f"Accuracy: {correct / total:.2%}")

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except json.JSONDecodeError:
        print("Error: The label_to_id.json file is not in valid JSON format.")
    except Exception as e:
        print(f"An error occurred: {e}")

else:
    print("Invalid choice! Please enter '1' or '2'.")
