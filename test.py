from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "./results/csebuetnlp/banglabert-genre-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example input
lyrics = "উজ্জ্বল করো হে আজি এ আনন্দরাতিবিকাশিয়া তোমার আনন্দমুখভাতি।সভা-মাঝে তুমি আজ বিরাজো হে রাজরাজ,আনন্দে রেখেছি তব সিংহাসন পাতি॥সুন্দর করো, হে প্রভু, জীবন যৌবনতোমারি মাধুরীসুধা করি বরিষন।লহো তুমি লহো তুলে তোমারি চরণমূলেনবীন মিলনমালা প্রেমসূত্রে গাঁথি॥মঙ্গল করো হে, আজি মঙ্গলবন্ধনতব শুভ আশীর্বাদ করি বিতরণ।বরিষ হে ধ্রুবতারা, কল্যাণকিরণধারা--দুর্দিনে সুদিনে তুমি থাকো চিরসাথী॥"  # Replace with Bangla text
inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, max_length=256)

# Prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs).item()

print(f"Predicted genre: {predicted_label}")
