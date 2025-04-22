import json
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


file_path = r"C:\Users\sabya\NLP-Minor-Project\results\checkpoint-1880-20250128T180459Z-001\checkpoint-1880\trainer_state.json"
with open(file_path, 'r') as f:
    data = json.load(f)

log_history = data.get('log_history', [])

# Initialize lists for plotting
eval_epochs = []
eval_accuracy = []
eval_f1 = []
eval_loss = []
eval_precision = []
eval_recall = []
grad_epochs = []
grad_norm = []
lr_epochs = []
learning_rate = []
train_epochs = []
train_loss = []
all_true_labels = []
all_predicted_labels = []

# Populate lists with values
for log in log_history:
    if 'eval_accuracy' in log:
        eval_epochs.append(log['epoch'])
        eval_accuracy.append(log['eval_accuracy'])
    if 'eval_f1' in log:
        eval_f1.append(log['eval_f1'])
    if 'eval_loss' in log:
        eval_loss.append(log['eval_loss'])
    if 'eval_precision' in log:
        eval_precision.append(log['eval_precision'])
    if 'eval_recall' in log:
        eval_recall.append(log['eval_recall'])
    if 'grad_norm' in log:
        grad_epochs.append(log['epoch'])
        grad_norm.append(log['grad_norm'])
    if 'learning_rate' in log:
        lr_epochs.append(log['epoch'])
        learning_rate.append(log['learning_rate'])
    if 'loss' in log:
        train_epochs.append(log['epoch'])
        train_loss.append(log['loss'])
  
    if 'true_labels' in log and 'predicted_labels' in log:
        all_true_labels.extend(log['true_labels'])
        all_predicted_labels.extend(log['predicted_labels'])

# If true and predicted labels are missing, generate dummy data for testing
if not all_true_labels and not all_predicted_labels:
    print("True labels and predicted labels not found in JSON. Generating dummy data for testing...")
    all_true_labels = np.random.randint(0, 3, size=100) 
    all_predicted_labels = np.random.randint(0, 3, size=100)  

# Output directory for saving plots
output_dir = r"C:\Users\sabya\NLP-Minor-Project\results\plots"
os.makedirs(output_dir, exist_ok=True)

# Accuracy plot
if eval_accuracy:
    plt.figure()
    plt.plot(eval_epochs, eval_accuracy, label='Eval Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'eval_accuracy.png'))
    plt.close()

# F1 Score plot
if eval_f1:
    plt.figure()
    plt.plot(eval_epochs, eval_f1, label='Eval F1 Score', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Evaluation F1 Score vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'eval_f1.png'))
    plt.close()

# Loss plot
if eval_loss:
    plt.figure()
    plt.plot(eval_epochs, eval_loss, label='Eval Loss', marker='o', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'eval_loss.png'))
    plt.close()

# Precision and Recall plot
if eval_precision and eval_recall:
    plt.figure()
    plt.plot(eval_epochs, eval_precision, label='Eval Precision', marker='o', color='green')
    plt.plot(eval_epochs, eval_recall, label='Eval Recall', marker='o', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Precision and Recall vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall.png'))
    plt.close()

# Grad Norm plot
if grad_norm:
    plt.figure()
    plt.plot(grad_epochs, grad_norm, label='Grad Norm', marker='o', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'grad_norm.png'))
    plt.close()

# Learning Rate plot
if learning_rate:
    plt.figure()
    plt.plot(lr_epochs, learning_rate, label='Learning Rate', marker='o', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
    plt.close()

# Training Loss plot
if train_loss:
    plt.figure()
    plt.plot(train_epochs, train_loss, label='Training Loss', marker='o', color='pink')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

# Confusion Matrix
if len(all_true_labels) > 0 and len(all_predicted_labels) > 0:
    # Compute confusion matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 8))  
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')

    # Add title and labels
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)

    # Save the confusion matrix plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print("Confusion Matrix plot saved successfully!")
else:
    print("Confusion Matrix cannot be plotted: true labels or predicted labels missing.")

print(f"Plots saved in {output_dir}")
