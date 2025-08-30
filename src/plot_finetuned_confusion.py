# plot_finetuned_confusion.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import pickle

# -----------------------------
# Load fine-tuned SVM
# -----------------------------
saved = joblib.load("models/svm_model_finetuned.pkl")
svm_model = saved["model"]
label_encoder = saved["label_encoder"]

# -----------------------------
# Load eval_lq embeddings and true labels
# -----------------------------
with open("embeddings_eval_lq.pkl", "rb") as f:
    data = pickle.load(f)

y_true = data["labels"]       # True labels
embeddings = data["embeddings"]

# Encode true labels numerically
y_true_encoded = label_encoder.transform(y_true)

# Predict with fine-tuned SVM
y_pred = svm_model.predict(embeddings)

# -----------------------------
# Compute confusion matrix
# -----------------------------
cm = confusion_matrix(y_true_encoded, y_pred)
classes = label_encoder.classes_

# -----------------------------
# Plot heatmap
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Fine-Tuned SVM on eval_lq')
plt.tight_layout()

# Save figure for Overleaf
plt.savefig('images/finetuned_confusion_matrix.png', dpi=300)
plt.show()
