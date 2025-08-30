# hq_svm_eval_fixed2.py
import pickle
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# -----------------------------
# Paths
# -----------------------------
EMB_PATH = "embeddings_eval_lq.pkl"  # low-quality eval embeddings
SVM_PATH = "models/svm_model.pkl"    # HQ-trained SVM saved as dict

# -----------------------------
# Load embeddings
# -----------------------------
with open(EMB_PATH, "rb") as f:
    data = pickle.load(f)

embeddings_eval = np.array(data["embeddings"])
labels_eval = np.array(data["labels"])

print(f"✅ Loaded {len(embeddings_eval)} embeddings from {EMB_PATH}")

# -----------------------------
# Load trained SVM dict
# -----------------------------
saved = joblib.load(SVM_PATH)
svm_model = saved["model"]
le = saved["label_encoder"]
print("✅ HQ-trained SVM and LabelEncoder loaded")

# -----------------------------
# Make predictions
# -----------------------------
pred_encoded = svm_model.predict(embeddings_eval)
pred_labels = le.inverse_transform(pred_encoded)

# -----------------------------
# Compute metrics
# -----------------------------
accuracy = accuracy_score(labels_eval, pred_labels)
f1 = f1_score(labels_eval, pred_labels, average='weighted')
cm = confusion_matrix(labels_eval, pred_labels)

print("\n📊 SVM Evaluation Metrics (HQ-trained SVM on eval_lq):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print("\nConfusion Matrix:\n", cm)

# Optional: detailed per-class metrics
report = classification_report(labels_eval, pred_labels)
print("\nClassification Report:\n", report)
