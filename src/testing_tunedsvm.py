# evaluate_finetuned_svm.py

import pickle
import joblib
from sklearn.metrics import accuracy_score
import numpy as np
import os

# -------------------------------
# Paths
# -------------------------------
HIGH_TEST_PATH = "test_embeddings.pkl"       # high-quality test embeddings
LOW_TEST_PATH  = "test_embeddings_lq.pkl"    # low-quality test embeddings (optional)
MODEL_PATH     = "models/svm_model_finetuned.pkl"

# -------------------------------
# Load fine-tuned SVM
# -------------------------------
svm_data = joblib.load(MODEL_PATH)
clf = svm_data["model"]
le = svm_data["label_encoder"]

print("✅ Fine-tuned SVM loaded successfully\n")

# -------------------------------
# Helper function to evaluate
# -------------------------------
def evaluate(emb_path, name):
    print(f"🔽 Evaluating on {name} embeddings...")
    with open(emb_path, "rb") as f:
        data = pickle.load(f)
    
    X_test = np.array(data["embeddings"])
    y_test = data["labels"]
    
    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Accuracy on {name}: {accuracy*100:.2f}%\n")
    
    # Show first 5 predictions
    print("Some predictions:")
    for i in range(min(5, len(y_test))):
        print(f"Actual: {y_test[i]} | Predicted: {y_pred[i]}")
    print("\n"+"-"*40+"\n")

# -------------------------------
# Evaluate high-quality test set
# -------------------------------
if os.path.exists(HIGH_TEST_PATH):
    evaluate(HIGH_TEST_PATH, "High-Quality Test Set")

# -------------------------------
# Evaluate low-quality test set
# -------------------------------
if os.path.exists(LOW_TEST_PATH):
    evaluate(LOW_TEST_PATH, "Low-Quality Test Set")
