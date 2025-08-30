# train_svm_finetune.py

import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# -------------------------------
# Paths
# -------------------------------
HIGH_EMB_PATH = "train_embeddings.pkl"       # high-quality embeddings
LOW_EMB_PATH  = "train_embeddings_lq.pkl"    # low-quality embeddings
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "svm_model_finetuned.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Load embeddings
# -------------------------------
print("🔽 Loading high-quality embeddings...")
with open(HIGH_EMB_PATH, "rb") as f:
    high_data = pickle.load(f)

print("🔽 Loading low-quality embeddings...")
with open(LOW_EMB_PATH, "rb") as f:
    low_data = pickle.load(f)

# Combine embeddings and labels
X_train = np.vstack([high_data["embeddings"], low_data["embeddings"]])
y_train = high_data["labels"] + low_data["labels"]

print(f"✅ Combined training data: {len(X_train)} embeddings")

# -------------------------------
# Encode labels
# -------------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# -------------------------------
# Train SVM
# -------------------------------
print("🔄 Training SVM classifier on combined embeddings...")
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train_enc)

# -------------------------------
# Save model + label encoder
# -------------------------------
checkpoint = {
    "model": clf,
    "label_encoder": le
}

joblib.dump(checkpoint, MODEL_PATH)

print(f"✅ Fine-tuned SVM model and label encoder saved at {MODEL_PATH}")
