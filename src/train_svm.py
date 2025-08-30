# src/train_svm.py

import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Paths
TRAIN_EMB_PATH = "train_embeddings.pkl"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

# Ensure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

print("🔽 Loading training embeddings...")
with open(TRAIN_EMB_PATH, "rb") as f:
    train_data = pickle.load(f)

X_train = train_data["embeddings"]
y_train = train_data["labels"]

print(f"✅ Loaded {len(X_train)} training embeddings")

# Encode labels (names -> integers)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

print("🔄 Training SVM classifier...")
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train_enc)

# Save model + label encoder
joblib.dump({"model": clf, "label_encoder": le}, MODEL_PATH)

print(f"✅ SVM model saved at {MODEL_PATH}")
