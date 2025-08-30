# src/split_embeddings.py
import pickle
from sklearn.model_selection import train_test_split

# Load full embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
labels = data["labels"]

# Split into train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

# Save train set
with open("train_embeddings.pkl", "wb") as f:
    pickle.dump({"embeddings": X_train, "labels": y_train}, f)

# Save test set
with open("test_embeddings.pkl", "wb") as f:
    pickle.dump({"embeddings": X_test, "labels": y_test}, f)

print("✅ Train/Test embeddings saved successfully!")
print(f"🔹 Train size: {len(X_train)}")
print(f"🔹 Test size: {len(X_test)}")
