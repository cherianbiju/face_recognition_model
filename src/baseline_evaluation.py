# step3_baseline_eval.py
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# -----------------------------
# Load embeddings
# -----------------------------
EMB_PATH = "embeddings_eval_lq.pkl"

with open(EMB_PATH, "rb") as f:
    data = pickle.load(f)

embeddings = np.array(data["embeddings"])
labels = np.array(data["labels"])

print(f"✅ Loaded {len(embeddings)} embeddings from {EMB_PATH}")

# -----------------------------
# Compute pairwise cosine similarity
# -----------------------------
similarity_matrix = cosine_similarity(embeddings)

y_true = []
y_score = []
num_samples = len(labels)

for i in range(num_samples):
    for j in range(i + 1, num_samples):
        y_true.append(1 if labels[i] == labels[j] else 0)
        y_score.append(similarity_matrix[i, j])

y_true = np.array(y_true)
y_score = np.array(y_score)

print(f"✅ Computed similarity for {len(y_score)} pairs")

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Baseline ROC - FaceNet embeddings')
plt.legend()
plt.show()

# -----------------------------
# Precision-Recall Curve
# -----------------------------
precision, recall, _ = precision_recall_curve(y_true, y_score)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, color='g')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Baseline Precision-Recall Curve')
plt.show()

# -----------------------------
# Similarity Distribution
# -----------------------------
same_scores = y_score[y_true == 1]
diff_scores = y_score[y_true == 0]

plt.figure(figsize=(6,5))
plt.hist(same_scores, bins=50, alpha=0.6, label='Same Identity')
plt.hist(diff_scores, bins=50, alpha=0.6, label='Different Identity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Similarity Distribution - Baseline')
plt.legend()
plt.show()

# -----------------------------
# Optional: summary statistics
# -----------------------------
print(f"Average similarity (same identity): {np.mean(same_scores):.4f}")
print(f"Average similarity (different identity): {np.mean(diff_scores):.4f}")
