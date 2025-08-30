# test_lq_eval_full_integrated.py

import os
import torch
from PIL import Image
import joblib
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "models/svm_model_finetuned.pkl"
IMAGE_DIR = r"data/eval_lq"
FACENET_PATH = "model/facenet_vggface2.pth"
OUTPUT_CSV = "low_quality_predictions.csv"

# Confidence threshold
CONF_THRESHOLD = 0.7  # below this, prediction is flagged low confidence

# -------------------------------
# Load SVM
# -------------------------------
svm_data = joblib.load(MODEL_PATH)
clf = svm_data["model"]
le = svm_data["label_encoder"]

# -------------------------------
# Load FaceNet
# -------------------------------
model = InceptionResnetV1(pretrained=None, classify=False, num_classes=512)
state_dict = torch.load(FACENET_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# -------------------------------
# Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -------------------------------
# Collect predictions
# -------------------------------
results = []

folders = [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]

for person_name in folders:
    person_folder = os.path.join(IMAGE_DIR, person_name)
    print(f"\n📁 Processing folder: {person_name}")

    folder_results = []

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            
            if min(img.size) < 20:
                print(f"⚠️ Skipping tiny image: {img_file}")
                continue

            img_tensor = transform(img).unsqueeze(0)

            # Get embedding
            with torch.no_grad():
                embedding = model(img_tensor).squeeze(0).numpy()
            embedding = embedding.reshape(1, -1)

            # Predict
            pred_enc = clf.predict(embedding)
            pred_name = le.inverse_transform(pred_enc)[0]
            prob = clf.predict_proba(embedding)[0].max()
            low_conf_flag = prob < CONF_THRESHOLD

            # Print live
            print(f"Image: {img_file} | Predicted: {pred_name} | Probability: {prob:.4f}")

            record = {
                "image": img_file,
                "actual": person_name,
                "predicted": pred_name,
                "probability": prob,
                "low_confidence": low_conf_flag
            }

            results.append(record)
            folder_results.append(record)

        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
            record = {
                "image": img_file,
                "actual": person_name,
                "predicted": "ERROR",
                "probability": 0.0,
                "low_confidence": True,
                "error": str(e)
            }
            results.append(record)
            folder_results.append(record)

    # -------------------------------
    # Folder-level evaluation
    # -------------------------------
    y_true = [r["actual"] for r in folder_results if r["predicted"] != "ERROR"]
    y_pred = [r["predicted"] for r in folder_results if r["predicted"] != "ERROR"]

    if y_true:
        acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
        print(f"\n📊 Folder Accuracy: {acc*100:.2f}%")
        print("\n📋 Folder Classification Report:")
        print(classification_report(y_true, y_pred))
        print("\n🧮 Folder Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
        print(cm)

# -------------------------------
# Save CSV
# -------------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Predictions saved to {OUTPUT_CSV}")

# -------------------------------
# Overall evaluation
# -------------------------------
y_true_overall = [r["actual"] for r in results if r["predicted"] != "ERROR"]
y_pred_overall = [r["predicted"] for r in results if r["predicted"] != "ERROR"]

overall_acc = sum(yt == yp for yt, yp in zip(y_true_overall, y_pred_overall)) / len(y_true_overall)
print(f"\n📊 Overall Accuracy on low-quality eval set: {overall_acc*100:.2f}%")
print("\n📋 Overall Classification Report:")
print(classification_report(y_true_overall, y_pred_overall))
print("\n🧮 Overall Confusion Matrix:")
cm_overall = confusion_matrix(y_true_overall, y_pred_overall, labels=le.classes_)
print(cm_overall)
