# extract_embeddings.py
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from facenet_pytorch import InceptionResnetV1

# Paths
MODEL_PATH = "model/facenet_vggface2.pth"
DATA_DIR = "data/train"
OUTPUT_PATH = "embeddings.pkl"

# Define transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load model (InceptionResnetV1)
model = InceptionResnetV1(pretrained=None, classify=False, num_classes=512)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("✅ FaceNet model loaded successfully!")

# Helper function: get embedding
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0)

# Walk through dataset
embeddings = []
labels = []

for person_name in os.listdir(DATA_DIR):
    person_folder = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        try:
            emb = get_embedding(img_path)
            embeddings.append(emb.numpy())
            labels.append(person_name)
            print(f"✅ Processed {img_file} -> {person_name}")
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

# Save embeddings + labels
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"embeddings": embeddings, "labels": labels}, f)

print(f"\n✅ Saved {len(embeddings)} embeddings to {OUTPUT_PATH}")
