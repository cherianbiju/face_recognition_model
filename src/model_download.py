import torch
from facenet_pytorch import InceptionResnetV1
import os

# Path to save model
save_path = "model/facenet_vggface2.pth"
os.makedirs("model", exist_ok=True)

# Step 1: Load pretrained model (with classifier)
print("🔽 Downloading FaceNet (VGGFace2) pretrained model...")
model = InceptionResnetV1(pretrained="vggface2", classify=True)

# Step 2: Remove classifier weights (logits.*) before saving
state_dict = model.state_dict()
clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("logits.")}

torch.save(clean_state_dict, save_path)
print(f"✅ Model saved at {save_path} (classifier removed)")

# Step 3: Reload model without classifier
print("🔄 Reloading model from saved file...")
model_local = InceptionResnetV1(pretrained=None, classify=False)
state_dict = torch.load(save_path, map_location="cpu")
model_local.load_state_dict(state_dict, strict=True)
model_local.eval()

print("✅ Model reloaded successfully and ready for embeddings!")
