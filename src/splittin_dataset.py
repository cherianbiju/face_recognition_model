import os
import shutil
import random

low_quality_root = "data/eval"
train_lq_root = "data/train_lq"
eval_lq_root = "data/eval_lq"

os.makedirs(train_lq_root, exist_ok=True)
os.makedirs(eval_lq_root, exist_ok=True)

split_ratio = 0.5

for person in os.listdir(low_quality_root):
    person_folder = os.path.join(low_quality_root, person)
    low_quality_folder = os.path.join(person_folder, "low_quality")

    if not os.path.exists(low_quality_folder):
        print(f"Skipping {person}: no low_quality folder found")
        continue

    images = [f for f in os.listdir(low_quality_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not images:
        print(f"No images found in {low_quality_folder}")
        continue

    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    eval_images = images[split_index:]

    train_person_folder = os.path.join(train_lq_root, person)
    eval_person_folder = os.path.join(eval_lq_root, person)
    os.makedirs(train_person_folder, exist_ok=True)
    os.makedirs(eval_person_folder, exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(low_quality_folder, img),
                     os.path.join(train_person_folder, img))
    for img in eval_images:
        shutil.copy2(os.path.join(low_quality_folder, img),
                     os.path.join(eval_person_folder, img))

    print(f"{person}: {len(train_images)} images -> train_lq, {len(eval_images)} images -> eval_lq")

print("Split complete!")
