import os
import shutil
import pandas as pd

# Paths
CSV_PATH = "low_quality_predictions.csv"  # your CSV from step 4
IMAGE_DIR = r"data/eval_lq"               # original eval_lq folder
FAILED_DIR = r"data/failed_predictions"   # folder to copy failed predictions

# Create folder if it doesn't exist
os.makedirs(FAILED_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)

# Filter failed predictions
failed = df[df["predicted"] != df["actual"]]

for _, row in failed.iterrows():
    person_folder = os.path.join(IMAGE_DIR, row["actual"])
    src_path = os.path.join(person_folder, row["image"])
    dst_path = os.path.join(FAILED_DIR, row["image"])
    
    try:
        shutil.copy(src_path, dst_path)  # copy, do not cut
        print(f"Copied: {row['image']}")
    except Exception as e:
        print(f"❌ Error copying {row['image']}: {e}")

print(f"\n✅ Copied {len(failed)} failed prediction images to {FAILED_DIR}")
