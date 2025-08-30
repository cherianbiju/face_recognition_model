# FACE RECOGNITION <img width="400" height="312" alt="image" src="https://github.com/user-attachments/assets/f118e78f-1331-453b-b768-7036e57f1266" />

 

A project demonstrating face recognition on low-quality images using **FaceNet embeddings** and **SVM classifiers**. The project includes fine-tuning the SVM on mixed-quality embeddings to improve performance on degraded images.

---

## Project Objective

- Implement a pre-trained FaceNet model to extract discriminative embeddings from facial images.
- Train and evaluate a **Support Vector Machine (SVM)** classifier on high-quality (HQ) and low-quality (LQ) images.
- Fine-tune the SVM using a mixed-quality dataset to improve classification accuracy on low-quality images.
- Provide detailed evaluation metrics including accuracy, F1-score, and confusion matrices.

---

## Directory Structure

face_recognition_project/
├── src/ # Source code scripts
│ ├── extract_embeddings.py
│ ├── hqsvm_training.py
│ ├── finetune_svm.py
│ ├── evaluate_svm.py
│ └── plot_confusion.py
├── data/ # Dataset (not included in repo)
│ ├── train/ # High-quality training images
│ ├── train_lq/ # Low-quality training images for fine-tuning
│ └── eval_lq/ # Low-quality evaluation images
├── images/ # Generated images (confusion matrices, ROC curves)
├── models/ # Saved SVM models (ignored in repo)
└── README.md


---

## Dataset Instructions

- **High-Quality (HQ) Images:** Used to train the initial SVM.  
- **Low-Quality (LQ) Images:** Split into:
  - `train_lq/` → used for SVM fine-tuning
  - `eval_lq/` → used strictly for evaluation (never for training)  

> ⚠️ The dataset is not included in the repository.



