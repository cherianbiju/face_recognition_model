# FACE RECOGNITION 🎭

 

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


## Usage Instructions

Follow these steps to replicate the training and evaluation pipeline for face recognition.

---

### 1. Extract Embeddings

Use the pre-trained FaceNet model to extract embeddings from your dataset.
```
embeddings_extracttion_hq.py  ->  extract embeddings from High-quality training images
embeddings_extraction_train_lq.py  ->  extract embeddings from train_lq
embedding_extraction_eval_lq.py  ->  extract embeddings from eval_lq
```
Input: data/train, data/train_lq, or data/eval_lq folders
Output: .pkl files containing embeddings and labels 

2. Train HQ SVM

Train a linear SVM classifier on embeddings extracted from high-quality images.
```
python src/train_svm.py
```
Input: HQ embeddings 
Output: models/svm_model.pkl (HQ-trained SVM)

3. Fine-Tune SVM on Mixed-Quality Embeddings

Train a new SVM on a combination of high-quality and low-quality embeddings to improve performance on degraded images.
```
python src/finetuning_svm.py
```
Input: HQ embeddings + train_lq embeddings
Output: models/svm_model_finetuned.pkl (fine-tuned SVM)

4. Evaluate SVM on Low-Quality Images

Evaluate fine-tuned SVM on the eval_lq dataset.
```
python src/finetuned_svm_main_test.py
```
Output:

Accuracy & F1-score
Classification report
Confusion matrix (printed in console)

5. Plot Confusion Matrix and ROC Curves

Generate visualizations for the classifier’s performance.
```
python src/plot_finetuned_confusion.py
```
Output: PNG images in the images/ 


### Note 📌⚡

Always use the evaluation set eval_lq only for testing; do not use it in training.

