# 🧾 OCR Handwritten Digit Recognition — SVM & CNN (LeNet + STN)

## 📌 Overview

This project implements an **OCR pipeline** for handwritten digit recognition from scanned PDF forms.

It combines two different approaches:

* 🔹 **Classical Machine Learning**: HOG + SVM
* 🔹 **Deep Learning**: LeNet CNN enhanced with Spatial Transformer Network (STN)

The system processes PDF documents, extracts digit regions, preprocesses them, and predicts the corresponding digits.

---


## ⚙️ Features

* 📄 PDF → Image conversion
* 📐 Automatic page alignment using QR markers
* ✂️ Extraction of digit regions (bounding boxes)
* 🧼 Preprocessing (denoising, binarization, normalization)
* 🔍 Feature extraction using HOG
* 🤖 Classification using:

  * SVM (HOG features)
  * CNN (LeNet architecture)
  * STN for spatial invariance
* 📊 Batch processing & performance evaluation

---

## 🧠 Models

### 🔹 HOG + SVM

* Feature-based approach
* Fast and lightweight
* Good baseline performance

### 🔹 LeNet + STN

* Deep learning approach
* Handles distortions and misalignments
* Higher robustness on real scanned data

---

## 📂 Dataset

### Generate MNIST dataset

```bash
python src/generate_mnist_dataset.py
```

### Extract crops from PDFs

```bash
python src/extract_true_crops.py \
    --input data/scans \
    --json config/atomic-boxes.json \
    --output data/true_crops
```

---

## 🏋️ Training

### Train SVM model

```bash
python src/main.py
```

### Retrain with real data

```bash
python src/retrain.py \
    --mnist data/dataset \
    --crops data/true_crops \
    --model models/svm_retrained.joblib
```

---

## 🔍 Inference

### Predict a single image

```bash
python main.py --predict path/to/image.png
```

---

## 📦 Batch Processing

```bash
python src/batch_pipeline.py \
    --input data/scans \
    --json config/atomic-boxes.json \
    --model models/svm_retrained.joblib \
    --output data/results
```

---

## 📊 Results

* Accuracy: depends on dataset and training strategy
* SVM: fast inference (~ms per image)
* CNN + STN: more robust on noisy real-world data

---

## 🛠️ Technologies

* Python
* OpenCV
* NumPy
* Scikit-learn
* Scikit-image
* pdf2image
* PyTorch / TensorFlow (for CNN)

---

## 👨‍💻 Author

**Wali Wassim**
Engineering Student 

---

## 📄 License

This project is for academic and research purposes.
