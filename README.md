# 🧠 CIFAR-10 Image Classifier Web App (PyTorch + Flask)

An end-to-end deep learning project where we trained a Convolutional Neural Network (CNN) on the CIFAR-10 dataset to classify real-world images into 10 categories. The final model achieves **86% test accuracy**, and we deployed it as a **Flask web application** with support for custom image uploads and predictions.

---

## ▶️ Web App Video Demo

> Watch a quick demonstration of the web application in action!

![Video Demo](static/demo.gif)

*Note: If the video doesn't display properly on GitHub, you can view it directly in the `static/demo.mp4` file or run the Flask app locally to see the embedded video.*

---

## 📌 Table of Contents

- [📊 Overview](#-overview)
- [🧠 Project Goals](#-project-goals)
- [📁 Dataset](#-dataset)
- [🛠️ Tools & Technologies](#️-tools--technologies)
- [🧪 Experiments & Model Evolution](#-experiments--model-evolution)
- [🚀 Final Model Architecture](#-final-model-architecture)
- [🖼 Web App Demo](#-web-app-demo)
- [💡 Key Learnings](#-key-learnings)
- [📦 Installation](#-installation)
- [🧩 Future Improvements](#-future-improvements)
- [🙌 Acknowledgements](#-acknowledgements)

---

## 📊 Overview

This project aims to solve the image classification problem using the CIFAR-10 dataset. We started with simple models and iteratively built better-performing CNN architectures through experiments, learning rate tuning, regularization, and augmentation. Our final model, `CIFAR10ModelV4`, achieved **~86% accuracy** and was deployed using **Flask** for interactive prediction.

---

## 🧠 Project Goals

- Build a deep learning model to classify CIFAR-10 images.
- Learn about architecture design, loss functions, and optimizers.
- Improve performance through tuning and data augmentation.
- Deploy the trained model on a Flask web app for real-world use.

---

## 📁 Dataset

**Dataset**: CIFAR-10  
**Source**: `torchvision.datasets.CIFAR10`

### Classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Each image is RGB with a shape of `(3, 32, 32)`.

---

## 🛠️ Tools & Technologies

| Purpose | Tools/Frameworks |
|---------|------------------|
| Programming | Python |
| Deep Learning | PyTorch, Torchvision |
| Web Framework | Flask |
| Web UI | HTML, CSS, JavaScript |
| Visualization | Matplotlib, mlxtend, Torchmetrics |
| Hosting | GitHub Pages (for repo), Flask (local app) |
| Environment | Conda |

---

## 🧪 Experiments & Model Evolution

### 1. 🧱 Baseline Fully Connected Model (`CIFAR10ModelV0`)
- Flattened input, two `Linear` layers
- No activation
- Result: **~37% accuracy**
- Issue: Lost spatial information

### 2. 🔁 Added ReLU Activations (`CIFAR10ModelV1`)
- Added non-linear layers
- Learning rate too high (0.1) → performance dropped
- Result: **~33% accuracy**

### 3. 🧠 TinyVGG CNN (`CIFAR10ModelV2`)
- Basic CNN with 2 Conv blocks
- Result: **~56% accuracy**
- Improvement via convolutional feature extraction

### 4. 🔧 Learning Rate Tuning
- Changed `lr` from `0.1` to `0.01`
- Trained for 20 epochs instead of 5
- Accuracy improved to **~60%**

### 5. 🌀 Data Augmentation
- Added `RandomHorizontalFlip`, `ColorJitter`, `RandomCrop`
- Helped reduce overfitting and generalize better

### 6. 🔬 Learning Rate Sweep
- Trained for 3 epochs on multiple LRs (`1e-1`, `1e-2`, `1e-3`)
- Chose LR that dropped loss fastest

### 7. 🧠 Advanced Deep CNN (`CIFAR10ModelV4`)
- 4 Conv blocks with:
  - Batch Normalization
  - ReLU activations
  - Dropout regularization
- Final classifier with:
  - Linear → ReLU → Dropout → Linear
- Result: **~86% accuracy on test set**

---

## 🚀 Final Model Architecture

```
Conv2d(3 → 32) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
Conv2d(32 → 64) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
Conv2d(64 → 128) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
Conv2d(128 → 256) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
→ Flatten → Linear(1024 → 512) → ReLU → Dropout → Linear(512 → 10)
```

---

## 🖼 Web App Demo

- Upload image via browser
- Prediction made using `model_9` (CIFAR10ModelV4)
- Responsive UI using HTML/CSS/JS
- Simple REST API via Flask

### To run locally:
```bash
python app.py
```

### Navigate to:
```
http://127.0.0.1:5000
```

---

## 💡 Key Learnings

- CNNs significantly outperform linear models for vision tasks
- BatchNorm and Dropout greatly improve generalization
- Data augmentation is essential to avoid overfitting
- Learning rate tuning is critical
- Flask makes model deployment simple and interactive

---

## 📦 Installation

### Clone the repo
```bash
git clone https://github.com/yourusername/cifar10-image-classifier.git
cd cifar10-image-classifier
```

### Create environment & install requirements
```bash
conda create -n cifar10env python=3.10 -y
conda activate cifar10env
pip install -r requirements.txt
```

### Run the web app
```bash
python app.py
```

### Then visit:
```
http://127.0.0.1:5000
```

---

## 🧩 Future Improvements

- Add Grad-CAM to visualize model decisions
- Add upload preview and confidence scores
- Support images of any size via resizing
- Deploy live using Render or HuggingFace Spaces

---

## 🙌 Acknowledgements

- Daniel Bourke's PyTorch Course (ZTM)
- CIFAR Dataset
- Built using 🧠, 🐍, and ☕