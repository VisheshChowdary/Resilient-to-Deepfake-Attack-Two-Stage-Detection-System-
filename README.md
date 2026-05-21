# Resilient to Deepfake Attack : Two-Stage Detection System

Resilient to Deepfake Attack is a deep learning–based face forgery detection system designed to identify manipulated facial content with high accuracy and efficiency. The project combines CNN-based feature extraction with Kolmogorov–Arnold Networks (KAN) classifiers and introduces a fusion strategy for improved robustness against unseen deepfake attacks.

---

# Features

## Deepfake Detection

- **Two-Stage Detection System:** Developed a two-stage deepfake face detection pipeline using CNN-based feature extraction and KAN classifiers.
- **High Accuracy:** Achieved up to **94.39% accuracy** on benchmark deepfake datasets.
- **Robust Forgery Detection:** Improved resistance against unseen manipulation techniques and forged facial content.

## KAN-Based Classification

- **KAN Integration:** Replaced traditional MLP classifiers with Kolmogorov–Arnold Network (KAN) layers.
- **Efficient Inference:** Reduced inference time by **26.6%** while maintaining or improving classification performance.
- **Memory Optimization:** Reduced GPU memory usage by **37.3%**.

## Fusion-Based Architecture

- **CNN Fusion Strategy:** Implemented a fusion architecture combining **ResNet-18** and **DenseNet-121**.
- **Weighted Fusion:** Applied a weighted fusion mechanism using **0.7 / 0.3 weighting** for enhanced feature representation.
- **Performance Improvement:** Improved classification accuracy by **1.34%** compared to standalone CNN models.

---

# Tech Stack

## Deep Learning Frameworks

- **Python**
- **PyTorch**

## CNN Architectures

- **ResNet-18**
- **DenseNet-121**

## Machine Learning

- **Kolmogorov–Arnold Networks (KAN)**
- **CNN Feature Extraction**
- **Weighted Feature Fusion**

## Libraries/Tools

- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **Torchvision**
- **OpenCV**

---

# Workflow

1. **Dataset Preparation:** Load and preprocess deepfake image datasets.
2. **Feature Extraction:** Extract facial features using CNN architectures (ResNet-18 and DenseNet-121).
3. **Feature Fusion:** Combine extracted features using weighted fusion techniques.
4. **KAN Classification:** Pass fused features through KAN-based classifiers for prediction.
5. **Evaluation:** Measure performance using accuracy, inference time, GPU usage, and robustness metrics.
