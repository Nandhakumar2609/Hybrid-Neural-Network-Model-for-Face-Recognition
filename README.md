
# Project Report — Facial Recognition using ANN + CNN

## 1. Project Overview
This project implements facial recognition using a hybrid deep learning model that combines Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The aim is to detect and recognize faces in images or videos efficiently and accurately.

---

## 2. Dataset
- The dataset consists of facial images organized per individual (e.g., `data/<person_name>/*.jpg`).
- Images are preprocessed (cropped, aligned, and resized) for model training

---

## 3. Methodology

### Step 1: Data Preprocessing
- Images are loaded and converted to grayscale or RGB.
- Faces are detected using **OpenCV Haar Cascade** or the **`face_recognition`** library.
- Cropped faces are resized to a fixed shape (e.g., 128x128).
- Pixel values are normalized (0–1 range).

### Step 2: Feature Extraction (CNN)
- Convolutional layers extract spatial features.
- Pooling layers reduce dimensionality.
- Flattening layer converts images to feature vectors.

### Step 3: Classification (ANN)
- Dense layers process extracted features.
- The output layer predicts the identity/class of the face.
- Loss: categorical cross-entropy.
- Optimizer: Adam.

### Step 4: Model Evaluation
- Train/test split: typically 80/20.
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

---

## 4. Results (Example Template)
*(Replace with actual results from your notebook)*

| Metric | Score |
|--------|--------|
| Accuracy | 92.3% |
| Precision | 90.1% |
| Recall | 91.4% |
| F1-score | 90.7% |

- The model performed well on clear and front-facing images.
- Accuracy decreased in cases with occlusions or poor lighting.

---

## 5. Tools & Technologies
- **Python**
- **OpenCV** for face detection
- **NumPy**, **Pandas**, **Matplotlib**
- **TensorFlow / PyTorch**
- **scikit-learn** for evaluation
- **face-recognition** library (optional)

---

## 7. Future Improvements
- Implement transfer learning (e.g., ResNet, VGGFace, or MobileNet).
- Add real-time video recognition.
- Improve accuracy with data augmentation.
- Explore embedding-based approaches (FaceNet, ArcFace).

---

## 8. References
- https://pytorch.org/
- https://opencv.org/
- https://scikit-learn.org/
- https://github.com/ageitgey/face_recognition

- --------------------------



