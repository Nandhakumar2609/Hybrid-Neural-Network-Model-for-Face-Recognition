🧠 Hybrid Neural Network Model for Face Recognition

A deep learning-based facial recognition system that combines Convolutional Neural Networks (CNN) for facial feature extraction and Artificial Neural Networks (ANN) for classification.
This hybrid model enhances accuracy and performs efficiently on both image-based and real-time video facial recognition.

-------------------------------------------------------------

📌 Objectives

1.Detect and recognize human faces from images and live video streams.

2.Extract deep facial features using CNN layers.

3.Classify identities through ANN layers.

4.Evaluate system performance with key metrics (accuracy, precision, recall, F1-score).

-------------------------------------------------------

🧩 Methodology

1.Data Preprocessing:

Face detection using OpenCV (Haar Cascade/Face Detector).

Cropping, resizing, and normalizing images.

2.Model Architecture:

CNN: Extracts spatial facial features.

ANN: Classifies extracted embeddings.

3.Training & Evaluation:

Optimizer: Adam

Loss Function: Categorical Crossentropy

Achieved ~92% accuracy

4.Video Testing:

Real-time face recognition using OpenCV VideoCapture.

Continuous frame detection, feature extraction, and classification.

-------------------------------------------------------------

🛠️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

scikit-learn

-------------------------------------------------

📊 Results

Model Accuracy: ~92%

Works effectively on static images and real-time videos.

Demonstrated robust recognition performance under varying lighting conditions.

----------------------------------
## 5. Tools & Technologies
- **Python**
- **OpenCV** for face detection
- **NumPy**, **Pandas**, **Matplotlib**
- **TensorFlow / PyTorch**
- **scikit-learn** for evaluation
- **face-recognition** library (optional)

-----------------------------------

🚀 Future Enhancements

1.Implement transfer learning using VGGFace or ResNet.

2.Deploy the model via Flask or Streamlit for web applications.

3.Add emotion detection and face embeddings (FaceNet, ArcFace).

-------------------------------------------

## 8. References
- https://pytorch.org/
- https://opencv.org/
- https://scikit-learn.org/
- https://github.com/ageitgey/face_recognition

- --------------------------



