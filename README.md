# Face Mask Detector 😷

A real-time deep learning system for detecting face masks in images and video streams using Convolutional Neural Networks (CNN).

## 🎯 Overview

This project implements an intelligent face mask detection system leveraging TensorFlow/Keras and OpenCV to classify whether individuals are wearing masks in real-time. The model achieves high accuracy through advanced deep learning techniques including data augmentation and transfer learning.

## ✨ Key Features

- **Real-Time Detection**: Process video streams and webcam feeds with live mask detection
- **High Accuracy**: CNN-based architecture optimized for robust performance
- **Transfer Learning**: Leverages pre-trained models for improved accuracy
- **Data Augmentation**: Advanced preprocessing for handling diverse real-world scenarios
- **Easy to Use**: Simple inference pipeline for image and video inputs

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, NumPy
- **Model Architecture**: Convolutional Neural Networks (CNN)
- **Language**: Python 3.8+

## 📋 Requirements

```bash
pip install tensorflow keras opencv-python numpy
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Shivamstar394/face-mask-detector.git
cd face-mask-detector

# Run real-time detection on webcam
python realtime_mask_detect.py

# Train the model
python mask_train.ipynb
```

## 📂 Files

- **mask_train.ipynb** - Model training and evaluation notebook
- **realtime_mask_detect.py** - Real-time detection script for webcam/video
- **README.md** - Project documentation

## 📊 Model Performance

- Achieves high accuracy on diverse datasets
- Optimized for real-time inference
- Robust across various lighting conditions and face angles

## 🏗️ Architecture

The CNN model features:
- Multiple convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Dropout regularization to prevent overfitting
- Dense output layers for binary classification (mask/no mask)

## 🔧 Model Training

The `mask_train.ipynb` notebook includes:
- Data preprocessing and augmentation
- Model architecture definition
- Training with validation
- Performance evaluation and metrics

## 📈 Future Enhancements

- Multi-face detection and tracking
- Real-time statistics dashboard
- Integration with surveillance systems
- Edge deployment optimization
- Mobile app implementation

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
