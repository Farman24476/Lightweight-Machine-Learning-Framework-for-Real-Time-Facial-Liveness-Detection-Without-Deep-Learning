# Lightweight-Machine-Learning-Framework-for-Real-Time-Facial-Liveness-Detection-Without-Deep-Learning
This project introduces a CNN-free facial liveness detection system that leverages handcrafted feature extraction methods—Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and Gabor filters—combined with ensemble machine learning classifiers (SVM, Random Forest, XGBoost). It aims to provide a lightweight, efficient, and highly accurate solution for real-time face anti-spoofing in biometric authentication systems.

## Project Highlights
- Completely **deep learning-free** facial liveness detection.
- Uses **handcrafted features** (HOG, LBP, Gabor) for spoof vs. real face discrimination.
- Employs **ensemble ML models**: SVM, Random Forest, and XGBoost.
- Achieves **up to 100% accuracy** on benchmark datasets.
- Designed for **real-time, low-resource environments** (IoT, mobile devices).

## Data Source
- **iBeta Level-1 Liveness Detection Dataset (Part 1)**:  
  - Contains **42,280 videos** of both real and spoofing attempts (print, replay, mask, etc.)  
  - Total size ~338 GB  
  - Available on Kaggle: [iBeta Level-1 Liveness Detection Dataset – Part 1](https://www.kaggle.com/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1) :contentReference[oaicite:1]{index=1}

## Problem Statement
While CNN-based liveness detection systems offer strong performance, they suffer from high computational cost, poor generalization, and the need for large training datasets. This project addresses these limitations by building a machine learning model that:
- Is lightweight and scalable.
- Works effectively with small datasets.
- Is robust to multiple spoofing methods (print, video, 3D mask).

## Methodology
1. **Dataset**: iBeta Level-1 Liveness Detection Dataset  
2. **Preprocessing**:
   - Convert videos to grayscale frames
   - Resize to 128×128 pixels
   - Normalize with StandardScaler
   - Balance classes using SMOTE
3. **Feature Extraction**:
   - LBP for texture features
   - HOG for edge and shape features
   - Gabor filters for spatial frequency analysis
4. **Model Training**:
   - Train on 80% of data, test on 20%
   - Classifiers: SVM (RBF kernel), Random Forest, XGBoost
   - Model evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
5. **Real-Time Implementation**:
   - Frame-based prediction with OpenCV pipeline
   - Deployment-ready using `joblib` for saved models

## Results
| Classifier     | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| SVM            | 100%     | 1.00      | 1.00   | 1.00     |
| XGBoost        | 99%      | 0.99      | 0.99   | 0.99     |
| Random Forest  | 98%      | 0.98      | 0.98   | 0.98     |

- SVM showed the best performance, making it ideal for real-time deployment.
- XGBoost was nearly as accurate, offering a good trade-off for large-scale systems.

## Key Contributions
- Proposes a CNN-free anti-spoofing solution for face recognition systems.
- Demonstrates high accuracy with low computational cost.
- Enhances generalization across spoofing types without deep learning.

## Future Work
- Incorporate motion-based temporal features (e.g., blink detection).
- Expand to multimodal datasets (thermal, IR, depth).
- Improve model optimization for deployment on edge devices.
- Explore resistance to adversarial attacks (e.g., GAN-based deepfakes).

## Technologies Used
- Language: Python
- Libraries: OpenCV, scikit-learn, XGBoost, joblib, imbalanced-learn

## Institution
School of Computing and Informatics, Albukhary International University, Alor Setar, Kedah, Malaysia.

## License
This project is for academic and research purposes. Please cite appropriately if used.
