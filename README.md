# 🧠 Diabetic Retinopathy Detection Using Deep Learning

This repository contains a deep learning project for the automated classification of Diabetic Retinopathy (DR) stages using high-resolution retinal fundus images. The project evaluates multiple convolutional neural network (CNN) architectures and uses transfer learning for improved accuracy and clinical relevance.

## 📌 Project Objective
To build, train, and compare several CNN models for multi-class classification of DR severity from fundus images. The best-performing model is selected based on metrics such as accuracy, AUC-ROC, and F1-score.

---

## 🗂️ Dataset
- **Name**: APTOS 2019 Blindness Detection
- **Source**: [Kaggle Dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
- **Classes**:
  - 0: No DR
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Proliferative DR
- **Preprocessing**:
  - Resizing
  - Normalization
  - Extensive Data Augmentation (rotation, zoom, brightness, flip, etc.)
  - Class balancing using weights

---

## 🏗️ Models Used
The following pre-trained models were fine-tuned:
- ResNet-50
- InceptionV3
- Xception
- DenseNet121 ✅ (Best performance)

### 🧪 Training Strategy
- Phase 1: Feature extraction (freeze base model)
- Phase 2: Fine-tuning (unfreeze top layers)
- Loss function: Categorical Crossentropy + L2 Regularization
- Optimizer: Adam
- EarlyStopping and ReduceLROnPlateau used

---

## 📊 Results

| Model       | Accuracy | AUC-ROC |
|-------------|----------|---------|
| ResNet-50   | ~63%     | ~83%    |
| InceptionV3 | ~68%     | ~86%    |
| Xception    | ~66%     | ~85%    |
| DenseNet121 | **72.58%** | **90.01%** ✅ |

- **Confusion Matrix** and **Training Curves** are available in the `results/` folder.

---

## 📈 Visuals
- Confusion matrices per model
- Training and validation accuracy/loss plots
- ROC curves (optional)

---

## 💡 Key Findings
- DenseNet121 achieved the highest performance and most consistent results.
- AUC-ROC was significantly higher than accuracy, highlighting strong class-separation ability.
- Class imbalance and image noise affected minority class detection.

---

## 🧰 Tools and Libraries
- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

---

## 📁 Folder Structure

├── dataset/
│ └── preprocessed_images/
├── models/
│ └── trained_weights/
├── notebooks/
│ └── model_training.ipynb
├── results/
│ └── confusion_matrices/
│ └── training_curves/
├── README.md
└── requirements.txt


---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dr-detection-project.git
   cd dr-detection-project
2. Install dependencies:
pip install -r requirements.txt
3. Run the training notebook:
jupyter notebook notebooks/model_training.ipynb

🙋 Author
Omar Ibrahim
AI Engineering Student
Email: oelwan514@gmail.com

📄 License
This project is for educational and research purposes only.

---

Let me know if you want a version with badges (e.g., Python version, license, stars) or GitHub Actions CI setup.

