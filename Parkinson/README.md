# 🧠 Parkinson's Disease MRI Detection System

A CNN-based deep learning system for detecting Parkinson's Disease from brain MRI scans.

**Final Year Academic Project** | For educational and research purposes only.

---

## ⚠️ Medical Disclaimer

**This is an AI-based educational tool for research purposes only. It should NOT be used for medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage Instructions](#usage-instructions)
6. [CNN Model Architecture](#cnn-model-architecture)
7. [Dataset Information](#dataset-information)
8. [For Your Viva](#for-your-viva)

---

## 📖 Project Overview

This system analyzes brain MRI scans using a Convolutional Neural Network (CNN) to classify images as either **Healthy** or **Parkinson's Disease**.

### Key Features

| Feature               | Description                                        |
| --------------------- | -------------------------------------------------- |
| **MRI Preprocessing** | Resize to 224×224, normalize pixel values          |
| **CNN Architecture**  | 4 Conv blocks with BatchNorm and Dropout           |
| **Training**          | Data augmentation, class weights, early stopping   |
| **Evaluation**        | Confusion matrix, ROC curve, classification report |
| **Web Interface**     | Modern Flask UI with drag-and-drop upload          |

---

## 📁 Project Structure

```
Parkinson/
│
├── parkinsons_dataset/          # MRI Dataset
│   ├── normal/                  # 610 healthy brain MRI scans
│   └── parkinson/               # 221 Parkinson's MRI scans
│
├── models/                      # Trained Models
│   ├── best_model.keras         # Best model checkpoint
│   └── final_model.keras        # Final trained model
│
├── logs/                        # Training Outputs
│   ├── training_history.png     # Accuracy/Loss curves
│   ├── confusion_matrix.png     # Confusion matrix
│   └── roc_curve.png            # ROC curve
│
├── webapp/                      # Flask Web Application
│   ├── static/
│   │   ├── css/style.css        # Stylesheet
│   │   └── uploads/             # Uploaded images
│   ├── templates/
│   │   ├── index.html           # Home page
│   │   ├── result.html          # Results page
│   │   └── about.html           # About page
│   └── app.py                   # Flask app
│
├── config.py                    # Configuration settings
├── preprocessing.py             # Image preprocessing
├── model.py                     # CNN architecture
├── train.py                     # Training pipeline
├── evaluate.py                  # Model evaluation
├── predict.py                   # Inference module
├── run.py                       # ⭐ MAIN ENTRY POINT
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Navigate to Project

```bash
cd c:\Users\jothi\Desktop\Parkinson
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you encounter long path issues on Windows, packages may be installed to `C:\tf_libs`. Set the PYTHONPATH accordingly.

---

## 🚀 Quick Start

### Run the Application

```bash
python run.py
```

This opens an interactive menu:

1. **Train the CNN model** - First time setup
2. **Evaluate the trained model** - Generate metrics
3. **Make a prediction** - CLI inference
4. **Start the web application** - Launch UI
5. **View project structure** - See files
6. **Exit**

### Direct Web App Access

```bash
cd webapp
python app.py
```

Open: **http://127.0.0.1:5000**

---

## 📚 Usage Instructions

### Training (First Time Only)

```bash
python train.py
```

- Duration: 10-30 minutes
- Output: Models saved to `models/`
- Visualizations saved to `logs/`

### Evaluation

```bash
python evaluate.py
```

Generates confusion matrix, ROC curve, and classification report.

### Command-Line Prediction

```bash
python predict.py path/to/mri_image.png
```

### Web Application

```bash
cd webapp && python app.py
```

- Navigate to http://127.0.0.1:5000
- Upload MRI image
- View prediction with confidence score

---

## 🧠 CNN Model Architecture

```
Input (224 × 224 × 3)
        ↓
Conv2D (32) → BatchNorm → MaxPool
        ↓
Conv2D (64) → BatchNorm → MaxPool
        ↓
Conv2D (128) → BatchNorm → MaxPool
        ↓
Conv2D (256) → BatchNorm → MaxPool
        ↓
Flatten → Dense (256) → Dropout (50%)
        ↓
Dense (1, Sigmoid) → Output
```

### Key Design Decisions

| Component              | Purpose                       |
| ---------------------- | ----------------------------- |
| **4 Conv Blocks**      | Hierarchical feature learning |
| **BatchNormalization** | Training stability            |
| **MaxPooling**         | Dimensionality reduction      |
| **Dropout (50%)**      | Prevent overfitting           |
| **Sigmoid Output**     | Binary probability [0, 1]     |

---

## 📊 Dataset Information

| Class            | Count   | Percentage |
| ---------------- | ------- | ---------- |
| Healthy (Normal) | 610     | 73.4%      |
| Parkinson        | 221     | 26.6%      |
| **Total**        | **831** | 100%       |

### Handling Class Imbalance

- **Class Weights**: Higher penalty for minority class
- **Stratified Split**: Maintains ratio in train/val sets
- **Calibrated Threshold**: 0.35 instead of 0.5

---

## 🎓 For Your Viva

### Key Talking Points

1. **Why CNN for MRI?**
   - Automatic hierarchical feature extraction
   - Translation invariance for detecting patterns anywhere in image
   - Proven success in medical imaging

2. **Preprocessing Steps**
   - Resize to 224×224 (fixed input size)
   - Normalize to [0, 1] (better gradient optimization)
   - Convert to RGB (3-channel input)

3. **Overfitting Prevention**
   - Data augmentation (rotation, shift, zoom, flip)
   - Dropout regularization (50%)
   - Early stopping (patience=10)

4. **Threshold Calibration**
   - Default 0.5 caused class imbalance issues
   - Calibrated to 0.35 using Youden's J statistic
   - Balances sensitivity and specificity

### Common Viva Questions

| Question                    | Answer                                                     |
| --------------------------- | ---------------------------------------------------------- |
| Why Sigmoid activation?     | Maps output to probability [0,1] for binary classification |
| What is BatchNormalization? | Normalizes layer inputs, stabilizes training               |
| Why stratified split?       | Maintains class proportions in train/val sets              |
| What does AUC represent?    | Model's ability to distinguish classes (1.0 = perfect)     |

---

## 📋 Files Summary

| File               | Purpose                           |
| ------------------ | --------------------------------- |
| `run.py`           | **Main entry point** - Start here |
| `config.py`        | All configuration parameters      |
| `preprocessing.py` | Image preprocessing functions     |
| `model.py`         | CNN architecture definition       |
| `train.py`         | Model training pipeline           |
| `evaluate.py`      | Metrics and visualizations        |
| `predict.py`       | Inference on new images           |
| `webapp/app.py`    | Flask web application             |

---

## ✅ Pre-Demo Checklist

- [ ] Dependencies installed
- [ ] Model trained (`models/` has `.keras` files)
- [ ] Evaluation run (`logs/` has `.png` files)
- [ ] Web app tested with sample images
- [ ] README reviewed

---

## 📜 License & Disclaimer

**For educational and research purposes only.**

Not intended for clinical use or medical diagnosis. Always consult qualified healthcare professionals.

---

_Final Year Academic Project - CNN-Based Medical Image Analysis_
