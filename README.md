# Parkinson Disease Detection

> CNN-based brain MRI image classification for Parkinson's disease research and education.

**Status:** Public · **License:** MIT

## Overview

This project demonstrates an end-to-end computer vision workflow for binary classification of brain MRI images into two classes: `Healthy` and `Parkinson`. It includes image preprocessing, CNN model training, evaluation, calibrated prediction, and a Flask web interface for uploading an MRI image and viewing the result.

> **Important:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or make clinical decisions about Parkinson's disease.

## Features

- CNN-based binary MRI image classification
- Standardized image resizing to `224 x 224 x 3`
- Pixel normalization and grayscale-to-RGB conversion
- Data augmentation for improved training robustness
- Class weighting for imbalanced datasets
- Early stopping, model checkpoints, and learning-rate reduction
- Accuracy, AUC, precision, recall, confusion matrix, and ROC evaluation
- Calibrated prediction threshold support
- Flask web interface for image upload and prediction results
- Command-line workflow for training, evaluation, prediction, and web serving

## Model Workflow

```text
MRI image
   ↓
Resize and normalize
   ↓
CNN feature extraction
   ↓
Binary classification
   ↓
Healthy / Parkinson prediction with confidence
```

The CNN uses convolutional blocks with batch normalization and max pooling, followed by a dense classification layer, dropout regularization, and a sigmoid output for binary prediction.

## Project Structure

```text
.
├── Parkinson/
│   ├── config.py                 # Paths, classes, model, and training settings
│   ├── preprocessing.py          # Image loading and dataset preparation
│   ├── model.py                  # CNN architecture and training callbacks
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Metrics and evaluation visualizations
│   ├── predict.py                # Inference and thresholded predictions
│   ├── run.py                    # Interactive command-line entry point
│   ├── requirements.txt          # Python dependencies
│   └── webapp/
│       └── app.py                # Flask prediction interface
├── Parkinson/README.md           # Project-level documentation
└── LICENSE
```

Generated or large assets such as datasets, trained models, uploads, and logs are intentionally excluded from version control. See the configuration file for expected paths.

## Requirements

- Python 3.8 or newer
- TensorFlow 2.10 or newer
- A compatible environment for TensorFlow and Pillow
- Brain MRI image dataset arranged into `normal/` and `parkinson/` folders

## Installation

```bash
git clone https://github.com/jothivasan/Parkinson-Disease-Detection.git
cd Parkinson-Disease-Detection/Parkinson
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Dataset Layout

Place the training images in the following structure:

```text
Parkinson/
└── parkinsons_dataset/
    ├── normal/
    │   ├── image_001.png
    │   └── ...
    └── parkinson/
        ├── image_001.png
        └── ...
```

The class mapping is:

- `normal` → `Healthy` (`0`)
- `parkinson` → `Parkinson` (`1`)

## Usage

From the `Parkinson/` directory:

### Train the CNN

```bash
python train.py
```

### Evaluate the model

```bash
python evaluate.py
```

Evaluation outputs are written to the `logs/` directory, including confusion-matrix and ROC-curve visualizations.

### Predict one MRI image

```bash
python predict.py path/to/mri-image.png
```

### Start the web application

```bash
python run.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The web app accepts PNG, JPG, JPEG, GIF, and BMP images and exposes a prediction endpoint at `POST /api/predict`.

## Configuration

Key settings are defined in `Parkinson/config.py`:

- Image size and channel configuration
- Class labels and folder mapping
- Batch size, epochs, learning rate, and validation split
- Data augmentation parameters
- Model checkpoint and threshold paths
- Flask host, port, and upload limits

## License

This project is available under the [MIT License](LICENSE).
