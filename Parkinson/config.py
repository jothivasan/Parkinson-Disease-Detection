"""
Configuration file for Parkinson's Disease MRI Detection System
================================================================
This file contains all configurable parameters for the project.
For educational/research purposes only.
"""

import sys
import os

# Add custom package path for installed dependencies
CUSTOM_LIBS_PATH = r'C:\tf_libs'
if os.path.exists(CUSTOM_LIBS_PATH) and CUSTOM_LIBS_PATH not in sys.path:
    sys.path.insert(0, CUSTOM_LIBS_PATH)

# ========================================
# DIRECTORY PATHS
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'parkinsons_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
UPLOADS_DIR = os.path.join(BASE_DIR, 'webapp', 'static', 'uploads')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ========================================
# IMAGE PREPROCESSING PARAMETERS
# ========================================
# Input image size for the CNN model
# 224x224 is a standard size that balances detail preservation with computational efficiency
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3  # RGB channels (converted from grayscale MRI)

# ========================================
# CLASS MAPPING
# ========================================
# Binary classification: 0 = Healthy, 1 = Parkinson
CLASS_NAMES = ['Healthy', 'Parkinson']
CLASS_MAPPING = {
    'normal': 0,     # Healthy brain
    'parkinson': 1   # Parkinson's affected brain
}

# ========================================
# TRAINING PARAMETERS
# ========================================
# Batch size: Number of samples per gradient update
# Smaller batch sizes can help with generalization but increase training time
BATCH_SIZE = 32

# Number of training epochs
# Early stopping will prevent overfitting before reaching max epochs
EPOCHS = 80

# Validation split ratio
# 20% of training data used for validation
VALIDATION_SPLIT = 0.2

# Learning rate for Adam optimizer
# A moderate learning rate for stable convergence
LEARNING_RATE = 0.0001

# Early stopping patience
# Stop training if validation loss doesn't improve for this many epochs
EARLY_STOPPING_PATIENCE = 15

# Model checkpoint settings
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
THRESHOLD_CONFIG_PATH = os.path.join(MODEL_DIR, 'threshold_config.json')

# ========================================
# DATA AUGMENTATION PARAMETERS
# ========================================
# Data augmentation helps prevent overfitting and improves generalization
# These augmentations are appropriate for medical imaging
AUGMENTATION_PARAMS = {
    'rotation_range': 15,      # Random rotation up to 15 degrees
    'width_shift_range': 0.1,  # Random horizontal shift
    'height_shift_range': 0.1, # Random vertical shift
    'zoom_range': 0.1,         # Random zoom
    'horizontal_flip': True,   # Random horizontal flip
    'fill_mode': 'nearest'     # Fill mode for empty pixels
}

# ========================================
# MODEL ARCHITECTURE PARAMETERS
# ========================================
# Dropout rate for regularization
# Helps prevent overfitting by randomly setting inputs to 0
DROPOUT_RATE = 0.5

# Dense layer units
DENSE_UNITS = 256

# ========================================
# FLASK WEB APPLICATION SETTINGS
# ========================================
FLASK_DEBUG = False
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Maximum upload file size (16 MB)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
