"""
MRI Image Preprocessing Module
==============================
This module handles all MRI image preprocessing operations.
For educational/research purposes only.

Preprocessing Steps:
1. Load image from file
2. Resize to consistent dimensions (224x224)
3. Convert grayscale to RGB (if needed)
4. Normalize pixel values to [0, 1] range
"""

import numpy as np
from PIL import Image
import os
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS


def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single MRI image.
    
    Parameters:
    -----------
    image_path : str
        Path to the MRI image file
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed image array with shape (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        
    Explanation for Academic Report:
    --------------------------------
    MRI preprocessing is crucial for CNN-based analysis because:
    1. Standardization: CNNs require fixed input dimensions
    2. Normalization: Scaling pixel values to [0,1] helps gradient-based optimization
    3. Channel conversion: Ensures consistent 3-channel input for the model
    """
    # Load image using PIL
    img = Image.open(image_path)
    
    # Resize to target dimensions
    # Bilinear interpolation preserves image quality during resizing
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    
    # Convert to RGB if grayscale (MRI images are typically grayscale)
    # CNN expects 3 channels, so we replicate grayscale across RGB channels
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize pixel values to [0, 1] range
    # Original pixel values are 0-255, dividing by 255 normalizes them
    img_array = img_array / 255.0
    
    return img_array


def preprocess_for_prediction(image_path):
    """
    Preprocess an image for model prediction.
    
    Parameters:
    -----------
    image_path : str
        Path to the MRI image file
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed image array with shape (1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        The batch dimension is added for model compatibility
    """
    img_array = load_and_preprocess_image(image_path)
    
    # Add batch dimension (model expects batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def load_dataset(dataset_path):
    """
    Load the entire dataset from directory structure.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset directory containing 'normal' and 'parkinson' folders
        
    Returns:
    --------
    tuple
        (images, labels) where:
        - images: numpy array of shape (N, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        - labels: numpy array of shape (N,) with 0=Healthy, 1=Parkinson
        
    Directory Structure Expected:
    -----------------------------
    dataset_path/
    ├── normal/      (healthy brain MRI scans)
    │   ├── image1.png
    │   └── ...
    └── parkinson/   (Parkinson's affected brain MRI scans)
        ├── image1.png
        └── ...
    """
    images = []
    labels = []
    
    # Define class folders and their corresponding labels
    class_folders = {
        'normal': 0,      # Label 0 for healthy
        'parkinson': 1    # Label 1 for Parkinson's
    }
    
    for class_name, label in class_folders.items():
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: Class folder '{class_name}' not found at {class_path}")
            continue
            
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        print(f"Loading {len(image_files)} images from '{class_name}' class...")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                img_array = load_and_preprocess_image(img_path)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total images: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Healthy samples: {np.sum(labels == 0)}")
    print(f"Parkinson samples: {np.sum(labels == 1)}")
    
    return images, labels


def get_class_weights(labels):
    """
    Calculate class weights to handle class imbalance.
    
    Parameters:
    -----------
    labels : numpy.ndarray
        Array of labels
        
    Returns:
    --------
    dict
        Dictionary mapping class indices to weights
        
    Explanation for Academic Report:
    --------------------------------
    Class imbalance (more samples of one class than another) can bias
    the model towards the majority class. Class weights penalize
    misclassification of the minority class more heavily, encouraging
    the model to learn from both classes equally.
    
    Formula: weight_i = n_samples / (n_classes * n_samples_i)
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get unique classes
    classes = np.unique(labels)
    
    # Compute balanced class weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    # Create dictionary mapping class index to weight
    class_weights = dict(zip(classes, weights))
    
    print(f"\nClass weights calculated:")
    print(f"  Healthy (0): {class_weights[0]:.4f}")
    print(f"  Parkinson (1): {class_weights[1]:.4f}")
    
    return class_weights


if __name__ == "__main__":
    """
    Test preprocessing functions when running this file directly.
    """
    from config import DATASET_DIR
    
    print("=" * 60)
    print("MRI PREPROCESSING MODULE TEST")
    print("=" * 60)
    
    # Test loading a single image
    test_image_path = os.path.join(DATASET_DIR, 'normal', 'Mag_Images_001.png')
    
    if os.path.exists(test_image_path):
        print(f"\nTesting single image preprocessing...")
        img = load_and_preprocess_image(test_image_path)
        print(f"  Image shape: {img.shape}")
        print(f"  Pixel value range: [{img.min():.2f}, {img.max():.2f}]")
        print(f"  Data type: {img.dtype}")
        
        # Test prediction preprocessing
        img_batch = preprocess_for_prediction(test_image_path)
        print(f"\nPrediction batch shape: {img_batch.shape}")
    
    # Test loading full dataset
    print("\n" + "=" * 60)
    print("Testing full dataset loading...")
    print("=" * 60)
    
    images, labels = load_dataset(DATASET_DIR)
    class_weights = get_class_weights(labels)
