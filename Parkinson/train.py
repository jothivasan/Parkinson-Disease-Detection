"""
Training Module for Parkinson's Disease Detection CNN
=====================================================
This module handles the complete training pipeline including:
1. Dataset loading and preprocessing
2. Data splitting (train/validation)
3. Data augmentation
4. Model training with callbacks
5. Training visualization
For educational/research purposes only.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    DATASET_DIR, MODEL_DIR, LOGS_DIR,
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    AUGMENTATION_PARAMS, MODEL_CHECKPOINT_PATH
)
from preprocessing import load_dataset, get_class_weights
from model import create_cnn_model, get_callbacks, print_model_summary


def create_data_generators():
    """
    Create data generators with augmentation for training.
    
    Returns:
    --------
    tuple
        (train_datagen, val_datagen) - ImageDataGenerator objects
        
    Data Augmentation Explanation (for Viva/Report):
    ------------------------------------------------
    Data augmentation artificially increases the training dataset by
    applying random transformations to images. This helps:
    
    1. Prevent Overfitting: More diverse training data reduces
       the model's tendency to memorize training examples
       
    2. Improve Generalization: Model learns to recognize patterns
       regardless of orientation, position, or scale
       
    3. Medical Imaging Specific: Only realistic transformations are
       applied (no vertical flip for brain scans, limited rotation)
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=AUGMENTATION_PARAMS['rotation_range'],
        width_shift_range=AUGMENTATION_PARAMS['width_shift_range'],
        height_shift_range=AUGMENTATION_PARAMS['height_shift_range'],
        zoom_range=AUGMENTATION_PARAMS['zoom_range'],
        horizontal_flip=AUGMENTATION_PARAMS['horizontal_flip'],
        fill_mode=AUGMENTATION_PARAMS['fill_mode']
    )
    
    # Validation data generator - NO augmentation
    # Validation data should represent real-world data without modifications
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', 
                 color='#2E86AB', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', 
                 color='#A23B72', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Plot Loss
    axes[1].plot(history.history['loss'], label='Training Loss', 
                 color='#2E86AB', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', 
                 color='#A23B72', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
    
    plt.show()


def train_model():
    """
    Main training function that orchestrates the entire training pipeline.
    
    Returns:
    --------
    tuple
        (model, history) - Trained model and training history
        
    Training Pipeline Explanation (for Viva/Report):
    ------------------------------------------------
    
    1. Data Loading:
       - Load all MRI images from normal/parkinson folders
       - Preprocess images (resize, normalize)
       
    2. Data Splitting:
       - 80% training, 20% validation (stratified)
       - Stratified split ensures class balance in both sets
       
    3. Class Weight Calculation:
       - Compute weights to handle class imbalance
       - Higher weight for minority class (Parkinson)
       
    4. Data Augmentation:
       - Apply random transformations to training data
       - No augmentation for validation data
       
    5. Model Training:
       - Train with callbacks (early stopping, checkpointing)
       - Monitor validation metrics
       
    6. Result Visualization:
       - Plot accuracy and loss curves
       - Save best model
    """
    
    print("=" * 70)
    print("PARKINSON'S DISEASE MRI DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    print("For educational/research purposes only")
    print("=" * 70)
    
    # ========================================
    # STEP 1: Load Dataset
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 1: Loading Dataset")
    print("-" * 50)
    
    images, labels = load_dataset(DATASET_DIR)
    
    # ========================================
    # STEP 2: Split Data
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 2: Splitting Data into Train/Validation Sets")
    print("-" * 50)
    
    # Stratified split to maintain class balance
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=VALIDATION_SPLIT,
        random_state=42,        # For reproducibility
        stratify=labels         # Maintain class balance
    )
    
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Training set - Healthy: {np.sum(y_train == 0)}, Parkinson: {np.sum(y_train == 1)}")
    print(f"Validation set - Healthy: {np.sum(y_val == 0)}, Parkinson: {np.sum(y_val == 1)}")
    
    # ========================================
    # STEP 3: Calculate Class Weights
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 3: Computing Class Weights for Imbalance Handling")
    print("-" * 50)
    
    class_weights = get_class_weights(y_train)
    
    # ========================================
    # STEP 4: Create Data Generators
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 4: Creating Data Generators with Augmentation")
    print("-" * 50)
    
    train_datagen, val_datagen = create_data_generators()
    
    # Fit the training generator to compute statistics
    train_datagen.fit(X_train)
    
    print("Training data augmentation enabled:")
    for key, value in AUGMENTATION_PARAMS.items():
        print(f"  - {key}: {value}")
    
    # ========================================
    # STEP 5: Create Model
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 5: Creating CNN Model")
    print("-" * 50)
    
    model = create_cnn_model()
    print_model_summary(model)
    
    # ========================================
    # STEP 6: Train Model
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 6: Training Model")
    print("-" * 50)
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print("\nTraining started...\n")
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Create generators for training
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Calculate steps per epoch
    # Use ceil to avoid dropping the final partial batch.
    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # ========================================
    # STEP 7: Evaluate and Save
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 7: Final Evaluation")
    print("-" * 50)
    
    # Evaluate on validation set
    val_metrics = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    val_loss = float(val_metrics.get('loss', 0.0))
    val_accuracy = float(val_metrics.get('accuracy', 0.0))
    
    print(f"\nFinal Validation Results:")
    print(f"  Validation Loss:     {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, 'final_model.keras')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to: {MODEL_CHECKPOINT_PATH}")
    
    # ========================================
    # STEP 8: Plot Training History
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 8: Generating Training Visualization")
    print("-" * 50)
    
    plot_path = os.path.join(LOGS_DIR, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return model, history


if __name__ == "__main__":
    """
    Run training when this file is executed directly.
    """
    model, history = train_model()
