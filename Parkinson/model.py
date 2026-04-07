"""
CNN Model Architecture for Parkinson's Disease Detection
=========================================================
This module defines the Convolutional Neural Network architecture
for binary classification of brain MRI scans.
For educational/research purposes only.

Architecture Overview:
----------------------
The CNN consists of:
1. Convolutional layers for feature extraction
2. Pooling layers for spatial dimension reduction
3. Dense layers for classification
4. Dropout layers for regularization

Why CNN for MRI Analysis?
-------------------------
CNNs are ideally suited for medical image analysis because:
1. Hierarchical Feature Learning: CNNs automatically learn low-level
   features (edges, textures) and high-level features (patterns, shapes)
2. Translation Invariance: CNN can detect features regardless of their
   position in the image
3. Parameter Sharing: Reduces model complexity compared to fully connected networks
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    LEARNING_RATE, DROPOUT_RATE, DENSE_UNITS,
    MODEL_CHECKPOINT_PATH, EARLY_STOPPING_PATIENCE
)


def create_cnn_model():
    """
    Create and compile the CNN model for Parkinson's detection.
    
    Returns:
    --------
    tensorflow.keras.Model
        Compiled CNN model ready for training
        
    Architecture Explanation (for Viva/Report):
    -------------------------------------------
    
    Layer-by-Layer Breakdown:
    
    1. Input Layer (224, 224, 3):
       - Takes preprocessed MRI images as RGB
       
    2. Conv2D + BatchNorm + MaxPool Block 1:
       - 32 filters of size 3x3 detect basic patterns (edges, gradients)
       - BatchNormalization stabilizes training
       - MaxPooling reduces spatial dimensions by half
       
    3. Conv2D + BatchNorm + MaxPool Block 2:
       - 64 filters learn more complex patterns
       - Builds upon features from previous layer
       
    4. Conv2D + BatchNorm + MaxPool Block 3:
       - 128 filters capture high-level features
       - Detects structural patterns relevant to Parkinson's
       
    5. Conv2D + BatchNorm + MaxPool Block 4:
       - 256 filters for the most abstract features
       - Maximum feature abstraction before classification
       
    6. Flatten Layer:
       - Converts 3D feature maps to 1D vector
       
    7. Dense Layer (256 units):
       - Fully connected layer for feature combination
       - ReLU activation for non-linearity
       
    8. Dropout (0.5):
       - Randomly drops 50% of connections during training
       - Prevents overfitting by encouraging feature redundancy
       
    9. Output Layer (1 unit, Sigmoid):
       - Outputs probability of Parkinson's disease
       - Sigmoid maps output to [0, 1] for binary classification
    """
    
    model = Sequential([
        # Input specification
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        
        # ========================================
        # CONVOLUTIONAL BLOCK 1
        # ========================================
        # First convolutional layer: learns basic features
        # 32 filters, 3x3 kernel, ReLU activation
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # ========================================
        # CONVOLUTIONAL BLOCK 2
        # ========================================
        # Second convolutional layer: learns intermediate features
        # 64 filters detect more complex patterns
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # ========================================
        # CONVOLUTIONAL BLOCK 3
        # ========================================
        # Third convolutional layer: learns high-level features
        # 128 filters for structural patterns
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # ========================================
        # CONVOLUTIONAL BLOCK 4
        # ========================================
        # Fourth convolutional layer: learns abstract features
        # 256 filters for maximum abstraction
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # ========================================
        # CLASSIFICATION LAYERS
        # ========================================
        # Flatten the 3D output to 1D
        Flatten(),
        
        # Fully connected layer
        Dense(DENSE_UNITS, activation='relu'),
        
        # Dropout for regularization
        Dropout(DROPOUT_RATE),
        
        # Output layer: single neuron with sigmoid for binary classification
        # Sigmoid outputs probability in range [0, 1]
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',  # Standard loss for binary classification
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def get_callbacks():
    """
    Create training callbacks for model optimization.
    
    Returns:
    --------
    list
        List of Keras callbacks
        
    Callbacks Explanation (for Viva/Report):
    ----------------------------------------
    
    1. EarlyStopping:
       - Monitors validation loss
       - Stops training if no improvement for 'patience' epochs
       - Prevents overfitting and saves training time
       - Restores best weights automatically
       
    2. ModelCheckpoint:
       - Saves the best model based on validation accuracy
       - Ensures we keep the best performing model
       
    3. ReduceLROnPlateau:
       - Reduces learning rate when training plateaus
       - Helps escape local minima
       - Enables fine-tuning in later epochs
    """
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_auc',
            patience=EARLY_STOPPING_PATIENCE,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate when training stalls
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,           # Reduce LR by half
            patience=5,           # Wait 5 epochs before reducing
            min_lr=1e-7,          # Minimum learning rate
            verbose=1
        )
    ]
    
    return callbacks


def print_model_summary(model):
    """
    Print a detailed model summary with layer information.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        The compiled CNN model
    """
    print("\n" + "=" * 70)
    print("PARKINSON'S DISEASE DETECTION CNN - MODEL ARCHITECTURE")
    print("=" * 70)
    
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print("\n" + "-" * 70)
    print("PARAMETER SUMMARY")
    print("-" * 70)
    print(f"Total parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("-" * 70)
    
    print("\nModel Input Shape:  ", model.input_shape)
    print("Model Output Shape: ", model.output_shape)
    print("\n")


if __name__ == "__main__":
    """
    Test model creation when running this file directly.
    """
    print("Creating CNN model for Parkinson's Disease Detection...")
    model = create_cnn_model()
    print_model_summary(model)
    
    print("\nTraining callbacks configured:")
    callbacks = get_callbacks()
    for i, callback in enumerate(callbacks, 1):
        print(f"  {i}. {type(callback).__name__}")
