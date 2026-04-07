"""
Inference/Prediction Module for Parkinson's Disease Detection
===============================================================
This module provides functions for making predictions on new MRI images.
For educational/research purposes only.

Usage:
------
1. From command line:
   python predict.py path/to/mri_image.png

2. From Python code:
   from predict import predict_mri
   result = predict_mri('path/to/image.png')
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from config import (
    MODEL_CHECKPOINT_PATH, CLASS_NAMES, MODEL_DIR, THRESHOLD_CONFIG_PATH
)
from preprocessing import preprocess_for_prediction


class ParkinsonsPredictor:
    """
    A class for making predictions on MRI images.
    
    This class encapsulates the model loading and prediction logic,
    making it easy to use in the Flask web application.
    
    Attributes:
    -----------
    model : tensorflow.keras.Model
        The loaded CNN model
    model_loaded : bool
        Flag indicating if model is loaded successfully
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor and load the model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the trained model. Uses best model by default.
        """
        self.model = None
        self.model_loaded = False
        self.optimal_threshold = 0.5
        
        if model_path is None:
            # Try to find the best model path
            if os.path.exists(MODEL_CHECKPOINT_PATH):
                model_path = MODEL_CHECKPOINT_PATH
            else:
                final_model_path = os.path.join(MODEL_DIR, 'final_model.keras')
                if os.path.exists(final_model_path):
                    model_path = final_model_path
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            self._load_threshold()
        else:
            print("Warning: Model not found. Please train the model first.")

    def _load_threshold(self):
        """Load calibrated decision threshold saved by evaluate.py."""
        try:
            if os.path.exists(THRESHOLD_CONFIG_PATH):
                with open(THRESHOLD_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                threshold = float(payload.get('optimal_threshold', 0.5))
                if 0.0 < threshold < 1.0:
                    self.optimal_threshold = threshold
                    print(f"Using calibrated threshold: {self.optimal_threshold:.4f}")
                    return
            print("Threshold config not found; using default threshold: 0.5000")
        except Exception as e:
            print(f"Warning: Could not load threshold config: {e}")
            print("Falling back to default threshold: 0.5000")
    
    def _load_model(self, model_path):
        """
        Load the trained model from file.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        """
        try:
            print(f"Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def predict(self, image_path):
        """
        Make a prediction on a single MRI image.
        
        Parameters:
        -----------
        image_path : str
            Path to the MRI image file
            
        Returns:
        --------
        dict
            Dictionary containing:
            - prediction: 'Parkinson' or 'Healthy'
            - confidence: Confidence score (0-100%)
            - probability: Raw probability value
            - status: 'success' or 'error'
            - message: Additional information
            
        Prediction Logic Explanation (for Viva/Report):
        -----------------------------------------------
        The model outputs a single probability value between 0 and 1:
        - Values close to 0 indicate Healthy
        - Values close to 1 indicate Parkinson's
        - Calibrated threshold of 0.35 is used for classification
        
        Confidence is calculated based on distance from threshold:
        - Further from threshold = higher confidence
        """
        if not self.model_loaded:
            return {
                'status': 'error',
                'message': 'Model not loaded. Please train the model first.',
                'prediction': None,
                'confidence': 0,
                'probability': 0
            }
        
        # Validate image path
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'message': f'Image file not found: {image_path}',
                'prediction': None,
                'confidence': 0,
                'probability': 0
            }
        
        try:
            # Preprocess the image
            img = preprocess_for_prediction(image_path)
            
            # Make prediction
            prediction_prob = self.model.predict(img, verbose=0)[0][0]
            
            # Use calibrated threshold generated by evaluate.py (fallback: 0.5)
            optimal_threshold = self.optimal_threshold
            
            # Convert probability to class using calibrated threshold
            if prediction_prob > optimal_threshold:
                predicted_class = 'Parkinson'
                # Higher probability = higher confidence
                confidence = 50 + (prediction_prob - optimal_threshold) / (1 - optimal_threshold) * 50
            else:
                predicted_class = 'Healthy'
                # Lower probability = higher confidence for healthy
                confidence = 50 + (optimal_threshold - prediction_prob) / optimal_threshold * 50
            
            return {
                'status': 'success',
                'message': 'Prediction completed successfully',
                'prediction': predicted_class,
                'confidence': round(confidence, 2),
                'probability': round(float(prediction_prob), 4)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error during prediction: {str(e)}',
                'prediction': None,
                'confidence': 0,
                'probability': 0
            }


def predict_mri(image_path, model_path=None):
    """
    Convenience function to make a single prediction.
    
    Parameters:
    -----------
    image_path : str
        Path to the MRI image
    model_path : str, optional
        Path to the trained model
        
    Returns:
    --------
    dict
        Prediction results
    """
    predictor = ParkinsonsPredictor(model_path)
    return predictor.predict(image_path)


def print_prediction_result(result):
    """
    Print prediction result in a formatted way.
    
    Parameters:
    -----------
    result : dict
        Prediction result dictionary
    """
    print("\n" + "=" * 50)
    print("MRI ANALYSIS RESULT")
    print("=" * 50)
    
    if result['status'] == 'success':
        prediction = result['prediction']
        confidence = result['confidence']
        probability = result['probability']
        
        # Color coding (for terminal)
        if prediction == 'Parkinson':
            status_text = f"⚠️  {prediction.upper()}"
        else:
            status_text = f"✓  {prediction.upper()}"
        
        print(f"\nPrediction:  {status_text}")
        print(f"Confidence:  {confidence:.2f}%")
        print(f"Raw Probability: {probability:.4f}")
        
        print("\n" + "-" * 50)
        print("INTERPRETATION")
        print("-" * 50)
        
        if prediction == 'Parkinson':
            print("""
The model has detected patterns in the MRI scan that are
associated with Parkinson's Disease.

IMPORTANT DISCLAIMER:
This is an AI-based educational tool and should NOT be used
for medical diagnosis. Please consult a qualified neurologist
for professional medical evaluation.
""")
        else:
            print("""
The model has detected patterns in the MRI scan that are
consistent with a healthy brain.

IMPORTANT DISCLAIMER:
This is an AI-based educational tool and should NOT be used
for medical diagnosis. Please consult a healthcare professional
for complete medical evaluation.
""")
    else:
        print(f"\n❌ Error: {result['message']}")
    
    print("=" * 50)
    print("For educational/research purposes only")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    """
    Command-line interface for making predictions.
    
    Usage: python predict.py path/to/mri_image.png
    """
    print("=" * 60)
    print("PARKINSON'S DISEASE MRI DETECTION - INFERENCE MODULE")
    print("=" * 60)
    print("For educational/research purposes only")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <path_to_mri_image>")
        print("\nExample:")
        print("  python predict.py parkinsons_dataset/normal/Mag_Images_001.png")
        print("  python predict.py parkinsons_dataset/parkinson/T2W_TSE_001.png")
        
        # Demo with a sample image if available
        from config import DATASET_DIR
        sample_healthy = os.path.join(DATASET_DIR, 'normal', 'Mag_Images_001.png')
        sample_parkinson = os.path.join(DATASET_DIR, 'parkinson', 'T2W_TSE_001.png')
        
        if os.path.exists(sample_healthy):
            print("\n" + "-" * 60)
            print("DEMO: Running prediction on sample images...")
            print("-" * 60)
            
            print("\n[1] Testing with Healthy sample:")
            result = predict_mri(sample_healthy)
            print_prediction_result(result)
            
            print("\n[2] Testing with Parkinson sample:")
            result = predict_mri(sample_parkinson)
            print_prediction_result(result)
    else:
        # Make prediction on provided image
        image_path = sys.argv[1]
        result = predict_mri(image_path)
        print_prediction_result(result)
