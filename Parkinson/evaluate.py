"""
Model Evaluation Module for Parkinson's Disease Detection
==========================================================
This module provides comprehensive evaluation metrics including:
1. Accuracy, Precision, Recall, F1-Score
2. Confusion Matrix visualization
3. Classification Report
4. ROC Curve and AUC Score
For educational/research purposes only.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import (
    DATASET_DIR, MODEL_CHECKPOINT_PATH, LOGS_DIR,
    CLASS_NAMES, VALIDATION_SPLIT, THRESHOLD_CONFIG_PATH
)
from preprocessing import load_dataset


def load_trained_model(model_path=None):
    """
    Load a trained model from file.
    
    Parameters:
    -----------
    model_path : str, optional
        Path to the saved model. Uses best model by default.
        
    Returns:
    --------
    tensorflow.keras.Model
        Loaded model
    """
    if model_path is None:
        model_path = MODEL_CHECKPOINT_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first by running train.py"
        )
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    return model


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot and save confusion matrix visualization.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    save_path : str, optional
        Path to save the plot
        
    Confusion Matrix Explanation (for Viva/Report):
    -----------------------------------------------
    A confusion matrix shows the performance of a classification model:
    
                        Predicted
                    Healthy  Parkinson
    Actual Healthy    TN        FP
    Actual Parkinson  FN        TP
    
    - True Negatives (TN): Correctly predicted as Healthy
    - True Positives (TP): Correctly predicted as Parkinson
    - False Negatives (FN): Parkinson cases missed (Type II error)
    - False Positives (FP): Healthy cases wrongly flagged (Type I error)
    
    For medical diagnosis, minimizing FN is crucial (reducing missed cases).
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={'size': 16}
    )
    
    plt.title('Confusion Matrix\nParkinson\'s Disease Detection', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add additional information
    tn, fp, fn, tp = cm.ravel()
    info_text = f'TN={tn}, FP={fp}, FN={fn}, TP={tp}'
    plt.figtext(0.5, -0.02, info_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC curve and calculate AUC.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_prob : numpy.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    float
        Area Under the Curve (AUC) score
        
    ROC Curve Explanation (for Viva/Report):
    ----------------------------------------
    The ROC (Receiver Operating Characteristic) curve shows:
    - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
    - Y-axis: True Positive Rate (TPR) = TP / (TP + FN)
    
    AUC (Area Under Curve) interpretation:
    - AUC = 1.0: Perfect classifier
    - AUC = 0.5: Random classifier (diagonal line)
    - AUC > 0.8: Good classifier
    - AUC > 0.9: Excellent classifier
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='#2E86AB', linewidth=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='#A23B72', linestyle='--', 
             linewidth=2, label='Random Classifier')
    
    # Fill area under curve
    plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Parkinson\'s Disease Detection', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()
    
    return roc_auc


def print_classification_report(y_true, y_pred, y_prob):
    """
    Print comprehensive classification metrics.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    y_prob : numpy.ndarray
        Predicted probabilities
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{'Metric':<20} {'Score':<15} {'Explanation'}")
    print("-" * 60)
    print(f"{'Accuracy:':<20} {accuracy:.4f} ({accuracy*100:.2f}%)   "
          f"Overall correct predictions")
    print(f"{'Precision:':<20} {precision:.4f} ({precision*100:.2f}%)   "
          f"Correctly identified Parkinson's")
    print(f"{'Recall:':<20} {recall:.4f} ({recall*100:.2f}%)   "
          f"Parkinson's cases detected")
    print(f"{'F1-Score:':<20} {f1:.4f} ({f1*100:.2f}%)   "
          f"Harmonic mean of P & R")
    
    # Detailed classification report
    print("\n" + "-" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_true, y_pred, 
                                target_names=CLASS_NAMES,
                                digits=4))
    
    # Interpretation
    print("-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    print(f"""
Medical Significance of Metrics:

• Recall (Sensitivity): {recall:.4f}
  This represents the model's ability to correctly identify patients
  with Parkinson's disease. A high recall is crucial to avoid missing
  patients who need treatment.

• Precision: {precision:.4f}
  This indicates how many of the predicted Parkinson's cases are actually
  positive. High precision reduces false alarms.

• F1-Score: {f1:.4f}
  The F1-score balances precision and recall. For medical screening,
  we typically prioritize recall to minimize missed diagnoses.

Note: This is for educational/research purposes only. 
Clinical diagnosis requires professional medical evaluation.
""")


def evaluate_model(model=None, model_path=None):
    """
    Complete model evaluation pipeline.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model, optional
        Trained model. If None, loads from model_path.
    model_path : str, optional
        Path to saved model
        
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    print("=" * 70)
    print("PARKINSON'S DISEASE DETECTION - MODEL EVALUATION")
    print("=" * 70)
    print("For educational/research purposes only")
    print("=" * 70)
    
    # ========================================
    # Load Model
    # ========================================
    if model is None:
        model = load_trained_model(model_path)
    
    # ========================================
    # Load and Prepare Data
    # ========================================
    print("\n" + "-" * 50)
    print("Loading validation data...")
    print("-" * 50)
    
    images, labels = load_dataset(DATASET_DIR)
    
    # Use same split as training
    _, X_val, _, y_val = train_test_split(
        images, labels,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=labels
    )
    
    print(f"Validation samples: {len(X_val)}")
    print(f"  Healthy: {np.sum(y_val == 0)}")
    print(f"  Parkinson: {np.sum(y_val == 1)}")
    
    # ========================================
    # Generate Predictions
    # ========================================
    print("\n" + "-" * 50)
    print("Generating predictions...")
    print("-" * 50)
    
    # Get probability predictions
    y_prob = model.predict(X_val, verbose=0).flatten()
    
    # Find optimal threshold using Youden's J statistic
    from sklearn.metrics import roc_curve as roc
    fpr, tpr, thresholds = roc(y_val, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal Classification Threshold: {optimal_threshold:.4f}")
    print(f"(Default threshold is 0.5, optimal found using Youden's J statistic)")

    # Persist threshold so predict.py and webapp use the same decision boundary.
    threshold_payload = {
        'optimal_threshold': float(optimal_threshold),
        'method': "Youden's J statistic",
        'source': 'evaluate.py'
    }
    with open(THRESHOLD_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(threshold_payload, f, indent=2)
    print(f"Saved threshold configuration to: {THRESHOLD_CONFIG_PATH}")
    
    # Convert probabilities to class labels using optimal threshold
    y_pred = (y_prob > optimal_threshold).astype(int)
    
    # ========================================
    # Calculate and Display Metrics
    # ========================================
    print_classification_report(y_val, y_pred, y_prob)
    
    # ========================================
    # Plot Confusion Matrix
    # ========================================
    print("\n" + "-" * 50)
    print("Generating Confusion Matrix...")
    print("-" * 50)
    
    cm_path = os.path.join(LOGS_DIR, 'confusion_matrix.png')
    cm = plot_confusion_matrix(y_val, y_pred, save_path=cm_path)
    
    # ========================================
    # Plot ROC Curve
    # ========================================
    print("\n" + "-" * 50)
    print("Generating ROC Curve...")
    print("-" * 50)
    
    roc_path = os.path.join(LOGS_DIR, 'roc_curve.png')
    roc_auc = plot_roc_curve(y_val, y_prob, save_path=roc_path)
    
    # ========================================
    # Compile Results
    # ========================================
    results = {
        'optimal_threshold': float(optimal_threshold),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'auc': roc_auc,
        'confusion_matrix': cm
    }
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nVisualization files saved to: {LOGS_DIR}")
    
    return results


if __name__ == "__main__":
    """
    Run evaluation when this file is executed directly.
    """
    results = evaluate_model()
