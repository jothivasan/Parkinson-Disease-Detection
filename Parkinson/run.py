"""
Parkinson's Disease MRI Detection System
=========================================
Main Entry Point - Run this file to start the application

This script provides an interactive menu to:
1. Train the CNN model
2. Evaluate the trained model
3. Make predictions on new images
4. Start the web application

For educational/research purposes only.
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def print_banner():
    """Print application banner."""
    print("\n" + "=" * 70)
    print("   🧠 PARKINSON'S DISEASE MRI DETECTION SYSTEM")
    print("   CNN-Based Medical Image Analysis for Academic Research")
    print("=" * 70)
    print("   ⚠️  For educational/research purposes only")
    print("   ⚠️  Not intended for medical diagnosis")
    print("=" * 70 + "\n")


def print_menu():
    """Print interactive menu options."""
    print("\nPlease select an option:\n")
    print("  1. Train the CNN model")
    print("  2. Evaluate the trained model")
    print("  3. Make a prediction on an MRI image")
    print("  4. Start the web application")
    print("  5. View project structure")
    print("  6. Exit")
    print()


def train_model():
    """Train the CNN model."""
    print("\n" + "-" * 50)
    print("Starting model training...")
    print("-" * 50 + "\n")
    
    from train import train_model as run_training
    model, history = run_training()
    return model


def evaluate_model():
    """Evaluate the trained model."""
    print("\n" + "-" * 50)
    print("Starting model evaluation...")
    print("-" * 50 + "\n")
    
    from evaluate import evaluate_model as run_evaluation
    results = run_evaluation()
    return results


def make_prediction():
    """Make prediction on a single image."""
    from predict import predict_mri, print_prediction_result
    from config import MODEL_CHECKPOINT_PATH
    
    # Check if model exists
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print("\n❌ Error: No trained model found!")
        print("Please train the model first (Option 1)")
        return
    
    print("\n" + "-" * 50)
    print("Enter the path to the MRI image:")
    print("-" * 50)
    print("\nExample paths:")
    print("  - parkinsons_dataset/normal/Mag_Images_001.png")
    print("  - parkinsons_dataset/parkinson/T2W_TSE_001.png")
    print()
    
    image_path = input("Image path: ").strip()
    
    if not image_path:
        print("No path provided. Returning to menu.")
        return
    
    # Make prediction
    result = predict_mri(image_path)
    print_prediction_result(result)


def start_webapp():
    """Start the Flask web application."""
    from config import MODEL_CHECKPOINT_PATH
    
    # Check if model exists
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print("\n⚠️  Warning: No trained model found!")
        print("The web application will start, but predictions won't work")
        print("until you train the model (Option 1)")
        print()
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    print("\n" + "-" * 50)
    print("Starting web application...")
    print("-" * 50)
    print("\nOpen your browser and navigate to:")
    print("  http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50 + "\n")
    
    # Import and run Flask app
    os.chdir(os.path.join(PROJECT_ROOT, 'webapp'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'webapp'))
    
    from app import app
    from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)


def view_structure():
    """Display project structure."""
    print("\n" + "=" * 70)
    print("PROJECT STRUCTURE")
    print("=" * 70)
    
    structure = """
Parkinson/
│
├── 📁 parkinsons_dataset/          # MRI Dataset
│   ├── 📁 normal/                  # Healthy brain MRI scans (610 images)
│   └── 📁 parkinson/               # Parkinson's brain MRI scans (221 images)
│
├── 📁 models/                      # Saved trained models
│   ├── best_model.keras            # Best model (saved during training)
│   └── final_model.keras           # Final model after training
│
├── 📁 logs/                        # Training logs and visualizations
│   ├── training_history.png        # Accuracy/Loss curves
│   ├── confusion_matrix.png        # Evaluation confusion matrix
│   └── roc_curve.png               # ROC curve with AUC score
│
├── 📁 webapp/                      # Flask Web Application
│   ├── 📁 static/
│   │   ├── 📁 css/
│   │   │   └── style.css           # Application stylesheet
│   │   └── 📁 uploads/             # Uploaded images for analysis
│   ├── 📁 templates/
│   │   ├── index.html              # Home page (upload)
│   │   ├── result.html             # Prediction results page
│   │   └── about.html              # About page
│   └── app.py                      # Flask application
│
├── config.py                       # Configuration settings
├── preprocessing.py                # MRI image preprocessing
├── model.py                        # CNN model architecture
├── train.py                        # Training pipeline
├── evaluate.py                     # Model evaluation metrics
├── predict.py                      # Inference/prediction module
├── run.py                          # THIS FILE - Main entry point
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
"""
    print(structure)
    print("=" * 70)


def main():
    """Main application loop."""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                train_model()
            elif choice == '2':
                evaluate_model()
            elif choice == '3':
                make_prediction()
            elif choice == '4':
                start_webapp()
            elif choice == '5':
                view_structure()
            elif choice == '6':
                print("\n" + "=" * 50)
                print("Thank you for using the Parkinson's Detection System!")
                print("=" * 50 + "\n")
                break
            else:
                print("\n❌ Invalid choice. Please enter a number between 1 and 6.")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            continue
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
