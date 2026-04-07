"""
Flask Web Application for Parkinson's Disease MRI Detection
============================================================
This module provides a web interface for uploading MRI images
and receiving predictions with confidence scores.
For educational/research purposes only.

Features:
---------
1. Clean, modern user interface
2. MRI image upload functionality
3. Real-time prediction display
4. Confidence score visualization
5. Medical disclaimer
"""

import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FLASK_DEBUG, FLASK_HOST, FLASK_PORT,
    ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
)
from predict import ParkinsonsPredictor

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'parkinsons_detection_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize predictor (global instance)
predictor = None


def get_predictor():
    """Get or initialize the predictor instance."""
    global predictor
    if predictor is None or not predictor.model_loaded:
        predictor = ParkinsonsPredictor()
    return predictor


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle MRI image upload and prediction.
    
    Returns:
    --------
    Rendered template with prediction results
    """
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded. Please select an MRI image.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No file selected. Please choose an MRI image.', 'error')
        return redirect(url_for('index'))
    
    # Validate file type
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PNG, JPG, JPEG, GIF, or BMP image.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get predictor and make prediction
        pred = get_predictor()
        
        if not pred.model_loaded:
            flash('Model not loaded. Please train the model first by running train.py', 'error')
            return redirect(url_for('index'))
        
        result = pred.predict(filepath)
        
        if result['status'] == 'success':
            return render_template(
                'result.html',
                prediction=result['prediction'],
                confidence=result['confidence'],
                probability=result['probability'],
                image_filename=unique_filename
            )
        else:
            flash(f"Prediction error: {result['message']}", 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions (for programmatic access).
    
    Returns:
    --------
    JSON response with prediction results
    """
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        pred = get_predictor()
        result = pred.predict(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


if __name__ == '__main__':
    print("=" * 60)
    print("PARKINSON'S DISEASE MRI DETECTION - WEB APPLICATION")
    print("=" * 60)
    print("For educational/research purposes only")
    print("=" * 60)
    print(f"\nStarting server at http://{FLASK_HOST}:{FLASK_PORT}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
