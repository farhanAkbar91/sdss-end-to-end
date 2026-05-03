import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Constants
FEATURE_COLS = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'cam_col', 'redshift', 'plate', 'MJD']
CLASS_NAMES = ['GALAXY', 'QSO', 'STAR'] # 0, 1, 2 mapping

# Load models and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

try:
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_sdss.pkl'))
    models = {
        'Decision Tree': joblib.load(os.path.join(MODELS_DIR, 'model_dt.pkl')),
        'Random Forest': joblib.load(os.path.join(MODELS_DIR, 'model_rf.pkl')),
        'Logistic Regression': joblib.load(os.path.join(MODELS_DIR, 'model_logreg.pkl'))
    }
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
except Exception as e:
    print(f"Error loading models or metrics: {e}")
    scaler = None
    models = {}
    metrics = {}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features'}), 400
    
    features = data['features']
    if not isinstance(features, dict):
        return jsonify({'error': 'Features should be a dictionary'}), 400
    
    model_name = data.get('model', 'Random Forest')
    if model_name not in models:
        return jsonify({'error': 'Invalid model name'}), 400
    
    try:
        # Extract features in the correct order
        X_input = np.array([[float(features[col]) for col in FEATURE_COLS]])
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400
    except ValueError:
        return jsonify({'error': 'All features must be numeric'}), 400

    X_scaled = scaler.transform(X_input)
    model = models[model_name]
    
    y_pred_idx = model.predict(X_scaled)[0]
    prediction = CLASS_NAMES[y_pred_idx]
    
    # Optional: Probability if model supports it
    probabilities = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)[0]
        probabilities = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return jsonify({
        'model': model_name,
        'prediction': prediction,
        'probabilities': probabilities
    })

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features'}), 400
    
    features = data['features']
    if not isinstance(features, dict):
        return jsonify({'error': 'Features should be a dictionary'}), 400

    try:
        X_input = np.array([[float(features[col]) for col in FEATURE_COLS]])
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400
    except ValueError:
        return jsonify({'error': 'All features must be numeric'}), 400

    X_scaled = scaler.transform(X_input)
    
    results = {}
    for name, model in models.items():
        y_pred_idx = model.predict(X_scaled)[0]
        prediction = CLASS_NAMES[y_pred_idx]
        
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[0]
            probabilities = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)}
            
        model_metrics = metrics.get(name, {})
        results[name] = {
            'prediction': prediction,
            'probabilities': probabilities,
            'metrics': model_metrics
        }

    return jsonify({
        'results': results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
