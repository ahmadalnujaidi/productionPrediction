from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scaler
delay_model = joblib.load('delay_model.pkl')
defect_model = joblib.load('defect_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Define feature order to ensure consistency
FEATURE_ORDER = [
    'weight',
    'temperature',
    'speed',
    'processTime',
    'components',
    'efficiency',
    'quantity'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create feature array in correct order
        features = np.array([[float(data[feature]) for feature in FEATURE_ORDER]])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make predictions
        delay_prediction = delay_model.predict(scaled_features)[0]
        defect_prediction = defect_model.predict(scaled_features)[0]
        
        return jsonify({
            'delay_prediction': round(delay_prediction, 2),
            'defect_prediction': round(defect_prediction, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
