# Install required libraries
# pip install uvicorn joblib openai nest_asyncio pandas numpy scikit-learn flask

# Import libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from pydantic import BaseModel
import uvicorn
import openai
import nest_asyncio
import os
from sklearn.metrics import mean_absolute_error
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np


# Apply nest_asyncio
nest_asyncio.apply()

# ✅ Set API keys (Updated)

openai.api_key = os.getenv("OPENAI_API_KEY")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Set random seed
np.random.seed(42)

# Generate synthetic dataset
num_samples = 10000

data = {
    "Factory_ID": np.random.choice(["Factory_A", "Factory_B", "Factory_C"], num_samples),
    "Weight_kg": np.round(np.random.uniform(1.0, 10.0, num_samples), 2),
    "Temperature_C": np.random.randint(150, 350, num_samples).astype(float),
    "Production_Speed_mps": np.round(np.random.uniform(3.0, 15.0, num_samples), 2),
    "Process_Time_mins": np.round(np.random.uniform(10.0, 60.0, num_samples), 2),
    "Number_of_Components": np.random.randint(1, 20, num_samples),
    "Material_Usage_Efficiency": np.round(np.random.uniform(50.0, 100.0, num_samples), 2),
    "Order_Quantity": np.random.randint(1, 500, num_samples),
    "Delivery_Delay_days": np.random.randint(0, 10, num_samples),
    "Defect_Probability": np.round(np.random.uniform(0.0, 1.0, num_samples), 4)
}

df = pd.DataFrame(data)

# Adjust factory-specific data differences
df.loc[df["Factory_ID"] == "Factory_B", "Temperature_C"] = (df.loc[df["Factory_ID"] == "Factory_B", "Temperature_C"] - 32) * 5/9
df.loc[df["Factory_ID"] == "Factory_C", "Weight_kg"] = df.loc[df["Factory_ID"] == "Factory_C", "Weight_kg"] / 1000

# Ensure all weights are in kg
df["Weight_kg"] = df["Weight_kg"].astype(float)

# Fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Normalize numerical features
scaler = MinMaxScaler()
features = ["Weight_kg", "Temperature_C", "Production_Speed_mps",
            "Process_Time_mins", "Number_of_Components",
            "Material_Usage_Efficiency", "Order_Quantity"]

df[features] = scaler.fit_transform(df[features])

# Save the scaler
joblib.dump(scaler, "feature_scaler.pkl")

# Save the cleaned dataset
df.to_csv("cleaned_synthetic_data.csv", index=False)

# Display summary statistics
print(df.describe())

# Display a sample of the dataset
df.head()


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the cleaned dataset
df = pd.read_csv("cleaned_synthetic_data.csv")

# Define features and targets
X = df[["Weight_kg", "Temperature_C", "Production_Speed_mps",
        "Process_Time_mins", "Number_of_Components",
        "Material_Usage_Efficiency", "Order_Quantity"]]

y_delay = df["Delivery_Delay_days"]
y_defect = df["Defect_Probability"]

# Split dataset independently for each target variable
X_train_delay, X_test_delay, y_delay_train, y_delay_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
X_train_defect, X_test_defect, y_defect_train, y_defect_test = train_test_split(X, y_defect, test_size=0.2, random_state=42)

# Train delay prediction model
delay_model = RandomForestRegressor(n_estimators=100, random_state=42)
delay_model.fit(X_train_delay, y_delay_train)

# Train defect probability model
defect_model = RandomForestRegressor(n_estimators=100, random_state=42)
defect_model.fit(X_train_defect, y_defect_train)

# Evaluate models
delay_preds = delay_model.predict(X_test_delay)
defect_preds = defect_model.predict(X_test_defect)

delay_mae = mean_absolute_error(y_delay_test, delay_preds)
defect_mae = mean_absolute_error(y_defect_test, defect_preds)

print(f"Delay Model MAE: {delay_mae:.2f}")
print(f"Defect Model MAE: {defect_mae:.2f}")

# Save models
joblib.dump(delay_model, "delay_model.pkl")
joblib.dump(defect_model, "defect_model.pkl")

print(" Models saved successfully!")


import os

print("Checking if models exist:")
print("Delay Model Exists:", os.path.exists("delay_model.pkl"))
print("Defect Model Exists:", os.path.exists("defect_model.pkl"))


import joblib

# Load models
try:
    delay_model = joblib.load("delay_model.pkl")
    defect_model = joblib.load("defect_model.pkl")
    print(" Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")

# Verify model loading
print("Checking if models exist:")
print("Delay Model Exists:", os.path.exists("delay_model.pkl"))
print("Defect Model Exists:", os.path.exists("defect_model.pkl"))


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
