from flask import Flask, request, render_template
import joblib
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the demand prediction model
demand_model_filename = 'product_demand_model.pkl'
with open(demand_model_filename, 'rb') as file:
    demand_model = pickle.load(file)

# Load the price prediction model
price_model_filename = 'cotton_clothes_price_model.pkl'
price_model = joblib.load(price_model_filename)

# If you saved the scaler during training, load it; otherwise, you need to fit it
# Uncomment the line below if you have saved the scaler
# scaler = joblib.load('scaler.pkl')
scaler = StandardScaler()  # Initialize the scaler for price prediction

@app.route('/')
def home():
    return render_template('predict.html')  # HTML file for frontend

@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    # Retrieve input features for demand prediction
    footfall = request.form['footfall']
    buying_habits = request.form['buying_habits']
    latest_trends = request.form['latest_trends']

    # Prepare input for demand prediction
    input_data = np.array([[float(footfall), float(buying_habits), float(latest_trends)]])
    
    # Make demand prediction
    predicted_demand = demand_model.predict(input_data)[0]
    
    # Suggest inventory level
    safety_stock = 10  # You can adjust this as needed
    suggested_inventory = int(predicted_demand + safety_stock)
    
    return render_template('predict.html', predicted_demand=predicted_demand, suggested_inventory=suggested_inventory)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    # Retrieve input features for price prediction
    data = request.form  # Assuming form data input
    features = [
        data['Product_Type'],
        data['Color'],
        data['Size'],
        data['Material'],
        data['Brand']
    ]

    # Convert input to numpy array and reshape for prediction
    features = np.array(features).reshape(1, -1)

    # Note: Ensure the scaler has been fit on data prior to this step
    # You may need to fit the scaler with training data if it's not loaded
    scaled_features = scaler.transform(features)  # Scale the features

    # Get price prediction from the model
    predicted_price = price_model.predict(scaled_features)[0]

    return render_template('predict.html', predicted_price=predicted_price)

if __name__ == "_main_":
    app.run(debug=True)