
# Import necessary libraries
import joblib
import pandas as pd

# Import Flask tools
from flask import Flask, request, jsonify


# ---------------------------------------------------------
# Create the Flask application
# ---------------------------------------------------------

superkart_api = Flask("SuperKart")


# ---------------------------------------------------------
# Load the trained model
# app.py and rf_estimator.joblib are in the same folder
# ---------------------------------------------------------

model = joblib.load("xgb_estimator.joblib")


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------

@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


# ---------------------------------------------------------
# Endpoint for predicting the sales of ONE product
# ---------------------------------------------------------

@superkart_api.post("/v1/predict")
def predict_sales():

    # Get JSON data sent by the client
    data = request.get_json()

    # Extract the features expected by the trained model
    sample = {
        "Product_Weight": data["Product_Weight"],
        "Product_Sugar_Content": data["Product_Sugar_Content"],
        "Product_Allocated_Area": data["Product_Allocated_Area"],
        "Product_MRP": data["Product_MRP"],
        "Store_Size": data["Store_Size"],
        "Store_Location_City_Type": data["Store_Location_City_Type"],
        "Store_Type": data["Store_Type"],
        "Product_Id_char": data["Product_Id_char"],
        "Store_Age_Years": data["Store_Age_Years"],
        "Product_Type_Category": data["Product_Type_Category"]
    }

    # Convert the single observation into a DataFrame
    input_data = pd.DataFrame([sample])

    # Predict sales
    prediction = model.predict(input_data).tolist()[0]

    # Return prediction as JSON
    return jsonify({
        "Sales": prediction
    })


# ---------------------------------------------------------
# Endpoint for predicting multiple products from a CSV
# ---------------------------------------------------------

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():

    # Get the uploaded CSV file
    file = request.files["file"]

    # Read CSV into a DataFrame
    input_data = pd.read_csv(file)

    # Predict sales for every row
    predictions = model.predict(input_data).tolist()

    # Return predictions indexed by row number
    output_dict = {
        str(i): round(pred, 2)
        for i, pred in enumerate(predictions)
    }

    return output_dict


# ---------------------------------------------------------
# Start Flask
# ---------------------------------------------------------

if __name__ == "__main__":
    superkart_api.run(debug=True)
