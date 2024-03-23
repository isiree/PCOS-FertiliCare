import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("MlModel.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Extract numerical feature values from form
    float_features = [float(request.form['BMI']),
                      float(request.form['FSH/LH']),
                      float(request.form['PRL(ng/mL)'])]
    
    # Extract cycleValue from hidden input field
    cycle_value = float(request.form['Cycle(R/I)'])
    float_features.append(cycle_value)  # Append cycleValue to the feature list
    
    # Convert features to numpy array
    features = np.array([float_features])
    
    # Predict
    prediction = model.predict(features)

    # Map prediction to risk level
    if prediction == 6:
        risk_level = "High"
    elif prediction == 4:
        risk_level = "Medium"
    elif prediction == 2:
        risk_level = "Low"
    else:
        risk_level = "Unknown"

    return render_template("index.html", prediction_text="The infertility risk level is {}".format(risk_level))

if __name__ == "__main__":
    flask_app.run(debug=True)