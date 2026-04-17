from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    data = np.array([data])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1] * 100

    if prediction == 1:
        result = "High Risk"
        color = "red"
    else:
        result = "Low Risk"
        color = "green"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=round(probability, 2),
        result_color=color
    )

 
app.run(debug=True)
