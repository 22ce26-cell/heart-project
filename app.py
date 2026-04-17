from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        data = [float(x) for x in request.form.values()]
        data = np.array([data])

        # Scale input
        data = scaler.transform(data)

        # Prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1] * 100

        # Result logic
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

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error in input data",
            probability=0,
            result_color="black"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
