from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Example features
        age = float(request.form["age"])
        salary = float(request.form["salary"])

        features = np.array([[age, salary]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        result = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text="Error in input")

if __name__ == "__main__":
    app.run(debug=True)