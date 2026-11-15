import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template 

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json()

    row = pd.DataFrame([{
        "change_today": d["total_today"] - d["total_yesterday"],
        "change_7days": d["total_today"] - d["total_7days_ago"],
        "crci_share": d["crci_patients_today"] / d["total_today"],
        "is_weekend": d["is_weekend"],
    }])

    change = float(model.predict(row)[0])
    tomorrow = d["total_today"] + change

    return jsonify({
        "predicted_patients_tomorrow": round(tomorrow),
        "predicted_free_beds_tomorrow": round(d["total_beds"] - tomorrow),
    })


app.run(port=5000)