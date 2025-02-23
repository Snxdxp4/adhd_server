from flask import Flask, request, jsonify   # type: ignore
from flask_cors import CORS  # type: ignore
import joblib  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore


app = Flask(__name__)
CORS(app)

model = joblib.load("/model/AdaBoost.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    feature_names = [
        "Trouble wrapping up final details",
        "Difficulty getting things in order",
        "Problems remembering appointments",
        "Avoiding or delaying difficult tasks",
        "Fidgeting when sitting for long",
        "Feeling overly active",
        "Making careless mistakes",
        "Difficulty keeping attention",
        "Difficulty concentrating on speech",
        "Misplacing or losing things",
        "Distracted by activity or noise",
        "Leaving seat in meetings",
        "Feeling restless or fidgety",
        "Talking too much in social situations",
        "Finishing others' sentences",
        "Interrupting others",
        "Difficulty unwinding and relaxing",
        "Difficulty waiting turn",
    ]

    missing_features = [f for f in feature_names if f not in data]
    if missing_features:
        return (
            jsonify(
                {"error": "Missing features", "missing_features": missing_features}
            ),
            400,
        )

    input_features = np.array([data[feature] for feature in feature_names]).reshape(
        1, -1
    )

    prediction = model.predict(input_features)[0]

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(input_features).max()
    else:
        confidence = None

    return (
        jsonify(
            {
                "prediction": int(prediction),
                "Confidence": float(confidence) if confidence is not None else "N/A",
            }
        ),
        200,
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
