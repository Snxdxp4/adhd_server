from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import numpy as np
from flask_pymongo import PyMongo
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
import joblib
import bcrypt
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app) 
if mongo.db is None:
    print("❌ MongoDB is NOT connected. Check your MONGO_URI and Flask setup.")
else:
    print("✅ MongoDB connected successfully!")

app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
jwt = JWTManager(app)

model = joblib.load("./model/AdaBoost.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Hello World!"})
    
@app.route("/register", methods=["POST"])
def register():
    print(request.get_json())
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    if "users" not in mongo.db.list_collection_names():
        mongo.db.create_collection("users")
        
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    existing_user = mongo.db.users.find_one({"email": email})
    if existing_user:
        return jsonify({"error": "User already exists"}), 400

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    mongo.db.users.insert_one({"email":email,"username": username, "password": hashed_password})

    return jsonify({"message": "User registered successfully"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = mongo.db.users.find_one({"email": email})
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    # Generate JWT token
    access_token = create_access_token(identity=email)
    return jsonify({"access_token": access_token,"userName":user["username"]}), 200


@app.route("/predict", methods=["POST"])
@jwt_required()
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



