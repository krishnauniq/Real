from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import os
import joblib # ✅ ADDED BACK: For loading the scikit-learn model

app = Flask(__name__)
CORS(app)

# --- Configuration ---
CROP_CSV_PATH = "Crop.csv"
CROP_MODEL_PATH = "crop_model.pkl"
EXPECTED_COLUMNS = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

# --- Load Models and Data at Startup ---
try:
    df = pd.read_csv(CROP_CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()
    crop_model = joblib.load(CROP_MODEL_PATH)
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure '{CROP_CSV_PATH}' and '{CROP_MODEL_PATH}' exist.")
    crop_model = None # Set to None if model can't be loaded

# --- Helper Functions ---
def search_dataset(user_msg: str) -> str | None:
    """Search the dataset for quick answers based on keywords."""
    user_msg = user_msg.lower()
    # ... (your existing search logic is good) ...
    if "ph" in user_msg:
        min_ph, max_ph = df["ph"].min(), df["ph"].max()
        return f"The dataset shows soil pH ranges between {min_ph:.1f} and {max_ph:.1f}."
    if "rainfall" in user_msg:
        min_rf, max_rf = df["rainfall"].min(), df["rainfall"].max()
        return f"Rainfall in the dataset varies from {min_rf:.1f} mm to {max_rf:.1f} mm."
    if "crops" in user_msg or "suggest" in user_msg:
        crops = df["label"].unique().tolist()
        return f"Some crops in the dataset are: {', '.join(crops[:10])}."
    return None

def query_huggingface_api(message: str) -> str:
    """Queries the Hugging Face API for a chat response."""
    api_token = os.environ.get("HF_API_TOKEN")
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": f"Answer concisely about crops and farming: {message}"}
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result[0].get("generated_text", "Sorry, I couldn't get a response.")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Hugging Face API: {e}")
        return "Sorry, the AI assistant is currently unavailable."

# --- API Routes ---
@app.route('/')
def home():
    """✅ ADDED: Home route to show that the API is running."""
    return "JeeVan API is live and running!"

@app.route('/predict', methods=['POST'])
def predict():
    """✅ ADDED BACK: Predicts the best crop based on input data."""
    if crop_model is None:
        return jsonify({"error": "Prediction model is not loaded."}), 500
    try:
        data = request.json
        data = {key.lower(): value for key, value in data.items()}
        
        input_data = {col: [data.get(col)] for col in EXPECTED_COLUMNS}
        input_df = pd.DataFrame(input_data).astype(float)
        
        prediction = crop_model.predict(input_df)[0]
        return jsonify({"recommended_crop": prediction})
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Failed to process prediction request."}), 400

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat route: answers from dataset or Hugging Face API."""
    user_msg = request.json.get("message", "").strip()
    
    # First, try local dataset
    dataset_reply = search_dataset(user_msg)
    if dataset_reply:
        return jsonify({"reply": dataset_reply})

    # Otherwise, call the helper function for Hugging Face API
    api_reply = query_huggingface_api(user_msg)
    
    return jsonify({"reply": api_reply})

if __name__ == "__main__":
    app.run(debug=True)
