from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests # ✅ ADDED: For making API calls
import os       # ✅ ADDED: To read environment variables

app = Flask(__name__)
CORS(app)

# --- The local Transformers model has been removed ---

# ✅ KEPT: The logic for reading your local CSV is unchanged
df = pd.read_csv("Crop.csv")
df.columns = df.columns.str.strip().str.lower()

# ✅ KEPT: Your function to search the local dataset is unchanged
def search_dataset(user_msg):
    user_msg = user_msg.lower()

    if "ph" in user_msg:
        min_ph, max_ph = df["ph"].min(), df["ph"].max()
        return f"The dataset shows soil pH ranges between {min_ph:.1f} and {max_ph:.1f}."
    elif "rainfall" in user_msg:
        min_rf, max_rf = df["rainfall"].min(), df["rainfall"].max()
        return f"Rainfall in the dataset varies from {min_rf:.1f} mm to {max_rf:.1f} mm."
    elif "crops" in user_msg or "suggest" in user_msg:
        crops = df["label"].unique().tolist()
        return f"Some crops in the dataset are: {', '.join(crops[:10])}."
    return None

# -------------------------------------------------------------------
# ✅ UPDATED: The /chat route now calls the Hugging Face API
# -------------------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    api_token = os.environ.get("HF_API_TOKEN") # Reads your secret token

    # First, try your local dataset for simple questions
    dataset_reply = search_dataset(user_msg)
    if dataset_reply:
        return jsonify({"reply": dataset_reply})

    # If no simple answer, call the Hugging Face Inference API
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": f"Answer concisely about crops and farming: {user_msg}"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status() # Check for API errors
        result = response.json()
        reply = result[0].get('generated_text', "Sorry, I couldn't get a response.")
        return jsonify({"reply": reply})

    except requests.exceptions.RequestException as e:
        print(f"Error calling Hugging Face API: {e}")
        return jsonify({"reply": "Sorry, the AI assistant is currently unavailable."}), 503

if __name__ == "__main__":
    app.run(debug=True)
