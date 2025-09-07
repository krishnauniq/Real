from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

df = pd.read_csv("Crop.csv")  

df.columns = df.columns.str.strip().str.lower()  

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")

    # First, try dataset
    dataset_reply = search_dataset(user_msg)
    if dataset_reply:
        return jsonify({"reply": dataset_reply})

    # Else, fallback to LLM
    input_text = f"Answer concisely about crops and farming: {user_msg}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
