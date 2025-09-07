from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# ML / NLP - Simplified imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# ================== CONFIG ==================
# ✅ CHANGED: Point to your new, simplified CSV file
CSV_PATH = "Crop.csv" 
MODEL_PATH = "crop_model.pkl"

# ✅ CHANGED: This list now perfectly matches the columns in your CSV and the data from your React form
EXPECTED_COLUMNS = [
    'n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall'
]

# ================== LOAD DATASET ==================
# This part is fine
try:
    df = pd.read_csv(CSV_PATH)
    # Ensure column names are lowercase and stripped of whitespace
    df.columns = [col.strip().lower() for col in df.columns]
except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found. Please make sure it's in the correct directory.")
    exit()


# ================== CHATBOT MODEL ==================
# This part is fine
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

# ================== TRAIN OR LOAD CROP MODEL ==================
# ✅ CHANGED: Simplified the training function as we no longer have categorical features
def train_and_save_model(csv_path, model_path):
    print(f"Training a new model from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip().lower() for col in df.columns]

        if 'label' not in df.columns:
            raise ValueError("'label' column not found in dataset.")

        # Use the EXPECTED_COLUMNS list for features
        X = df[EXPECTED_COLUMNS] 
        y = df["label"]
        
        # Since all our features are now numeric, the pipeline is much simpler.
        # We don't need ColumnTransformer or OneHotEncoder anymore.
        model_pipeline = Pipeline(
            steps=[
                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
            ])

        model_pipeline.fit(X, y)
        joblib.dump(model_pipeline, model_path)
        print(f"Model trained and saved successfully to {model_path}")
        return model_pipeline

    except Exception as e:
        print(f"Error during model training: {e}")
        return None

# Load or train crop model
if not os.path.exists(MODEL_PATH):
    crop_model = train_and_save_model(CSV_PATH, MODEL_PATH)
    if crop_model is None:
        print("Model training failed. Exiting...")
        exit()
else:
    print(f"Loading existing crop model from {MODEL_PATH}")
    crop_model = joblib.load(MODEL_PATH)

# ================== ROUTES ==================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # ✅ FIX 1: Convert all incoming keys to lowercase
        data = {key.lower(): value for key, value in data.items()}
        
        print(f"Received data for prediction (lowercase keys): {data}")

        # Create DataFrame using the simplified EXPECTED_COLUMNS list
        input_data = {col: [data.get(col)] for col in EXPECTED_COLUMNS}
        input_df = pd.DataFrame(input_data)
        
        # ✅ FIX 2: Ensure all data is in a numeric format (float) for the model
        input_df = input_df.astype(float)
        
        print(f"DataFrame for prediction:\n{input_df}")

        prediction = crop_model.predict(input_df)[0]
        print(f"Prediction result: {prediction}")

        return jsonify({"recommended_crop": prediction})

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Failed to process the request."}), 500

@app.route("/chat", methods=["POST"])
def chat():
    # This part is fine
    user_msg = request.json.get("message", "")
    dataset_reply = search_dataset(user_msg)
    if dataset_reply:
        return jsonify({"reply": dataset_reply})
    input_text = f"Answer concisely about crops and farming: {user_msg}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = chat_model.generate(**inputs, max_new_tokens=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"reply": reply})

# ================== MAIN ==================
if __name__ == "__main__":
    app.run(port=5000, debug=True)