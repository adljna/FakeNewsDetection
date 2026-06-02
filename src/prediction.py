import pickle
import os
import re

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "pipeline_v1.pkl")

# load model sekali saat aplikasi start
with open(MODEL_PATH, "rb") as file:
    pipeline = pickle.load(file)


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    return text.lower().strip()


def predict_news(text):

    clean = preprocess(text)

    prediction = pipeline.predict([clean])[0]

    confidence = None

    # cek apakah model support probability
    if hasattr(pipeline, "predict_proba"):

        probs = pipeline.predict_proba([clean])[0]

        confidence = round(float(max(probs)) * 100, 2)

    # mapping hasil prediksi
    if prediction == 0:
        label = "FAKE NEWS"
        is_fake = True
    else:
        label = "REAL NEWS"
        is_fake = False

    return {
        "prediction": label,
        "is_fake": is_fake,
        "confidence": confidence,
        "model_version": "v1",
    }
