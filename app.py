"""Compatibility entry point for running the Flask app locally.

The main MLOps service is implemented in src/api.py. This file is kept so the
old command `python app.py` still works.
"""

import os

app = Flask(__name__, template_folder="./templates", static_folder="./static")

# Membaca file pipeline utuh yang baru saja kita buat dari script classifier
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "pipeline_v1.pkl")
pipeline = pickle.load(open(MODEL_PATH, "rb"))


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    return text.lower()


def fake_news_det(news):
    clean = preprocess(news)

    # Memprediksi kelas dan probabilitas langsung menggunakan gerbong Pipeline
    prediction = pipeline.predict([clean])[0]
    probabilities = pipeline.predict_proba([clean])[0]

    # Pemetaan probabilitas berdasarkan index scikit-learn (0 = False/Fake, 1 = True/Real)
    prob_fake = probabilities[0] * 100
    prob_real = probabilities[1] * 100

    # Memastikan sinkronisasi output dengan logika integer label dataset
    if prediction == 0:
        label = "FAKE NEWS (Berita Palsu)"
        confidence = f"{prob_fake:.2f}"  # Mengambil persentase tingkat kepalsuan
        is_fake = True
    else:
        label = "REAL NEWS (Berita Asli)"
        confidence = f"{prob_real:.2f}"  # Mengambil persentase tingkat keaslian
        is_fake = False

    return {
        "label": label,
        "confidence": confidence,
        "is_fake": is_fake,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("news", "").strip()
    if not message:
        return render_template(
            "index.html", error="Masukkan teks berita terlebih dahulu.", news_text=""
        )

    result = fake_news_det(message)
    return render_template(
        "index.html",
        news_text=message,
        pred_label=result["label"],
        pred_confidence=result["confidence"],
        pred_is_fake=result["is_fake"],
    )


if __name__ == "__main__":
    app.run(debug=True)
