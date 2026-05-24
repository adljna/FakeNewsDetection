import os
from flask import Flask, request, jsonify
from src.prediction import predict_news

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Fake News Detection API",
        "status": "running",
        "version": "1.0",
        "available_endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "Fake News Detection API",
            "version": "1.0",
            "model_version": "pipeline_v1",
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        if "text" not in data:
            return jsonify({"error": "Text field is required"}), 400

        text = str(data["text"]).strip()

        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        if len(text) < 20:
            return jsonify({"error": "Text too short for prediction"}), 400

        result = predict_news(text)

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
