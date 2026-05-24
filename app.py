from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fake News Detection API is running",
        "status": "ok",
        "available_endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Missing 'text' field"
        }), 400

    text = data["text"]

    # TODO: sesuaikan dengan pipeline/model kamu
    # prediction = model.predict([text])[0]

    return jsonify({
        "text": text,
        "prediction": "placeholder"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)