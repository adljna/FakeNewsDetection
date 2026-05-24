from flask import Flask, request, jsonify
from src.prediction import predict_news

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():

    return jsonify({
        "status": "healthy",
        "service": "Fake News Detection API",
        "version": "1.0"
    })


@app.route('/api/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        # validasi request kosong
        if not data:
            return jsonify({
                "error": "Request body is required"
            }), 400

        # validasi field text tidak ada
        if 'text' not in data:
            return jsonify({
                "error": "Text field is required"
            }), 400

        text = str(data['text']).strip()

        # validasi text kosong
        if not text:
            return jsonify({
                "error": "Text cannot be empty"
            }), 400

        # validasi text terlalu pendek
        if len(text) < 20:
            return jsonify({
                "error": "Text too short for prediction"
            }), 400

        # jalankan prediksi
        result = predict_news(text)

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)