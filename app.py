from flask import Flask, render_template, request
import re
import pickle
import os

app = Flask(__name__, template_folder='./templates', static_folder='./static')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'pipeline.pkl')
pipeline = pickle.load(open(MODEL_PATH, 'rb'))


def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    return text.lower()


def fake_news_det(news):
    clean = preprocess(news)
    prediction = pipeline.predict([clean])[0]
    prob = pipeline.predict_proba([clean])[0]  # [prob_fake, prob_real]

    if prediction == 0:
        label = "FAKE NEWS (Berita Palsu)"
        confidence = round(prob[0] * 100, 1)   # probabilitas FAKE
        is_fake = True
    else:
        label = "REAL NEWS (Berita Asli)"
        confidence = round(prob[1] * 100, 1)   # probabilitas REAL
        is_fake = False

    return {
        "label": label,
        "confidence": confidence,
        "is_fake": is_fake,
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('news', '').strip()
    if not message:
        return render_template('index.html',
                               error="Masukkan teks berita terlebih dahulu.",
                               news_text="")

    result = fake_news_det(message)
    return render_template('index.html',
                           news_text=message,
                           pred_label=result['label'],
                           pred_confidence=result['confidence'],
                           pred_is_fake=result['is_fake'])


if __name__ == '__main__':
    app.run(debug=True)