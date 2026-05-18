from flask import Flask, render_template, request
import re
import pickle
import os

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# ── Load the pipeline (TF-IDF vectorizer + classifier dalam satu objek) ──────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'pipeline.pkl')
pipeline = pickle.load(open(MODEL_PATH, 'rb'))


def preprocess(text):
    """Bersihkan teks: hapus karakter non-huruf, ubah ke lowercase."""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    return text.lower()


def fake_news_det(news):
    """
    Terima string berita, kembalikan string hasil prediksi.
    Fungsi ini WAJIB return nilai agar bisa dikirim ke template.
    """
    clean = preprocess(news)
    prediction = pipeline.predict([clean])[0]   # 1 = Real, 0 = Fake

    if prediction == 0:
        return "Berita ini terdeteksi sebagai FAKE NEWS (Berita Palsu)."
    else:
        return "Berita ini terdeteksi sebagai REAL NEWS (Berita Asli)."


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get('news', '').strip()
        if not message:
            return render_template('index.html',
                                   prediction="Masukkan teks berita terlebih dahulu.")
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    return render_template('index.html', prediction="Terjadi kesalahan.")


if __name__ == '__main__':
    app.run(debug=True)
