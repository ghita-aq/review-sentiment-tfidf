Step 1: Install dependencies (one-time only)
Open a terminal, navigate to the project folder, and install the Python
packages:

cd "Z:\MQL\S3\Gen AI\review-sentiment-tfidf"
pip install -r requirements.txt

This installs Django, scikit-learn, NLTK, joblib, and all other
dependencies.

---

Step 2: Start the Django app
From the back-end-django folder, run:

cd back-end-django
python manage.py runserver

You should see:

Starting development server at http://127.0.0.1:8000/

The model files (models/best_model.pkl, vectorizer.pkl,
label_encoder.pkl) will be loaded automatically at startup.

---

Step 3: Open the app in your browser
Go to http://127.0.0.1:8000/ — you'll see the sentiment analysis form.

How to use it:

1. Type or paste a customer review in the text box
2. Click Analyze Sentiment
3. The result appears instantly: Positive, Negative, or Neutral with a
confidence percentage

You can also click the example buttons (😊 Positive / 😞 Negative / 😐
Neutral) to try pre-filled reviews.

---

Quick reference — all the commands
╭───────────────────┬─────────────────────────────────────────────────╮
│ Action │ Command │
├───────────────────┼─────────────────────────────────────────────────┤
│ Install packages │ pip install -r requirements.txt (from │
│ │ review-sentiment-tfidf/) │
│ Start the web app │ cd back-end-django; python manage.py runserver │
│ Retrain the model │ python scripts/train.py (from │
│ │ review-sentiment-tfidf/) │
│ Evaluate the │ python scripts/evaluate.py (from │
│ model │ review-sentiment-tfidf/) │
│ Test │ python scripts/preprocess.py (from │
│ preprocessing │ review-sentiment-tfidf/) │
╰───────────────────┴─────────────────────────────────────────────────╯

---

API access (for developers)
The app also exposes a JSON API at http://127.0.0.1:8000/api/predict/.
Example request:

curl -X POST http://127.0.0.1:8000/api/predict/ `
  -H "Content-Type: application/json" `
  -d '{"review": "This product is amazing!"}'

Response:

{"sentiment": "positive", "confidence": 0.95}