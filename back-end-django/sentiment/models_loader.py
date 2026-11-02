import os
import joblib
import logging

logger = logging.getLogger(__name__)

# Resolve model paths relative to the project root (review-sentiment-tfidf/)
# The models/ directory sits alongside back-end-django/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

model = None
vectorizer = None
label_encoder = None


def load_models():
    """Load the trained model, vectorizer, and label encoder from disk."""
    global model, vectorizer, label_encoder

    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Models directory not found: {MODELS_DIR}")
        return False

    missing = []
    for name, path in [('model', MODEL_PATH), ('vectorizer', VECTORIZER_PATH), ('label encoder', LABEL_ENCODER_PATH)]:
        if not os.path.exists(path):
            missing.append(name)

    if missing:
        logger.warning(f"Missing model files: {', '.join(missing)}. Run 'python scripts/train.py' to train the model.")
        return False

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("Model artifacts loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return False


def is_model_loaded():
    """Check whether all model components are loaded."""
    return model is not None and vectorizer is not None and label_encoder is not None


def preprocess_text(text):
    """
    Preprocess review text using the same pipeline as training.
    Import preprocess module dynamically to avoid issues if it's not yet implemented.
    """
    try:
        # Try to import the project's preprocess module
        scripts_dir = os.path.join(BASE_DIR, 'scripts')
        if scripts_dir not in os.sys.path:
            os.sys.path.insert(0, scripts_dir)
        from preprocess import preprocess_review
        return preprocess_review(text)
    except ImportError:
        # Fallback: basic preprocessing if the module is not available
        import re
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
        text = text.lower().strip()
        return text


def predict_review(text):
    """
    Predict sentiment for a given review text.

    Returns:
        dict: {
            'sentiment': str,  # 'positive', 'negative', or 'neutral'
            'confidence': float  # 0.0 to 1.0
        }
        or None if the model is not loaded.
    """
    if not is_model_loaded():
        return None

    processed = preprocess_text(text)
    if not processed.strip():
        return {'sentiment': 'neutral', 'confidence': 0.0}

    # Transform text with TF-IDF vectorizer
    tfidf_features = vectorizer.transform([processed])

    # Get prediction probabilities
    probabilities = model.predict_proba(tfidf_features)[0]
    predicted_class_idx = probabilities.argmax()
    confidence = float(probabilities[predicted_class_idx])

    # Decode the label
    sentiment = label_encoder.inverse_transform([predicted_class_idx])[0]

    return {
        'sentiment': str(sentiment),
        'confidence': confidence
    }
