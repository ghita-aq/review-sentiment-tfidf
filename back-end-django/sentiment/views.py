import json
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from . import models_loader

logger = logging.getLogger(__name__)


def build_description(sentiment, confidence):
    """Build a human-readable description of the sentiment result."""
    if confidence >= 0.8:
        level = 'strongly'
    elif confidence >= 0.6:
        level = 'moderately'
    else:
        level = 'slightly'

    messages = {
        'positive': f'This review expresses a {level} positive sentiment. The customer appears satisfied with their experience.',
        'negative': f'This review expresses a {level} negative sentiment. The customer seems dissatisfied and may need attention.',
        'neutral': f'This review has a {level} neutral sentiment. The customer\'s opinion appears balanced or mixed.'
    }
    return messages.get(sentiment, 'Sentiment analysis complete.')


def predict(request):
    """
    Main view: renders the prediction form and handles POST submissions.
    - GET: Show the form
    - POST: Process the review and show the result on the same page
    """
    sentiment = None
    confidence = None
    description = None
    initial_review = ''
    model_error = not models_loader.is_model_loaded()

    if request.method == 'POST':
        review = request.POST.get('review', '').strip()
        if review:
            initial_review = review
            result = models_loader.predict_review(review)
            if result:
                sentiment = result['sentiment']
                confidence = round(result['confidence'] * 100, 1)
                description = build_description(sentiment, result['confidence'])
            else:
                model_error = True

    return render(request, 'sentiment/predict.html', {
        'sentiment': sentiment,
        'confidence': confidence,
        'description': description,
        'initial_review': initial_review,
        'model_error': model_error,
    })


@require_POST
def predict_api(request):
    """
    JSON API endpoint for programmatic access.

    Request body (JSON):
        {"review": "This product is amazing!"}

    Response (JSON):
        {"sentiment": "positive", "confidence": 0.92}
    """
    if not models_loader.is_model_loaded():
        return JsonResponse(
            {'error': 'Model not loaded. Please train the model first.'},
            status=503
        )

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, TypeError):
        return JsonResponse({'error': 'Invalid JSON body.'}, status=400)

    review = data.get('review', '').strip()
    if not review:
        return JsonResponse({'error': 'Please provide a review text.'}, status=400)

    result = models_loader.predict_review(review)
    if result is None:
        return JsonResponse(
            {'error': 'Prediction failed. Model may not be loaded.'},
            status=503
        )

    return JsonResponse({
        'sentiment': result['sentiment'],
        'confidence': round(result['confidence'], 4)
    })
