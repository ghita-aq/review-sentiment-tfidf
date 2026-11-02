/* ============================================
   Review Sentiment Classifier - Application Logic
   ============================================ */

(function () {
    'use strict';

    // API configuration — adjust this URL when deploying with Django
    const API_URL = '/api/predict/';

    // DOM Elements
    const form = document.getElementById('sentiment-form');
    const reviewTextarea = document.getElementById('review-text');
    const charCount = document.getElementById('char-count');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    const resultSection = document.getElementById('result-section');
    const sentimentBadge = document.getElementById('sentiment-badge');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');
    const resultDescription = document.getElementById('result-description');
    const errorToast = document.getElementById('error-toast');
    const errorMessage = document.getElementById('error-message');
    const exampleButtons = document.querySelectorAll('.example-btn');

    // ============================================
    // Character Counter
    // ============================================

    reviewTextarea.addEventListener('input', function () {
        const len = this.value.length;
        charCount.textContent = len + ' character' + (len !== 1 ? 's' : '');
    });

    // ============================================
    // Example Buttons
    // ============================================

    exampleButtons.forEach(function (btn) {
        btn.addEventListener('click', function () {
            var review = this.getAttribute('data-review');
            reviewTextarea.value = review;
            charCount.textContent = review.length + ' characters';
            reviewTextarea.focus();
        });
    });

    // ============================================
    // Form Submission
    // ============================================

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        var review = reviewTextarea.value.trim();
        if (!review) {
            showError('Please enter a review before analyzing.');
            return;
        }

        setLoading(true);
        resultSection.setAttribute('hidden', '');

        fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({ review: review })
        })
            .then(function (response) {
                if (!response.ok) {
                    return response.json().then(function (data) {
                        throw new Error(data.error || 'Server error. Please try again.');
                    });
                }
                return response.json();
            })
            .then(function (data) {
                displayResult(data);
            })
            .catch(function (error) {
                showError(error.message || 'Failed to connect to the server.');
            })
            .finally(function () {
                setLoading(false);
            });
    });

    // ============================================
    // Display Result
    // ============================================

    function displayResult(data) {
        var sentiment = data.sentiment || 'unknown';
        var confidence = data.confidence !== undefined ? data.confidence : 0;
        var confidencePercent = Math.round(confidence * 100);

        // Badge
        sentimentBadge.textContent = sentiment;
        sentimentBadge.className = 'badge ' + sentiment;

        // Confidence
        confidenceValue.textContent = confidencePercent + '%';
        confidenceFill.className = 'confidence-fill ' + sentiment;
        // Animate the bar
        requestAnimationFrame(function () {
            confidenceFill.style.width = confidencePercent + '%';
        });

        // Description
        resultDescription.textContent = buildDescription(sentiment, confidence);

        // Show result
        resultSection.removeAttribute('hidden');
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function buildDescription(sentiment, confidence) {
        var level;
        if (confidence >= 0.8) level = 'strongly';
        else if (confidence >= 0.6) level = 'moderately';
        else level = 'slightly';

        var messages = {
            positive: 'This review expresses a ' + level + ' positive sentiment. The customer appears satisfied with their experience.',
            negative: 'This review expresses a ' + level + ' negative sentiment. The customer seems dissatisfied and may need attention.',
            neutral: 'This review has a ' + level + ' neutral sentiment. The customer\'s opinion appears balanced or mixed.'
        };

        return messages[sentiment] || 'Sentiment analysis complete.';
    }

    // ============================================
    // Loading State
    // ============================================

    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        if (isLoading) {
            btnText.setAttribute('hidden', '');
            btnLoader.removeAttribute('hidden');
        } else {
            btnText.removeAttribute('hidden');
            btnLoader.setAttribute('hidden', '');
        }
    }

    // ============================================
    // Error Handling
    // ============================================

    function showError(message) {
        errorMessage.textContent = message;
        errorToast.removeAttribute('hidden');

        setTimeout(function () {
            errorToast.setAttribute('hidden', '');
        }, 5000);
    }

    // ============================================
    // CSRF Token Helper (Django)
    // ============================================

    function getCSRFToken() {
        var name = 'csrftoken';
        var cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue || '';
    }

})();
