"""
URL configuration for sentiment_app project.
"""
from django.urls import path, include

urlpatterns = [
    path('', include('sentiment.urls')),
]
