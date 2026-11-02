from django.urls import path
from . import views

app_name = 'sentiment'

urlpatterns = [
    path('', views.predict, name='predict'),
    path('api/predict/', views.predict_api, name='predict_api'),
]
