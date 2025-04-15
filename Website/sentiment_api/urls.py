from django.urls import path
from .views import predict_sentiment

urlpatterns = [
    path('predict/sentiment/', predict_sentiment, name='predict_sentiment'),
]
