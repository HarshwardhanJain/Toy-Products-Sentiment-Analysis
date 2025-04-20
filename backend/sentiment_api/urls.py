from django.urls import path
from . import views

urlpatterns = [
    path('predict/sentiment/', views.predict_sentiment, name='predict_sentiment'),
    path('random-image/', views.get_random_image, name='random_image'),
]
