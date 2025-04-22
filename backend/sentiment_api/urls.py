from django.urls import path
from . import views

urlpatterns = [
    path('predict/sentiment/', views.predict_sentiment, name='predict_sentiment'),
    path('all-images/', views.get_all_images, name='all_images'),
    path('clear-cache/', views.clear_cache, name='clear_cache'),
]
