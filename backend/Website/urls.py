from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('sentiment_api.urls')),  # All URLs in sentiment_api/ will be prefixed with /api/
]
