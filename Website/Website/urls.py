from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # Include the sentiment_api app's URLs under the base path "api/"
    path('api/', include('sentiment_api.urls')),
]
