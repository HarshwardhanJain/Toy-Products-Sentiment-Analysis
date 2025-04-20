from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('sentiment_api.urls')),  # All URLs in sentiment_api/ will be prefixed with /api/
]

# Serve media files (sample-toys images)
from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
