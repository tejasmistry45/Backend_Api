from django.urls import path
from .views import find_matches,get_key_points, upload_resume
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('upload/', upload_resume, name='upload_resume'),
    path('findmatches/', find_matches),
    path('getkeypoints/<int:resume_id>/',get_key_points)
    
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
