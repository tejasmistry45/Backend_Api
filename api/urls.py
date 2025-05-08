from django.urls import path
from .views import UploadResumeAPIView, FindMatchesAPIView, ResumeKeyPointsAPIView

urlpatterns = [
    path('api/upload-resume/', UploadResumeAPIView.as_view(), name='upload-resume'),
    path('api/find-matches/', FindMatchesAPIView.as_view(), name='find-matches'),
    path('api/resume/<int:resume_id>/key-points/', ResumeKeyPointsAPIView.as_view(), name='resume-key-points'),
]

