from django.urls import path
from .views import ProcessResumePathAPIView, FindMatchesAPIView, ResumeKeyPointsAPIView 

urlpatterns = [
    path('process-path/', ProcessResumePathAPIView.as_view(), name='process-path'),
    path('find-matches/', FindMatchesAPIView.as_view(), name='find-matches'),
    path('resume/<str:resume_id>/insights/', ResumeKeyPointsAPIView.as_view(), name='resume-insights'),
]