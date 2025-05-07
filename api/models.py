from django.db import models
import os

def resume_upload_path(instance, filename):
    # Store using original name inside "resumes/" folder
    return f"resumes/{filename}"

class Resume(models.Model):
    file_path = models.FileField(upload_to=resume_upload_path, unique=True)
    resume_text = models.TextField(blank=True)

    def __str__(self):
        return os.path.basename(self.file_path.name)
