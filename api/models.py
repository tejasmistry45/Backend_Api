from django.db import models

class Resume(models.Model):
    file_path = models.FileField(upload_to='resumes/')  # stores path under MEDIA_ROOT/resumes/
    resume_text = models.TextField(blank=True)  # blank until text is extracted

    def __str__(self):
        return str(self.file_path)
