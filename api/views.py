import os
import time
import numpy as np
import pandas as pd
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from sentence_transformers import SentenceTransformer
import faiss

from api.models import Resume
from backend import settings
from .utils import extract_text_from_pdf, extract_text_from_docx
from .llm_utils import extract_resume_info

# Load model and index only once
model = SentenceTransformer('all-mpnet-base-v2')
index = faiss.read_index('embeddings/faiss_index.index')
df = pd.read_csv('data/Final_Resume.csv')
resume_ids = df['resume_id'].astype(str).tolist()
resume_texts = df['resume_text'].tolist()


class UploadResumeAPIView(APIView):
    def post(self, request):
        uploaded_file = request.FILES.get('resume_file')
        if not uploaded_file:
            return Response({'error': 'No file uploaded.'}, status=status.HTTP_400_BAD_REQUEST)

        original_filename = uploaded_file.name
        file_storage_path = os.path.join('resumes', original_filename)
        full_path = os.path.join(settings.MEDIA_ROOT, file_storage_path)

        # Prevent duplicate files
        if Resume.objects.filter(file_path=file_storage_path).exists():
            return Response({'error': 'This resume is already uploaded.'}, status=status.HTTP_409_CONFLICT)

        # Save file
        default_storage.save(file_storage_path, ContentFile(uploaded_file.read()))

        # Extract text
        if original_filename.endswith('.pdf'):
            text = extract_text_from_pdf(full_path)
        elif original_filename.endswith('.docx'):
            text = extract_text_from_docx(full_path)
        else:
            return Response({'error': 'Unsupported file format. Only PDF and DOCX allowed.'}, status=status.HTTP_400_BAD_REQUEST)

        # Save to DB
        resume = Resume(file_path=file_storage_path, resume_text=text)
        resume.save()

        return Response({
            'message': 'Resume uploaded successfully.',
            'filename': original_filename,
            'resume_id': resume.id
        }, status=status.HTTP_201_CREATED)


class FindMatchesAPIView(APIView):
    def post(self, request):
        data = request.data
        job_title = data.get('job_title', '')
        location = data.get('location', '')
        years_exp = data.get('years_exp', '')
        skills = data.get('Skills', '')
        qualifications = data.get('Qualifications', '')

        job_text = f"{job_title} | {location} | {years_exp} yrs exp | Skills: {skills} | Qualifications: {qualifications}"
        job_embedding = model.encode([job_text])

        # FAISS search
        D, I = index.search(np.array(job_embedding).astype(np.float32), k=5)

        matches = []
        for i, idx in enumerate(I[0]):
            match = {
                "resume_id": resume_ids[idx],
                "score": D[0][i],
            }
            matches.append(match)

        matches.sort(key=lambda x: x['score'], reverse=True)

        return Response({'result': matches})


class ResumeKeyPointsAPIView(APIView):
    def get(self, request, resume_id):
        try:
            resume_id_str = str(resume_id)
            resume_ids_str = [str(rid) for rid in resume_ids]

            if resume_id_str not in resume_ids_str:
                return Response({'error': 'Resume ID not found.'}, status=status.HTTP_404_NOT_FOUND)

            idx = resume_ids_str.index(resume_id_str)
            resume_text = resume_texts[idx]

            start = time.time()
            llm_response = extract_resume_info(resume_text)
            end = time.time()
            print(f"LLM processed in {end - start:.2f} seconds.")

            return Response(data=llm_response, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
