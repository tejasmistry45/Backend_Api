import json
import os
from django.conf import settings
from django.db import connections
from django.http import FileResponse
import faiss
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from api.llm_utils import extract_resume_info
from .utils import extract_text_from_pdf, extract_text_from_docx
from .embeddings_utils import EMBED_FILE, ID_PATH, add_resume_to_index, MODEL, INDEX, EMBEDS
import numpy as np
import pyodbc


class ProcessResumePathAPIView(APIView):
    def post(self, request):
        rel_path = request.data.get('path')
        resume_id=request.data.get('resumeid')
        if not rel_path:
            return Response({'error': 'No path provided.'}, status=status.HTTP_400_BAD_REQUEST)
 
        full_path = os.path.join(settings.MEDIA_ROOT, rel_path)
        if not os.path.exists(full_path):
            return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)
 
        if rel_path.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(full_path)
        elif rel_path.lower().endswith('.docx'):
            resume_text = extract_text_from_docx(full_path)
        else:
            return Response({'error': 'Unsupported format.'}, status=status.HTTP_400_BAD_REQUEST)
 
        with connections['external_db'].cursor() as cursor:
            cursor.execute(
                "UPDATE JobInquiry SET resume_text = %s WHERE Id = %s",
                [resume_text, resume_id]
            )
       
        # Add to index using utility
        add_resume_to_index(resume_text,resume_id)
 
        return Response({'message': 'Processed successfully', 'resume_id': resume_id}, status=status.HTTP_201_CREATED)


class FindMatchesAPIView(APIView):
    def post(self, request):
        data = request.data

        # 1️⃣ Try plain job_description
        job_desc = data.get('job_description', '').strip()

        # 2️⃣ Try structured input if job_description is missing
        if not job_desc:
            job_title = data.get('job_title', '').strip()
            location = data.get('location', '').strip()
            years_exp = data.get('years_exp', '').strip()
            skills = data.get('Skills', '').strip()
            qualifications = data.get('Qualifications', '').strip()

            if all([job_title, location, years_exp]):
                job_desc = (
                    f"Job Title: {job_title}; "
                    f"Location: {location}; "
                    f"Years Exp: {years_exp}; "
                    f"Skills: {skills}; "
                    f"Qualifications: {qualifications}"
                )
            else:
                return Response(
                    {'error': 'Either provide "job_description" or all of "job_title", "location", and "years_exp".'},
                    status=status.HTTP_400_BAD_REQUEST
                )

        if not job_desc:
            return Response({'error': 'No job description provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # 3️⃣ Ensure FAISS index is populated
        if INDEX.ntotal == 0:
            return Response(
                {'error': 'FAISS index is empty. Please upload resumes first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        # 4️⃣ Encode query and search
        q_vec = MODEL.encode([job_desc]).astype('float32')
        D, I = INDEX.search(q_vec, k=5)  # top 5 matches

        # 5️⃣ Load resume ID mapping
        try:
            id_list = np.load(ID_PATH, allow_pickle=True).tolist()
        except FileNotFoundError:
            return Response(
                {'error': f'ID mapping file not found at {ID_PATH}.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 6️⃣ Build result list
        results = []
        with connections['external_db'].cursor() as cursor:
            for distance, index in zip(D[0], I[0]):
                index = int(index)
                if index == -1 or index >= len(id_list):
                    continue

                resume_id = id_list[index]

                # Confirm resume ID exists in database
                cursor.execute("SELECT 1 FROM JobInquiry WHERE id = %s", [resume_id])
                if cursor.fetchone():
                    results.append({
                        'resume_id': resume_id,
                        'score': float(distance)
                    })

        return Response({'matches': results})


class ResumeKeyPointsAPIView(APIView):
    def get(self, request, resume_id):
        try:
            # Connect to the Resume database
            conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=localhost;'
                'DATABASE=SkyHR;'
                'Trusted_Connection=yes;'
            )
            cursor = conn.cursor()

            # Fetch resume text from JobInquiry table
            cursor.execute("SELECT resume_text FROM JobInquiry WHERE id = ?", resume_id)
            row = cursor.fetchone()

            if not row:
                return Response({'error': 'Resume not found.'}, status=status.HTTP_404_NOT_FOUND)

            resume_text = row.resume_text

            # Extract insights directly
            insights = extract_resume_info(resume_text)

            # Return insights in JSON format
            return Response(insights, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            cursor.close()
            conn.close()