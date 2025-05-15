import json
import os
from django.conf import settings
from django.db import connections
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from api.llm_utils import extract_resume_info
from .utils import extract_text_from_pdf, extract_text_from_docx
from .embeddings_utils import ID_PATH, add_resume_to_index, MODEL, INDEX
import numpy as np
import pyodbc


import logging
logger = logging.getLogger(__name__)

class ProcessResumePathAPIView(APIView):
    def post(self, request):
        logger.info("Received request to process resume.")
        rel_path = request.data.get('path')
        resume_id = request.data.get('resumeid')

        if not rel_path:
            logger.warning("No path provided in request.")
            return Response({'error': 'No path provided.'}, status=status.HTTP_400_BAD_REQUEST)

        full_path = os.path.join(settings.MEDIA_ROOT, rel_path)
        if not os.path.exists(full_path):
            logger.warning(f"File not found at path: {full_path}")
            return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

        try:
            if rel_path.lower().endswith('.pdf'):
                logger.info(f"Extracting text from PDF: {full_path}")
                resume_text = extract_text_from_pdf(full_path)
            elif rel_path.lower().endswith('.docx'):
                logger.info(f"Extracting text from DOCX: {full_path}")
                resume_text = extract_text_from_docx(full_path)
            else:
                logger.warning(f"Unsupported file format: {rel_path}")
                return Response({'error': 'Unsupported format.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return Response({'error': 'Error processing resume text.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            with connections['external_db'].cursor() as cursor:
                logger.info(f"Updating resume_text for resume_id: {resume_id}")
                cursor.execute(
                    "UPDATE JobInquiry SET resume_text = %s WHERE Id = %s",
                    [resume_text, resume_id]
                )
        except Exception as e:
            logger.error(f"Database update failed: {str(e)}")
            return Response({'error': 'Database update failed.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        logger.info(f"Adding resume to FAISS index for ID: {resume_id}")
        add_resume_to_index(resume_text, resume_id)

        logger.info(f"Resume processed successfully for ID: {resume_id}")
        return Response({'message': 'Processed successfully', 'resume_id': resume_id}, status=status.HTTP_201_CREATED)


class FindMatchesAPIView(APIView):
    def post(self, request):
        logger.info("Received request to find resume matches.")
        data = request.data

        job_desc = data.get('job_description', '').strip()
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
                logger.info("Constructed job description from structured fields.")
            else:
                logger.warning("Insufficient structured data provided.")
                return Response(
                    {'error': 'Either provide "job_description" or all of "job_title", "location", and "years_exp".'},
                    status=status.HTTP_400_BAD_REQUEST
                )

        if not job_desc:
            logger.warning("No job description provided.")
            return Response({'error': 'No job description provided.'}, status=status.HTTP_400_BAD_REQUEST)

        if INDEX.ntotal == 0:
            logger.warning("FAISS index is empty.")
            return Response(
                {'error': 'FAISS index is empty. Please upload resumes first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        logger.info("Encoding job description and searching FAISS index.")
        q_vec = MODEL.encode([job_desc]).astype('float32')
        D, I = INDEX.search(q_vec, k=5)

        try:
            id_list = np.load(ID_PATH, allow_pickle=True).tolist()
            logger.info("Resume ID mapping loaded.")
        except FileNotFoundError:
            logger.error(f"ID mapping file not found at {ID_PATH}.")
            return Response(
                {'error': f'ID mapping file not found at {ID_PATH}.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        results = []
        with connections['external_db'].cursor() as cursor:
            for distance, index in zip(D[0], I[0]):
                index = int(index)
                if index == -1 or index >= len(id_list):
                    logger.debug(f"Skipping invalid index: {index}")
                    continue

                resume_id = id_list[index]
                cursor.execute("SELECT 1 FROM JobInquiry WHERE id = %s", [resume_id])
                if cursor.fetchone():
                    results.append({
                        'resume_id': resume_id,
                        'score': float(distance)
                    })
                    logger.debug(f"Match found: Resume ID {resume_id} with score {float(distance)}")

        logger.info(f"Total matches found: {len(results)}")
        return Response({'matches': results})


class ResumeKeyPointsAPIView(APIView):
    def get(self, request, resume_id):
        logger.info(f"Received request to extract key points for resume ID: {resume_id}")
        try:
            conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=localhost;'
                'DATABASE=SkyHR;'
                'Trusted_Connection=yes;'
            )
            cursor = conn.cursor()

            cursor.execute("SELECT resume_text FROM JobInquiry WHERE id = ?", resume_id)
            row = cursor.fetchone()

            if not row:
                logger.warning(f"No resume found for ID: {resume_id}")
                return Response({'error': 'Resume not found.'}, status=status.HTTP_404_NOT_FOUND)

            resume_text = row.resume_text
            logger.info("Extracting insights from resume text.")
            insights = extract_resume_info(resume_text)

            return Response(insights, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error extracting resume key points: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            cursor.close()
            conn.close()
            logger.info("Database connection closed.")



#  