import json
import os
import subprocess
import traceback
from django.conf import settings
from django.db import connections
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from api.llm_utils import extract_resume_info
from .utils import clean_ocr_text, convert_docx_to_pdf, convert_pdf_to_images, run_llamaocr_on_image
from .embeddings_utils import ID_PATH, add_resume_to_index, MODEL, INDEX
import numpy as np
import pyodbc


import logging
logger = logging.getLogger(__name__)

class ProcessResumePathAPIView(APIView):
    def post(self, request):
        resume_id = request.data.get('resumeid')
        rel_path = request.data.get('path')

        if not rel_path:
            return Response({'error': 'No path provided.'}, status=status.HTTP_400_BAD_REQUEST)

        full_path = os.path.join(settings.MEDIA_ROOT, rel_path)
        if not os.path.exists(full_path):
            return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

        try:
            # Convert DOCX to PDF if needed
            if full_path.lower().endswith('.docx'):
                logger.info("Converting DOCX to PDF")
                full_path = convert_docx_to_pdf(full_path)

            if not full_path.lower().endswith('.pdf'):
                return Response({'error': 'Only PDF or DOCX files are supported.'}, status=status.HTTP_400_BAD_REQUEST)

            # Convert PDF to images
            logger.info("Converting PDF pages to images")
            image_paths = convert_pdf_to_images(full_path)

            # Run OCR on each image and gather text
            ocr_texts = []
            for image_path in image_paths:
                logger.info(f"Running OCR on image: {image_path}")
                text = run_llamaocr_on_image(image_path)
                ocr_texts.append(text)

            # Clean up the image files after OCR
            for image_path in image_paths:
                try:
                    os.remove(image_path)
                    logger.info(f"Deleted temporary image: {image_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete image {image_path}: {e}")

            raw_text = "\n\n".join(ocr_texts)
            full_text = clean_ocr_text(raw_text)

            # Save to DB
            with connections['external_db'].cursor() as cursor:
                cursor.execute(
                    "UPDATE JobInquiry SET resume_text = %s WHERE Id = %s",
                    [full_text, resume_id]
                )

            # Add to FAISS index (assumed implemented)
            add_resume_to_index(full_text, resume_id)

            return Response({'message': 'Processed successfully', 'resume_id': resume_id}, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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