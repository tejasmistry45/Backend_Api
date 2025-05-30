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
from .logger_function import logger_function


filename=os.path.basename(__file__)[:-3]

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
                logger_function.info(filename,"Converting DOCX to PDF",1)
                full_path = convert_docx_to_pdf(full_path)

            if not full_path.lower().endswith('.pdf'):
                return Response({'error': 'Only PDF or DOCX files are supported.'}, status=status.HTTP_400_BAD_REQUEST)

            # Convert PDF to images
            logger_function(filename,"Converting PDF pages to images",1)
            image_paths = convert_pdf_to_images(full_path)

            # Run OCR on each image and gather text
            ocr_texts = []
            for image_path in image_paths:
                logger_function(filename,f"Running OCR on image: {image_path}",1)
                text = run_llamaocr_on_image(image_path)
                ocr_texts.append(text)

            # Clean up the image files after OCR
            for image_path in image_paths:
                try:
                    os.remove(image_path)
                    logger_function(filename,f"Deleted temporary image: {image_path}",1)
                except Exception as e:
                    logger_function(filename,f"Failed to delete image {image_path}: {e}",1)

            raw_text = "\n\n".join(ocr_texts)
            full_text = clean_ocr_text(raw_text)

            # Save to DB
            with connections["default"].cursor() as cursor:
                cursor.execute(
                    "UPDATE JobInquiry SET resume_text = %s WHERE Id = %s",
                    [full_text, resume_id]
                )

            # Add to FAISS index (assumed implemented)
            add_resume_to_index(full_text, resume_id)

            return Response({'message': 'Processed successfully', 'resume_id': resume_id}, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger_function(filename,f"Processing failed: {e}",1)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FindMatchesAPIView(APIView):
    def post(self, request):
        logger_function(filename,"Received request to find resume matches.",1)
        data = request.data

        # Extract structured fields
        job_title = data.get('job_title', '').strip()
        location = data.get('location', '').strip()
        years_exp = data.get('years_exp', '').strip()
        skills = data.get('Skills', '').strip()
        qualifications = data.get('Qualifications', '').strip()

        # Validate required fields
        if not all([job_title, location, years_exp]):
            logger_function(filename,"Insufficient structured data provided.",1)
            return Response(
                {'error': 'Please provide all of "job_title", "location", and "years_exp".'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Construct job description from structured fields
        job_desc = (
            f"Job Title: {job_title}; "
            f"Location: {location}; "
            f"Years Exp: {years_exp}; "
            f"Skills: {skills}; "
            f"Qualifications: {qualifications}"
        )
        logger_function(filename,"Constructed job description from structured fields.",1)

        # Check if FAISS index is available
        if INDEX.ntotal == 0:
            logger_function(filename,"FAISS index is empty.",1)
            return Response(
                {'error': 'FAISS index is empty. Please upload resumes first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        logger_function(filename,"Encoding job description and searching FAISS index.",1)
        q_vec = MODEL.encode([job_desc])
        q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

        # Search FAISS with cosine similarity (inner product)
        scores, indices = INDEX.search(q_vec.astype('float32'), k=10)

        # Load resume ID mapping
        try:
            id_list = np.load(ID_PATH, allow_pickle=True).tolist()
            logger_function(filename,"Resume ID mapping loaded.",1)
        except  :
            logger_function(filename,f"ID mapping file not found at {ID_PATH}.",1)
            return Response(
                {'error': f'ID mapping file not found at {ID_PATH}.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        results = []
        with connections['default'].cursor() as cursor:
        # with connections.cursor() as cursor:
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(id_list):
                    continue

                resume_id = id_list[idx]
                cursor.execute("SELECT 1 FROM JobInquiry WHERE id = %s", [resume_id])
                if cursor.fetchone():
                    results.append({
                        'resume_id': resume_id,
                        'score': round(float(score),4 )
                    })
                    logger_function(filename,f"Match found: Resume ID {resume_id} with score {float(score)}",1)

        logger_function(filename,f"Total matches found: {len(results)}",1)
        return Response({'matches': results})


class ResumeKeyPointsAPIView(APIView):
    def get(self, request, resume_id):
        logger_function(filename,f"Received request to extract key points for resume ID: {resume_id}",1)
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
                logger_function(filename,f"No resume found for ID: {resume_id}",1)
                return Response({'error': 'Resume not found.'}, status=status.HTTP_404_NOT_FOUND)

            resume_text = row.resume_text
            logger_function(filename,"Extracting insights from resume text.",1)
            insights = extract_resume_info(resume_text)

            return Response(insights, status=status.HTTP_200_OK)

        except Exception as e:
            logger_function(filename,f"Error extracting resume key points: {str(e)}",1)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            cursor.close()
            conn.close()
            logger_function(filename, "Database connection closed.",1)

