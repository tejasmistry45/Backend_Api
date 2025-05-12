import os
from django.conf import settings
import faiss
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Resume
from .serializers import ResumeSerializer
from .utils import extract_text_from_pdf, extract_text_from_docx
from .embeddings_utils import ID_PATH, add_resume_to_index, MODEL, INDEX, EMBEDS
import numpy as np


class ProcessResumePathAPIView(APIView):
    def post(self, request):
        rel_path = request.data.get('path')
        if not rel_path:
            return Response({'error': 'No path provided.'}, status=status.HTTP_400_BAD_REQUEST)

        full_path = os.path.join(settings.MEDIA_ROOT, rel_path)
        if not os.path.exists(full_path):
            return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

        if Resume.objects.filter(file_path=rel_path).exists():
            return Response({'error': 'Already processed.'}, status=status.HTTP_409_CONFLICT)

        if rel_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(full_path)
        elif rel_path.lower().endswith('.docx'):
            text = extract_text_from_docx(full_path)
        else:
            return Response({'error': 'Unsupported format.'}, status=status.HTTP_400_BAD_REQUEST)

        resume = Resume.objects.create(file_path=rel_path, resume_text=text)

        # Add to index using utility
        add_resume_to_index(resume)

        return Response({'message': 'Processed successfully', 'resume_id': resume.id}, status=status.HTTP_201_CREATED)


import os
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Resume
from .embeddings_utils import MODEL, INDEX, ID_PATH

class FindMatchesAPIView(APIView):
    def post(self, request):
        data = request.data

        # 1️⃣ Try existing format
        job_desc = data.get('job_description', '').strip()

        # 2️⃣ If not present, try structured format
        if not job_desc:
            job_title = data.get('job_title', '').strip()
            location = data.get('location', '').strip()
            years_exp = data.get('years_exp', '').strip()
            skills = data.get('Skills', '').strip()
            qualifications = data.get('Qualifications', '').strip()

            # Ensure required structured fields exist
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

        # 3️⃣ Make sure we have something to search on
        if not job_desc:
            return Response({'error': 'No job description provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # 4️⃣ Check that index is populated
        if INDEX.ntotal == 0:
            return Response(
                {'error': 'FAISS index is empty. Please upload resumes first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        # 5️⃣ Encode & search
        q_vec = MODEL.encode([job_desc]).astype('float32')
        D, I = INDEX.search(q_vec, k=5)

        # 6️⃣ Load the resume‐ID mapping
        try:
            id_list = np.load(ID_PATH, allow_pickle=True).tolist()
        except FileNotFoundError:
            return Response(
                {'error': f'ID mapping file not found at {ID_PATH}.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 7️⃣ Build results
        results = []
        for score, idx in zip(D[0], I[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(id_list):
                continue
            db_id = id_list[idx]
            try:
                resume = Resume.objects.get(id=db_id)
                results.append({'resume_id': resume.id, 'score': float(score)})
            except Resume.DoesNotExist:
                continue

        return Response({'matches': results})



class ResumeKeyPointsAPIView(APIView):
    def get(self, request, resume_id):
        from .llm_utils import extract_resume_info
        try:
            resume = Resume.objects.get(id=resume_id)
        except Resume.DoesNotExist:
            return Response({'error': 'Not found.'}, status=status.HTTP_404_NOT_FOUND)

        insights = extract_resume_info(resume.resume_text)
        return Response({'insights': insights})
