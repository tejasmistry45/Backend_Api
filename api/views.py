import os
from django.conf import settings
import faiss
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Resume
from .serializers import ResumeSerializer
from .utils import extract_text_from_pdf, extract_text_from_docx
from .embeddings_utils import add_resume_to_index, MODEL, INDEX, EMBEDS
import numpy as np

class ProcessResumePathAPIView(APIView):
    """
    POST: { "path": "resumes/filename.pdf" }
    """
    def post(self, request):
        # Get the relative file path from the request
        rel_path = request.data.get('path')
        if not rel_path:
            return Response({'error': 'No path provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # Get the full file path
        full_path = os.path.join(settings.MEDIA_ROOT, rel_path)
        if not os.path.exists(full_path):
            return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

        # Duplicate check in the database
        if Resume.objects.filter(file_path=rel_path).exists():
            return Response({'error': 'Already processed.'}, status=status.HTTP_409_CONFLICT)

        # Extract text from PDF or DOCX
        if rel_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(full_path)
        elif rel_path.lower().endswith('.docx'):
            text = extract_text_from_docx(full_path)
        else:
            return Response({'error': 'Unsupported format.'}, status=status.HTTP_400_BAD_REQUEST)

        # Save resume to database
        resume = Resume.objects.create(file_path=rel_path, resume_text=text)

        # Extract the resume's embedding
        embedding = MODEL.encode([text]).astype('float32')

        # Add the resume's embedding to the FAISS index
        INDEX.add(embedding)

        # Save the FAISS index to file after adding the new vector
        faiss.write_index(INDEX, 'faiss_index_file.index')

        # Load existing resume IDs
        try:
            resume_ids = np.load("resume_ids.npy", allow_pickle=True).tolist()
        except FileNotFoundError:
            resume_ids = []

        # Append the new resume ID
        resume_ids.append(resume.id)

        # Save updated resume IDs back to the numpy file
        np.save("resume_ids.npy", np.array(resume_ids))

        # Return a success response
        return Response({'message': 'Processed successfully', 'resume_id': resume.id}, status=status.HTTP_201_CREATED)
    

class FindMatchesAPIView(APIView):
    def post(self, request):
        job_desc = request.data.get('job_description', '')
        if not job_desc:
            return Response({'error': 'No job_description.'}, status=status.HTTP_400_BAD_REQUEST)

        # Encode job description into vector
        q_vec = MODEL.encode([job_desc]).astype('float32')

        # Perform search in FAISS index to get top 5 matches
        D, I = INDEX.search(q_vec, k=5)

        print(f"Number of vectors in FAISS index: {INDEX.ntotal}")

        # Load the ID mapping from resume_ids.npy
        try:
            id_list = np.load("resume_ids.npy", allow_pickle=True).tolist()
        except FileNotFoundError:
            return Response({'error': 'resume_ids.npy file not found.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        results = []
        for score, idx in zip(D[0], I[0]):
            idx = int(idx)
            if idx == -1 or idx >= len(id_list):
                continue

            db_id = int(id_list[idx])  # Map FAISS position to real DB ID
            try:
                obj = Resume.objects.get(id=db_id)
                results.append({'resume_id': obj.id, 'score': float(score)})
            except Resume.DoesNotExist:
                print(f"Resume with id={db_id} not found in database.")
                continue

        print(f"Top matching resumes: {results}")
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
