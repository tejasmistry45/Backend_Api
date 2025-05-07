import time
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd 
from .llm_utils import extract_resume_info
from rest_framework import status

import os
from django.conf import settings
from django.shortcuts import render
from .models import Resume
from PyPDF2 import PdfReader
from docx import Document
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile



model = SentenceTransformer('all-mpnet-base-v2')
index = faiss.read_index('embeddings/faiss_index.index')
 
df = pd.read_csv('data/Final_Resume.csv')
resume_ids = df['resume_id'].astype(str).tolist()
resume_texts = df['resume_text'].tolist()

@api_view(['POST'])
def find_matches(request):
    data = request.data

    # Example: Add two numbers sent from Angular
    # a = data.get('a', 0)
    # b = data.get('b', 0)
    # result = a + b
    job_title = data.get('job_title', '')
    location = data.get('location', '')
    years_exp = data.get('years_exp', '')
    skills = data.get('Skills', '')
    qualifications = data.get('Qualifications', '')

    job_text = f"{job_title} | {location} | {years_exp} yrs exp | Skills: {skills} | Qualifications: {qualifications}"
    print("Job text:", job_text)

    job_embedding = model.encode([job_text])

    # Search for the top 5 matching resumes using FAISS
    D, I = index.search(np.array(job_embedding).astype(np.float32), k=5)

    # Get the top matching resumes
    matches = []
    for i, idx in enumerate(I[0]):
            match = {
                "resume_id": resume_ids[idx],
                "score": D[0][i],
                # "name": df.iloc[idx]["name"],  # Assuming your CSV has a 'name' column for the resume
            }
            matches.append(match)
    # Sort matches by score in descending order (highest score first)
    matches.sort(key=lambda x: x['score'], reverse=True)


    return Response({'result': matches})

@api_view(['GET'])
def get_key_points(request,resume_id):
    try:
        # Ensure both resume_id and list items are strings
        resume_id_str = str(resume_id)
        resume_ids_str = [str(rid) for rid in resume_ids]
 
        if resume_id_str not in resume_ids_str:
            return "Resume ID not found.", 404
 
        resume_index = resume_ids_str.index(resume_id_str)
        resume_text = resume_texts[resume_index]
       
        start_time = time.time()
        llm_response = extract_resume_info(resume_text)
        end_time = time.time()
        print(f"LLM Response completed in {end_time - start_time:.2f} seconds.")
        print("llm response : ",llm_response)
        # print("strip : ",llm_response.strip())
        return Response(
        data=llm_response,
        status=status.HTTP_200_OK,
        # Optional; Django uses renderer classes anyway
    )
 
    except Exception as e:
        import traceback
        return f"Failed to extract insights: {str(e)}", 500
 

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(path):
    try:
        with open(path, 'rb') as f:
            reader = PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        return f"[Error extracting PDF text: {str(e)}]"

def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"[Error extracting DOCX text: {str(e)}]"

def upload_resume(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('resume_file')
        original_filename = uploaded_file.name
        file_storage_path = os.path.join('resumes', original_filename)
        full_path = os.path.join(settings.MEDIA_ROOT, file_storage_path)

        # Check if file already exists
        if Resume.objects.filter(file_path=file_storage_path).exists():
            return render(request, 'upload.html', {
                'error': 'This resume is already uploaded.'
            })

        # Save file manually
        default_storage.save(file_storage_path, ContentFile(uploaded_file.read()))

        # Extract text
        text = ""
        if original_filename.endswith('.pdf'):
            text = extract_text_from_pdf(full_path)
        elif original_filename.endswith('.docx'):
            text = extract_text_from_docx(full_path)
        else:
            text = "[Unsupported file format. Only .pdf and .docx are allowed.]"

        # Save to DB
        resume = Resume(file_path=file_storage_path, resume_text=text)
        resume.save()

        return render(request, 'upload.html', {
            'success': True,
            'filename': original_filename
        })

    return render(request, 'upload.html')
