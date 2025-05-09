import os
import numpy as np
import faiss
from django.conf import settings
from sentence_transformers import SentenceTransformer
from .models import Resume

# Initialize global
EMBED_FILE = os.path.join(settings.EMBED_DIR, 'resume_embeddings.npy')
INDEX_FILE = os.path.join(settings.EMBED_DIR, 'faiss_index.index')
ID_PATH = os.path.join(settings.EMBED_DIR, 'resume_ids.npy')

MODEL = SentenceTransformer('all-mpnet-base-v2')

# Ensure directories
os.makedirs(settings.EMBED_DIR, exist_ok=True)

# Load or initialize embeddings and index
if os.path.exists(EMBED_FILE):
    EMBEDS = np.load(EMBED_FILE)
else:
    EMBEDS = np.zeros((0, MODEL.get_sentence_embedding_dimension()), dtype=np.float32)

if os.path.exists(INDEX_FILE):
    INDEX = faiss.read_index(INDEX_FILE)
else:
    INDEX = faiss.IndexFlatL2(MODEL.get_sentence_embedding_dimension())

# # Add a new resume embedding and persist
# ID_PATH = "resume_ids.npy"



def add_resume_to_index(resume_obj):
    print("Embedding new resume...")
    embedding = MODEL.encode([resume_obj.resume_text])[0].astype('float32')

    # Load or initialize FAISS index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print("FAISS index loaded from file.")
    else:
        print("Creating new FAISS index...")
        index = faiss.IndexFlatL2(embedding.shape[0])

    # Add new embedding to the index
    index.add(np.array([embedding]))
    # faiss.write_index(index, INDEX_FILE)
    faiss.write_index(INDEX, 'faiss_index_file.index')

    print(f"FAISS index updated. Total vectors: {index.ntotal}")

    # Update and save embedding array (optional, used for analysis or backup)
    if os.path.exists(EMBED_FILE):
        embeddings = np.load(EMBED_FILE)
        embeddings = np.vstack([embeddings, embedding])
    else:
        embeddings = np.array([embedding])
    np.save(EMBED_FILE, embeddings)
    print("Resume embeddings saved/updated.")

    # Update and save resume ID list
    if os.path.exists(ID_PATH):
        ids = list(np.load(ID_PATH))
    else:
        ids = []
    ids.append(resume_obj.id)
    np.save(ID_PATH, ids)
    print(f"Resume ID {resume_obj.id} added. Total IDs stored: {len(ids)}")
