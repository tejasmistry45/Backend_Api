import os
import numpy as np
import faiss
from django.conf import settings
from sentence_transformers import SentenceTransformer

# Ensure the embedding directory exists
os.makedirs(settings.EMBED_DIR, exist_ok=True)

# File paths
EMBED_FILE = os.path.join(settings.EMBED_DIR, 'resume_embeddings.npy')
INDEX_FILE = os.path.join(settings.EMBED_DIR, 'faiss_index_file.index')
ID_PATH = os.path.join(settings.EMBED_DIR, 'resume_ids.npy')

# Model
MODEL = SentenceTransformer('all-mpnet-base-v2')

# Initialize or load index
if os.path.exists(INDEX_FILE):
    INDEX = faiss.read_index(INDEX_FILE)
else:
    INDEX = faiss.IndexFlatL2(MODEL.get_sentence_embedding_dimension())

# Load existing embeddings or start with empty
if os.path.exists(EMBED_FILE):
    EMBEDS = np.load(EMBED_FILE)
else:
    EMBEDS = np.zeros((0, MODEL.get_sentence_embedding_dimension()), dtype='float32')


def add_resume_to_index(resume_text,resume_id):
    print("Embedding new resume...")
 
    embedding = MODEL.encode([resume_text])[0].astype('float32')
 
    # Add to global INDEX
    INDEX.add(np.array([embedding]))
    faiss.write_index(INDEX, INDEX_FILE)
 
    # Save/update embedding array
    if os.path.exists(EMBED_FILE):
        embeddings = np.load(EMBED_FILE)
        embeddings = np.vstack([embeddings, embedding])
    else:
        embeddings = np.array([embedding])
    np.save(EMBED_FILE, embeddings)
 
    # Save/update resume IDs
    if os.path.exists(ID_PATH):
        ids = list(np.load(ID_PATH, allow_pickle=True))
    else:
        ids = []
    ids.append(resume_id)
    np.save(ID_PATH, np.array(ids))
 
    print(f"Resume ID {resume_id} added. Index size: {INDEX.ntotal}")