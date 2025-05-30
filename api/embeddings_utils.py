import os
import numpy as np
import faiss
from django.conf import settings
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from .logger_function import logger_function

filename = os.path.basename(__file__)[:-3]

# Ensure embedding directory exists
os.makedirs(settings.EMBED_DIR, exist_ok=True)

# File paths
EMBED_FILE = os.path.join(settings.EMBED_DIR, 'resume_embeddings.npy')
INDEX_FILE = os.path.join(settings.EMBED_DIR, 'faiss_index_file.index')
ID_PATH = os.path.join(settings.EMBED_DIR, 'resume_ids.npy')

# Model for embeddings
MODEL = SentenceTransformer('all-mpnet-base-v2')
DIM = MODEL.get_sentence_embedding_dimension()

# Load or initialize FAISS index with cosine similarity
try:
    if os.path.exists(INDEX_FILE):
        INDEX = faiss.read_index(INDEX_FILE)
        logger_function(filename, "FAISS index loaded.", 1)
    else:
        INDEX = faiss.IndexFlatIP(DIM)  # Cosine similarity (requires normalized vectors)
        logger_function(filename, "New FAISS index created.", 1)
except Exception as e:
    logger_function(filename, f"Error loading or creating FAISS index: {str(e)}", 2)

# Load existing embeddings (optional, not used directly in FAISS)
try:
    if os.path.exists(EMBED_FILE):
        EMBEDS = np.load(EMBED_FILE)
    else:
        EMBEDS = np.zeros((0, DIM), dtype='float32')
except Exception as e:
    logger_function(filename, f"Error loading embedding file: {str(e)}", 2)


def add_resume_to_index(resume_text, resume_id):
    try:
        logger_function(filename, f"Embedding new resume: {resume_id}", 1)

        # Encode and normalize embedding
        embedding = MODEL.encode([resume_text], convert_to_numpy=True)
        embedding = normalize(embedding, norm='l2')

        # Add to FAISS index
        INDEX.add(embedding.astype('float32'))
        faiss.write_index(INDEX, INDEX_FILE)
        logger_function(filename, "Normalized embedding added to FAISS index.", 1)

        # Save/update embedding array (optional)
        if os.path.exists(EMBED_FILE):
            embeddings = np.load(EMBED_FILE)
            embeddings = np.vstack([embeddings, embedding])
            logger_function(filename, "Existing embedding file loaded and updated.", 1)
        else:
            embeddings = embedding
            logger_function(filename, "New embedding array created.", 1)
        np.save(EMBED_FILE, embeddings)
        logger_function(filename, "Embedding array saved.", 1)

        # Save/update resume ID list
        if os.path.exists(ID_PATH):
            ids = list(np.load(ID_PATH, allow_pickle=True))
            logger_function(filename, "Existing ID list loaded.", 1)
        else:
            ids = []
            logger_function(filename, "New ID list created.", 1)
        ids.append(resume_id)
        np.save(ID_PATH, np.array(ids))
        logger_function(filename, "Resume ID list updated and saved.", 1)

        logger_function(filename, f"Resume {resume_id} added successfully. Index size: {INDEX.ntotal}", 1)

    except Exception as e:
        logger_function(filename, f"Error in add_resume_to_index: {str(e)}", 2)
