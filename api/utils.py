from PyPDF2 import PdfReader
from docx import Document

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
