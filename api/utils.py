import os
import uuid
import fitz 
import subprocess
from docx2pdf import convert as docx_to_pdf
from .logger_function import logger_function
import re


filename=os.path.basename(__file__)[:-3]

def convert_docx_to_pdf(docx_path):
    pdf_path = docx_path.replace('.docx', '.pdf')
    try:
        docx_to_pdf(docx_path, pdf_path)
        return pdf_path
    except Exception as e:
        raise Exception(f"Error converting DOCX to PDF: {e}")

def convert_pdf_to_images(pdf_path, output_folder='media/ocr_images'):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=200)  # Increase DPI for better OCR quality
        unique_name = f"{uuid.uuid4()}_page_{i+1}.jpg"
        image_path = os.path.join(output_folder, unique_name)
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths

def run_llamaocr_on_image(image_path):
    try:
        result = subprocess.run(
            ['node', 'llamaocr-service/index.js', image_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger_function.error(f"OCR failed: {result.stderr}")
            raise Exception(result.stderr)
        return result.stdout.strip()
    except Exception as e:
        raise Exception(f"Failed to run OCR: {e}")

def clean_ocr_text(text):
    # Remove markdown headings like ###, ##, #
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove markdown bold/italic like **bold**, *italic*, etc.
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
    text = re.sub(r'__([^_]+)__', r'\1', text)      # Bold with underscores

    # Remove horizontal rules like --- or ***
    text = re.sub(r'^[-*]{3,}$', '', text, flags=re.MULTILINE)

    # Remove backticks used for code
    text = re.sub(r'`+', '', text)

    # Collapse multiple newlines into a single newline
    text = re.sub(r'\n{2,}', '\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text
