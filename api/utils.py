import os
import uuid
import fitz 
import subprocess
from docx2pdf import convert as docx_to_pdf
import logging

logger = logging.getLogger(__name__)

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
            logger.error(f"OCR failed: {result.stderr}")
            raise Exception(result.stderr)
        return result.stdout.strip()
    except Exception as e:
        raise Exception(f"Failed to run OCR: {e}")
