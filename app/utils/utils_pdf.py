import fitz  # PyMuPDF
from PIL import Image
import io

def extract_first_page_image(pdf_path: str) -> Image.Image:
    """
    Extrait la première page d’un fichier PDF en tant qu’image PIL.

    Args:
        pdf_path (str): Chemin du fichier PDF.

    Returns:
        Image.Image: Première page du PDF rendue en image.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    return image
