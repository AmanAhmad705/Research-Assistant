import requests
import fitz  # PyMuPDF
import os

def download_pdf(url, save_path):
    """
    Downloads a PDF from a given URL and saves it to the specified path.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download PDF: {url}")
            return False
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return False


def extract_text_from_pdf(file_path):
    """
    Extracts all text from a PDF file using PyMuPDF.
    """
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""
