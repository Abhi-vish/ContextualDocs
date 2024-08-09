import PyPDF2
from docx import Document

def extract_text_from_pdf(document):
    pdf_reader = PyPDF2.PdfReader(document)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(document):
    doc = Document(document)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(document):
    return document.read().decode('utf-8')
