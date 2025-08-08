import PyPDF2
from docx import Document
import os
import re
from typing import List

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 400  # Smaller chunks for Railway
        self.chunk_overlap = 50

    def process_document(self, file_path: str) -> List[str]:
        try:
            if file_path.endswith('.pdf'):
                text = self._extract_pdf_text(file_path)
            elif file_path.endswith(('.docx', '.doc')):
                text = self._extract_docx_text(file_path)
            else:
                with open(file_path, 'r') as f:
                    text = f.read()
            
            return self._chunk_text(text)
        except Exception as e:
            print(f"Error: {str(e)}")
            return []

    def _extract_pdf_text(self, path: str) -> str:
        text = ""
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages[:5]:  # Only first 5 pages
                text += page.extract_text() + "\n"
        return text

    def _extract_docx_text(self, path: str) -> str:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    def _chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent + " "
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent + " "
        
        if current.strip():
            chunks.append(current.strip())
            
        return chunks[:200]  # Limit total chunks