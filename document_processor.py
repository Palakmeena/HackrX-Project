import PyPDF2
from docx import Document
import os
from typing import List

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 200  # Smaller chunks for better matching
        self.chunk_overlap = 50
    
    def process_document(self, file_path: str) -> List[str]:
        """Process document and return text chunks"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                text = self._extract_pdf_text(file_path)
            elif file_extension in ['.docx', '.doc']:
                text = self._extract_docx_text(file_path)
            elif file_extension == '.txt':
                text = self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Split into chunks
            chunks = self._split_text(text)
            
            # Clean up temp file (but keep sample_policy.txt)
            if os.path.exists(file_path) and "sample_policy" not in file_path:
                os.remove(file_path)
            
            return chunks
        
        except Exception as e:
            print(f"Error processing document: {e}")
            return []
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from .txt file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ""
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from Word document"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
        return text
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into smaller, more focused chunks"""
        # Split by lines first, then combine
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        chunks = []
        
        current_chunk = ""
        for line in lines:
            if len(current_chunk + line) < self.chunk_size:
                current_chunk += line + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + " "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks