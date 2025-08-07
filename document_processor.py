import PyPDF2
from docx import Document
import os
import re
from typing import List

class DocumentProcessor:
    def __init__(self):
        # Optimized for 4GB memory
        self.chunk_size = 600  # Reduced from 1000
        self.chunk_overlap = 100
        self.min_section_length = 30

    def process_document(self, file_path: str) -> List[str]:
        """Process document with memory-efficient chunking"""
        try:
            text = self._extract_text(file_path)
            return self._split_into_chunks(text)
        except Exception as e:
            print(f"Document processing error: {str(e)}")
            return []

    def _extract_text(self, file_path: str) -> str:
        """Unified text extraction"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self._extract_pdf_text(file_path)
        elif ext in ['.docx', '.doc']:
            return self._extract_docx_text(file_path)
        elif ext == '.txt':
            return self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_pdf_text(self, file_path: str) -> str:
        """Memory-efficient PDF extraction"""
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages[:20]:  # Limit to first 20 pages
                text += page.extract_text() + "\n"
        return text

    def _extract_docx_text(self, file_path: str) -> str:
        """DOCX extraction with section detection"""
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    def _extract_txt_text(self, file_path: str) -> str:
        """TXT with encoding fallback"""
        for encoding in ['utf-8', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError("Failed to decode text file")

    def _split_into_chunks(self, text: str) -> List[str]:
        """Memory-optimized chunking"""
        # First split by major sections
        sections = re.split(r'(?=\nSECTION [A-Z]\)|\n\d+\.\s)', text)
        chunks = []
        
        for section in sections:
            if not section.strip() or len(section) < self.min_section_length:
                continue
                
            # Split section into sentences
            sentences = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                
        return chunks[:500]  # Safety limit