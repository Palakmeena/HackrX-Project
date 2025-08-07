import PyPDF2
from docx import Document
import os
import re
from typing import List

class DocumentProcessor:
    def __init__(self):
        # Increased chunk size for better policy clause retention
        self.chunk_size = 1000  # Increased from 200
        self.chunk_overlap = 200  # Increased from 50
        self.min_section_length = 50  # Minimum chars to consider a section

    def process_document(self, file_path: str) -> List[str]:
        """Process document and return structured text chunks"""
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
            
            # Clean and structure the text
            cleaned_text = self._clean_text(text)
            
            # Split into meaningful chunks
            chunks = self._split_into_sections(cleaned_text)
            
            # Clean up temp file (but keep sample_policy.txt)
            if os.path.exists(file_path) and "sample_policy" not in file_path:
                os.remove(file_path)
            
            return chunks
        
        except Exception as e:
            print(f"Error processing document: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text)
        # Standardize section headers
        text = re.sub(r'\nSECTION\s+([A-Z])\)', r'\nSECTION \1) ', text)
        return text.strip()

    def _split_into_sections(self, text: str) -> List[str]:
        """Split document into logical sections first, then into chunks"""
        # First split by major sections
        sections = re.split(r'(\nSECTION [A-Z]\)[^\n]*)', text)
        
        # Combine section headers with their content
        structured_sections = []
        current_section = ""
        
        for part in sections:
            if re.match(r'\nSECTION [A-Z]\)', part):
                if current_section:
                    structured_sections.append(current_section)
                current_section = part + " "
            else:
                current_section += part + " "
        
        if current_section:
            structured_sections.append(current_section)

        # Now split each section into sized chunks
        final_chunks = []
        for section in structured_sections:
            if len(section) < self.min_section_length:
                continue
                
            section_header = re.search(r'\nSECTION [A-Z]\)[^\n]*', section)
            header = section_header.group(0) if section_header else "Policy Content"
            
            # Split section into sentences first for better context
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', section)
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < self.chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        final_chunks.append(f"{header}\n{current_chunk.strip()}")
                    current_chunk = sentence + " "
            
            if current_chunk.strip():
                final_chunks.append(f"{header}\n{current_chunk.strip()}")

        return final_chunks

    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from .txt file with encoding handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading TXT: {e}")
                return ""

    def _extract_pdf_text(self, file_path: str) -> str:
        """Improved PDF text extraction with layout preservation"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Add section markers if detected
                        if "SECTION" in page_text:
                            text += "\n" + page_text
                        else:
                            text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from Word document with formatting hints"""
        text = ""
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                # Preserve heading styles as section markers
                if para.style.name.startswith('Heading'):
                    text += f"\nSECTION {para.style.name[-1].upper()}) {para.text}\n"
                else:
                    text += para.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
        return text