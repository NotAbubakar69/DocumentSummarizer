import re
from pathlib import Path
import PyPDF2
import markdown

class DocumentParser:
    """Handles document ingestion and chunking"""
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def parse_document(self, file_path):
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            return self._parse_pdf(file_path)
        elif file_ext == '.txt':
            return self._parse_txt(file_path)
        elif file_ext == '.md':
            return self._parse_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _parse_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _parse_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _parse_markdown(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
            html = markdown.markdown(md_content)
            text = re.sub('<[^<]+?>', '', html)
            return text
    
    def chunk_document(self, text):
        if not text:
            return []
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_words = ' '.join(current_chunk[-self.chunk_overlap:]) if self.chunk_overlap > 0 else ''
                current_chunk = []
                if overlap_words:
                    current_chunk = overlap_words.split()
                current_size = len(current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
