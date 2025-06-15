import os
import time
import gc
from datetime import datetime
from pathlib import Path

from document_parser import DocumentParser
from embedding_engine import EmbeddingEngine
from summary_generator import SummaryGenerator

class RAGSummarizer:
    """Main RAG pipeline for document summarization with memory optimization"""
    
    def __init__(
        self, 
        embedding_model='all-MiniLM-L6-v2',  # Smallest model by default
        summary_model='sshleifer/distilbart-cnn-6-6',  # Smallest BART model
        chunk_size=300,  # Reduced chunk size
        chunk_overlap=50   # Reduced overlap
    ):
        self.parser = DocumentParser(chunk_size, chunk_overlap)
        self.embedding_engine = None
        self.summary_generator = None
        self.chunks = []
        self.embeddings = None
        
        # Store model names for lazy loading
        self.embedding_model_name = embedding_model
        self.summary_model_name = summary_model
        
    def _ensure_embedding_engine(self):
        """Lazy load embedding engine"""
        if self.embedding_engine is None:
            self.embedding_engine = EmbeddingEngine(self.embedding_model_name)
    
    def _ensure_summary_generator(self):
        """Lazy load summary generator"""
        if self.summary_generator is None:
            self.summary_generator = SummaryGenerator(self.summary_model_name)
        
    def process_document(self, file_path, top_k=3):  # Reduced top_k
        """Process document through the RAG pipeline with memory optimization"""
        start_time = time.time()
        
        try:
            # Parse document
            print(f"Processing document: {file_path}")
            document_text = self.parser.parse_document(file_path)
            
            if not document_text.strip():
                raise ValueError("Document appears to be empty or could not be parsed")
            
            # Limit document size to avoid memory issues
            words = document_text.split()
            max_words = 5000  # Limit document size
            if len(words) > max_words:
                print(f"Document too large ({len(words)} words), truncating to {max_words} words")
                document_text = ' '.join(words[:max_words])
            
            # Chunk document
            self.chunks = self.parser.chunk_document(document_text)
            print(f"Document split into {len(self.chunks)} chunks")
            
            if not self.chunks:
                raise ValueError("No chunks created from document")
            
            # Limit number of chunks to process
            max_chunks = 20
            if len(self.chunks) > max_chunks:
                print(f"Too many chunks ({len(self.chunks)}), processing first {max_chunks}")
                self.chunks = self.chunks[:max_chunks]
            
            # Load embedding engine and create embeddings
            print("Loading embedding model...")
            self._ensure_embedding_engine()
            
            self.embeddings = self.embedding_engine.create_embeddings_sync(self.chunks)
            self.embedding_engine.build_faiss_index(self.embeddings)
            
            # Retrieve relevant chunks for summarization
            query = "Provide a comprehensive summary of this document"
            distances, indices = self.embedding_engine.search(query, k=min(top_k, len(self.chunks)))
            
            # Prepare context for summary generation
            retrieved_chunks = [self.chunks[i] for i in indices]
            context = "\n\n".join(retrieved_chunks)
            
            # Clean up embedding engine to free memory
            self.embedding_engine.cleanup()
            gc.collect()
            
            # Load summary generator and generate summary
            print("Loading summarization model...")
            self._ensure_summary_generator()
            
            summary = self.summary_generator.generate_summary_sync(context)
            
            # Clean up summary generator
            self.summary_generator.cleanup()
            gc.collect()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Prepare results
            result = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "num_chunks": len(self.chunks),
                "retrieved_chunks": retrieved_chunks,
                "summary": summary,
                "processing_time": total_time,
                "similarity_scores": distances.tolist(),
                "timestamp": datetime.now().isoformat(),
                "models_used": {
                    "embedding": self.embedding_model_name,
                    "summary": self.summary_model_name
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            # Clean up on error
            self.cleanup()
            raise e
    
    def get_document_stats(self, file_path):
        """Get basic statistics about the document"""
        try:
            document_text = self.parser.parse_document(file_path)
            chunks = self.parser.chunk_document(document_text)
            
            return {
                "word_count": len(document_text.split()),
                "character_count": len(document_text),
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(chunk.split()) for chunk in chunks) / len(chunks) if chunks else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up all models from memory"""
        if self.embedding_engine:
            self.embedding_engine.cleanup()
            self.embedding_engine = None
        if self.summary_generator:
            self.summary_generator.cleanup()
            self.summary_generator = None
        gc.collect()
