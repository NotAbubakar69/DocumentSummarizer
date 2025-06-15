import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import gc
import os

class EmbeddingEngine:
    """Handles embedding generation and storage using open source models"""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self.index = None
        
        # Force CPU usage for memory-constrained systems
        self.device = "cpu"  # Always use CPU to avoid memory issues
        
        # Set environment variables to limit memory usage
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        try:
            self._load_model()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to smaller model...")
            self.model_name = 'all-MiniLM-L6-v2'  # Smallest model
            self._load_model()
    
    def _load_model(self):
        """Load the embedding model with memory optimization"""
        try:
            # Clear any existing model from memory
            if self.model is not None:
                del self.model
                gc.collect()
            
            # Load model with CPU-only configuration
            self.model = SentenceTransformer(
                self.model_name,
                device='cpu'  # Force CPU usage
            )
            
            # Get dimension
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Dimension: {self.dimension}")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e
        
    def create_embeddings_sync(self, chunks):
        """Create embeddings for chunks using sentence transformers"""
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 8  # Reduced batch size
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # Create embeddings for this batch
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=False,  # Disable progress bar for batches
                    device=self.device,
                    batch_size=4,  # Even smaller internal batch size
                    normalize_embeddings=True  # Normalize to save memory
                )
                
                all_embeddings.extend(batch_embeddings)
                
                # Force garbage collection after each batch
                gc.collect()
            
            embeddings = np.array(all_embeddings)
            print(f"Created {len(embeddings)} embeddings successfully")
            return embeddings
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise e
    
    def build_faiss_index(self, embeddings):
        """Build FAISS index for similarity search"""
        print("Building FAISS index...")
        try:
            # Use IndexFlatIP (inner product) which is more memory efficient
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Ensure embeddings are normalized and in correct format
            embeddings_normalized = embeddings.astype('float32')
            faiss.normalize_L2(embeddings_normalized)
            
            self.index.add(embeddings_normalized)
            print(f"FAISS index built with {self.index.ntotal} vectors")
            
            # Clean up embeddings from memory
            del embeddings_normalized
            gc.collect()
            
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            raise e
    
    def search(self, query, k=5):
        """Search for similar chunks"""
        if self.index is None:
            raise ValueError("Index not built. Call build_faiss_index first.")
        
        try:
            # Create embedding for query
            query_embedding = self.model.encode(
                [query], 
                device=self.device,
                normalize_embeddings=True
            )
            
            # Normalize for inner product search
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.index.search(query_embedding, k)
            
            # Clean up
            del query_embedding
            gc.collect()
            
            return distances[0], indices[0]
            
        except Exception as e:
            print(f"Error during search: {e}")
            raise e
    
    def cleanup(self):
        """Clean up model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.index is not None:
            del self.index
            self.index = None
        gc.collect()
