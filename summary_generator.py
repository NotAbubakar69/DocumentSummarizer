import time
import torch
import gc
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)

class SummaryGenerator:
    """Generates summaries using open source models with memory optimization"""
    def __init__(self, model_name='sshleifer/distilbart-cnn-6-6'):
        print(f"Loading summarization model: {model_name}")
        self.model_name = model_name
        self.device = "cpu"  # Force CPU usage
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        
        # Set environment variables for memory optimization
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        try:
            self._load_model()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to smaller model...")
            self.model_name = 'sshleifer/distilbart-cnn-6-6'  # Smallest BART model
            self._load_model()
    
    def _load_model(self):
        """Load model with memory optimization"""
        try:
            # Clear any existing models
            self.cleanup()
            
            # Use pipeline for better memory management
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=-1,  # Force CPU (-1 means CPU)
                model_kwargs={
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True
                }
            )
            
            print(f"Model loaded successfully: {self.model_name}")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            # Try even simpler approach
            try:
                print("Trying basic pipeline initialization...")
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=-1
                )
                print("Basic pipeline loaded successfully")
            except Exception as e2:
                print(f"Basic pipeline also failed: {e2}")
                raise e2
    
    def generate_summary_sync(self, text, max_length=150, min_length=30):
        """Generate summary using the loaded model with memory optimization"""
        print("Generating summary...")
        start_time = time.time()
        
        try:
            # Limit input text length to avoid memory issues
            max_input_length = 800  # Reduced from 1000
            words = text.split()
            
            if len(words) > max_input_length:
                print(f"Text too long ({len(words)} words), truncating to {max_input_length} words")
                text = ' '.join(words[:max_input_length])
            
            # Generate summary with conservative settings
            result = self.summarizer(
                text,
                max_length=min(max_length, 130),  # Reduced max length
                min_length=min(min_length, 20),   # Reduced min length
                do_sample=False,
                num_beams=2,  # Reduced from 4 to save memory
                early_stopping=True,
                truncation=True
            )
            
            summary = result[0]['summary_text']
            
            # Clean up
            gc.collect()
            
            end_time = time.time()
            print(f"Summary generation took {end_time - start_time:.2f} seconds")
            return summary
            
        except Exception as e:
            print(f"Error during summary generation: {str(e)}")
            
            # Fallback: return truncated original text
            words = text.split()
            if len(words) > 100:
                fallback_summary = ' '.join(words[:100]) + "..."
                return f"Summary generation failed. Here's a truncated version: {fallback_summary}"
            else:
                return f"Summary generation failed. Original text: {text[:500]}..."
    
    def cleanup(self):
        """Clean up models from memory"""
        if self.summarizer is not None:
            del self.summarizer
            self.summarizer = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
