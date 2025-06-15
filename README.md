# RAG Document Summarizer with Flask GUI

A modern document summarization system that combines Retrieval-Augmented Generation (RAG) with open source models. This application features a clean web interface built with Flask and uses FAISS for efficient vector similarity search.

## Features

- **Open Source Models**: Uses Hugging Face transformers and sentence-transformers
- **No API Keys Required**: Runs completely offline with local models
- **Web Interface**: Clean, responsive Flask-based GUI
- **Multiple Formats**: Supports PDF, TXT, and Markdown documents
- **GPU Acceleration**: Automatic CUDA detection for faster processing
- **Configurable Parameters**: Adjust chunk size, overlap, retrieval count, and models
- **Real-time Processing**: Live feedback during document processing
- **Detailed Results**: Shows processing time, similarity scores, and retrieved context

## Models Used

### Embedding Models
- **all-mpnet-base-v2** (default): Balanced performance and quality
- **all-MiniLM-L6-v2**: Fast and lightweight
- **all-MiniLM-L12-v2**: Better quality, slightly slower
- **paraphrase-multilingual-MiniLM-L12-v2**: Multilingual support
- **sentence-transformers/all-distilroberta-v1**: RoBERTa-based embeddings

### Language Models
- **facebook/bart-large-cnn** (default): Excellent for news summarization
- **google/pegasus-xsum**: Optimized for abstractive summarization
- **microsoft/DialoGPT-medium**: Conversational AI model
- **google/flan-t5-base**: Instruction-tuned T5 model
- **google/flan-t5-large**: Larger T5 model for better quality
- **sshleifer/distilbart-cnn-12-6**: Faster, distilled BART model

## Installation

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd rag-document-summarizer
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Optional: Set up environment variables**
   \`\`\`bash
   cp .env.example .env
   # Edit .env if you want to customize default models
   \`\`\`

4. **Run the application**
   \`\`\`bash
   python app.py
   \`\`\`

5. **Open your browser**
   Navigate to \`http://localhost:5000\`

## Usage

### Web Interface

1. **Upload Document**: Drag and drop or select a PDF, TXT, or MD file
2. **Configure Settings**: Adjust chunk size, overlap, and model selection
3. **Generate Summary**: Click the "Generate Summary" button
4. **Review Results**: View the generated summary and retrieved context chunks

### Configuration Options

- **Chunk Size**: Number of words per chunk (50-2000)
- **Chunk Overlap**: Overlapping words between chunks
- **Top-K Retrieval**: Number of most relevant chunks to use
- **Embedding Model**: Choose from various sentence-transformer models
- **Summary Model**: Select from Hugging Face summarization models

## API Endpoints

### Health Check
\`\`\`
GET /health
\`\`\`

### Upload Document
\`\`\`
POST /upload
Content-Type: multipart/form-data
\`\`\`

### Generate Summary
\`\`\`
POST /summarize
Content-Type: application/json

{
  "file_path": "path/to/uploaded/file",
  "chunk_size": 500,
  "chunk_overlap": 100,
  "top_k": 5,
  "embedding_model": "text-embedding-3-small",
  "summary_model": "gpt-4o-mini"
}
\`\`\`

## Project Structure

\`\`\`
rag-document-summarizer/
├── document_parser.py         # Document parsing and chunking
├── embedding_engine.py        # OpenAI embeddings + FAISS
├── summary_generator.py       # OpenAI GPT summarization
├── rag_summarizer.py         # Main RAG pipeline
├── templates/
│   └── index.html            # Web interface
├── uploads/                  # Uploaded documents
├── results/                  # Generated summaries
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
\`\`\`

## Key Changes from Original

1. **Models**: Switched from BART/SentenceTransformers to OpenAI GPT/Embeddings
2. **Interface**: Added modern Flask web GUI instead of CLI
3. **Vector Store**: Kept FAISS but integrated with OpenAI embeddings
4. **API Integration**: Uses OpenAI API with fallback handling
5. **Real-time Feedback**: Live processing updates in the web interface

## System Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended for larger models)
- GPU with 4GB+ VRAM (optional, but recommended for faster processing)
- Modern web browser

## Performance Notes

- **CPU Mode**: All models can run on CPU, but will be slower
- **GPU Mode**: Automatic CUDA detection for faster processing
- **Model Size**: Larger models provide better quality but require more resources
- **Memory Usage**: Models are loaded on-demand and cached in memory

## Troubleshooting

1. **Out of Memory**: Try smaller models (MiniLM, DistilBART) or reduce chunk size
2. **Slow Processing**: Enable GPU acceleration or use faster models
3. **Model Download**: First run will download models (may take time)
4. **Upload Failures**: Check file size (max 16MB) and format

## Offline Operation

This application runs completely offline once models are downloaded:
- No internet connection required after initial setup
- No API keys needed
- All processing happens locally
- Your documents never leave your machine

## License

MIT License - see LICENSE file for details
