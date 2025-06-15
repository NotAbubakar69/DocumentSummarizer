import os
import time
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import json
import torch
import gc

from rag_summarizer import RAGSummarizer

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported. Please upload PDF, TXT, or MD files.'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get document stats
        try:
            rag_summarizer = RAGSummarizer()
            stats = rag_summarizer.get_document_stats(file_path)
            # Clean up immediately after getting stats
            rag_summarizer.cleanup()
            del rag_summarizer
            gc.collect()
        except Exception as e:
            stats = {"error": str(e)}
        
        return jsonify({
            'success': True,
            'filename': filename,
            'file_path': file_path,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    rag_summarizer = None
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['file_path', 'chunk_size', 'chunk_overlap', 'top_k', 'embedding_model', 'summary_model']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate file exists
        if not os.path.exists(data['file_path']):
            return jsonify({'error': 'File not found'}), 404
        
        # Force garbage collection before starting
        gc.collect()
        
        # Initialize RAG summarizer with custom parameters
        print("Initializing RAG summarizer...")
        rag_summarizer = RAGSummarizer(
            embedding_model=data['embedding_model'],
            summary_model=data['summary_model'],
            chunk_size=int(data['chunk_size']),
            chunk_overlap=int(data['chunk_overlap'])
        )
        
        # Process document
        print("Processing document...")
        result = rag_summarizer.process_document(
            data['file_path'],
            top_k=int(data['top_k'])
        )
        
        # Save results
        result_filename = f"summary_{int(time.time())}.json"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        result['result_file'] = result_filename
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        error_msg = f'Summarization failed: {str(e)}'
        print(error_msg)
        return jsonify({'error': error_msg}), 500
    
    finally:
        # Always clean up, even if there's an error
        if rag_summarizer:
            try:
                rag_summarizer.cleanup()
                del rag_summarizer
            except:
                pass
        gc.collect()

@app.route('/results/<filename>')
def get_result(filename):
    try:
        result_path = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(result_path):
            return jsonify({'error': 'Result file not found'}), 404
        
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Failed to load result: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check system resources
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return jsonify({
            'status': 'healthy',
            'message': 'RAG Summarizer is running with open source models',
            'device': device,
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'cuda_available': torch.cuda.is_available(),
            'memory_optimized': True
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Check system capabilities
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting RAG Summarizer with device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("Running in CPU-only mode for better memory compatibility")
    
    # Set memory optimization environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    app.run(debug=True, host='0.0.0.0', port=5000)
