"""Mock document preparation module for RAG Sample (demo version)."""

import os
import requests
from pathlib import Path
from typing import List
from urllib.parse import urlparse
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_from_html(html_content: str, url: str) -> str:
    """Extract text content from HTML (simplified version)."""
    # Remove common HTML tags (basic implementation)
    import re
    
    # Remove script and style blocks
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_content)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def download_content(url: str) -> str:
    """Download content from a URL or local file."""
    try:
        # Handle local files for testing
        if url.startswith('file://') or (not url.startswith('http') and os.path.exists(url)):
            file_path = url.replace('file://', '') if url.startswith('file://') else url
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.endswith('.html'):
                return extract_text_from_html(content, url)
            else:
                return content
        
        # Handle HTTP/HTTPS URLs
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # For now, only handle HTML content
        if 'text/html' in response.headers.get('content-type', ''):
            return extract_text_from_html(response.text, url)
        else:
            return response.text
            
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at word boundary if not at end
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:  # Only if we don't lose too much
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks


def prepare_documents(urls: List[str], db_path: str) -> None:
    """Prepare documents by downloading, processing, and storing them (demo version)."""
    logger.info(f"Preparing {len(urls)} documents...")
    
    # Create directory for storing documents
    Path(db_path).mkdir(parents=True, exist_ok=True)
    
    all_documents = []
    
    # Process all URLs
    for i, url in enumerate(urls, 1):
        logger.info(f"Processing URL {i}/{len(urls)}: {url}")
        
        try:
            # Download content
            content = download_content(url)
            
            # Create document metadata
            doc_metadata = {
                "source": url,
                "url": url,
                "title": extract_title_from_url(url),
                "content_length": len(content)
            }
            
            # Split into chunks
            chunks = chunk_text(content)
            
            # Create document entries
            for j, chunk in enumerate(chunks):
                doc_entry = {
                    "id": f"{i}_{j}",
                    "content": chunk,
                    "metadata": {**doc_metadata, "chunk_index": j}
                }
                all_documents.append(doc_entry)
            
            logger.info(f"Created {len(chunks)} chunks from {url}")
            
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            continue
    
    if not all_documents:
        raise Exception("No documents were successfully processed")
    
    logger.info(f"Total chunks created: {len(all_documents)}")
    
    # Store documents as JSON (simple storage for demo)
    documents_file = Path(db_path) / "documents.json"
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)
    
    # Create metadata file
    metadata_file = Path(db_path) / "metadata.json"
    metadata = {
        "total_documents": len(all_documents),
        "source_urls": urls,
        "created_at": "demo_version"
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Documents stored at {db_path}")


def extract_title_from_url(url: str) -> str:
    """Extract a title from URL for metadata."""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if path:
        # Get the last part of the path as title
        title = path.split('/')[-1]
        # Remove file extension if present
        if '.' in title:
            title = title.rsplit('.', 1)[0]
        return title.replace('-', ' ').replace('_', ' ').title()
    else:
        return parsed.netloc