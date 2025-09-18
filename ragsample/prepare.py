"""Document preparation module for RAG Sample."""

import os
import requests
from pathlib import Path
from typing import List
from urllib.parse import urlparse
import tempfile
import logging

from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_content(url: str) -> str:
    """Download content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if it's a PDF
        if 'application/pdf' in response.headers.get('content-type', ''):
            return download_pdf(url, response.content)
        else:
            return extract_text_from_html(response.text, url)
            
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        raise


def download_pdf(url: str, content: bytes) -> str:
    """Download and extract text from PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(content)
        tmp_file.flush()
        
        try:
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            text = "\n\n".join([doc.page_content for doc in documents])
            return text
        finally:
            os.unlink(tmp_file.name)


def extract_text_from_html(html_content: str, url: str) -> str:
    """Extract text content from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean up text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text


def prepare_documents(urls: List[str], db_path: str) -> None:
    """Prepare documents by downloading, processing, and storing them."""
    logger.info(f"Preparing {len(urls)} documents...")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Process all URLs
    all_docs = []
    for i, url in enumerate(urls, 1):
        logger.info(f"Processing URL {i}/{len(urls)}: {url}")
        
        try:
            # Download content
            content = download_content(url)
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "source": url,
                    "url": url,
                    "title": extract_title_from_url(url)
                }
            )
            
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            all_docs.extend(chunks)
            
            logger.info(f"Created {len(chunks)} chunks from {url}")
            
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            continue
    
    if not all_docs:
        raise Exception("No documents were successfully processed")
    
    logger.info(f"Total chunks created: {len(all_docs)}")
    
    # Create vector store
    logger.info("Creating vector store...")
    embeddings = OpenAIEmbeddings()
    
    # Create the database directory if it doesn't exist
    Path(db_path).mkdir(parents=True, exist_ok=True)
    
    # Store documents in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    vectorstore.persist()
    logger.info(f"Vector store created at {db_path}")


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