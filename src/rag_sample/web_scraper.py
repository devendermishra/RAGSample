"""
Web scraping functionality for extracting content from URLs.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import Optional, Dict, Any
import re
from pathlib import Path


class WebScraper:
    """Web scraper for extracting text content from URLs."""
    
    def __init__(self, timeout: int = 30, max_content_length: int = 10 * 1024 * 1024):
        """Initialize web scraper.
        
        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length in bytes (10MB default)
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def extract_content(self, url: str) -> Dict[str, Any]:
        """Extract content from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            # Validate URL
            if not self.is_valid_url(url):
                return {
                    "success": False,
                    "error": "Invalid URL format",
                    "content": "",
                    "metadata": {}
                }
            
            # Make request
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_content_length:
                return {
                    "success": False,
                    "error": f"Content too large ({int(content_length) / (1024*1024):.1f}MB). Maximum size is {self.max_content_length / (1024*1024):.1f}MB.",
                    "content": "",
                    "metadata": {}
                }
            
            # Get content
            content = response.text
            
            # Check actual content length
            if len(content) > self.max_content_length:
                return {
                    "success": False,
                    "error": f"Content too large ({len(content) / (1024*1024):.1f}MB). Maximum size is {self.max_content_length / (1024*1024):.1f}MB.",
                    "content": "",
                    "metadata": {}
                }
            
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            # Extract main content
            text_content = self._extract_text_content(soup)
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content found on the page",
                    "content": "",
                    "metadata": metadata
                }
            
            return {
                "success": True,
                "content": text_content,
                "metadata": metadata,
                "url": url
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "content": "",
                "metadata": {}
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Scraping failed: {str(e)}",
                "content": "",
                "metadata": {}
            }
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from the page.
        
        Args:
            soup: BeautifulSoup object
            url: Original URL
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            "url": url,
            "title": "",
            "description": "",
            "author": "",
            "published_date": "",
            "domain": urlparse(url).netloc
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # Extract meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata["description"] = desc_tag.get('content', '').strip()
        
        # Extract author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag:
            metadata["author"] = author_tag.get('content', '').strip()
        
        # Extract published date
        date_tag = soup.find('meta', attrs={'property': 'article:published_time'})
        if not date_tag:
            date_tag = soup.find('meta', attrs={'name': 'date'})
        if date_tag:
            metadata["published_date"] = date_tag.get('content', '').strip()
        
        return metadata
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from the page.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted text content
        """
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Look for common main content selectors
        main_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '.article-content'
        ]
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            main_content = soup
        
        # Extract text
        text = main_content.get_text()
        
        # Clean up text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def get_domain_from_url(self, url: str) -> str:
        """Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"
