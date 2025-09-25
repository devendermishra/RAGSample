"""
Web scraping functionality for extracting content from URLs.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict, Any
import re


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
            
            # Debug: Check if we got meaningful content
            if not text_content or len(text_content.strip()) < 50:
                # Try alternative extraction methods
                print(f"Warning: Low content extracted ({len(text_content)} chars), trying alternative methods...")
                
                # Try extracting from all paragraphs
                paragraphs = soup.find_all('p')
                if paragraphs:
                    alt_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    if len(alt_text) > len(text_content):
                        text_content = alt_text
                
                # Try extracting from all divs with text
                if len(text_content) < 50:
                    divs = soup.find_all('div')
                    div_texts = [div.get_text(strip=True) for div in divs if div.get_text(strip=True)]
                    if div_texts:
                        alt_text = ' '.join(div_texts)
                        if len(alt_text) > len(text_content):
                            text_content = alt_text
            
            if not text_content or len(text_content.strip()) < 20:
                return {
                    "success": False,
                    "error": f"No meaningful text content found on the page (got {len(text_content)} characters)",
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
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
        
        # Try to find main content area with multiple strategies
        main_content = None
        
        # Strategy 1: Look for semantic HTML5 elements
        semantic_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.main-content',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '#main',
            '.main'
        ]
        
        for selector in semantic_selectors:
            try:
                main_content = soup.select_one(selector)
                if main_content and main_content.get_text(strip=True):
                    break
            except Exception:
                continue
        
        # Strategy 2: Look for common content patterns
        if not main_content:
            content_patterns = [
                '.post',
                '.entry',
                '.article',
                '.story',
                '.text',
                '.body',
                'div[class*="content"]',
                'div[class*="text"]',
                'div[class*="article"]'
            ]
            
            for pattern in content_patterns:
                try:
                    main_content = soup.select_one(pattern)
                    if main_content and main_content.get_text(strip=True):
                        break
                except Exception:
                    continue
        
        # Strategy 3: Find the largest text block
        if not main_content:
            # Get all divs and find the one with most text
            divs = soup.find_all('div')
            if divs:
                largest_div = max(divs, key=lambda x: len(x.get_text(strip=True)))
                if len(largest_div.get_text(strip=True)) > 100:  # Minimum content threshold
                    main_content = largest_div
        
        # Strategy 4: Use body as fallback
        if not main_content:
            main_content = soup.find('body')
        
        # Final fallback: use entire soup
        if not main_content:
            main_content = soup
        
        # Extract text with proper handling
        try:
            text = main_content.get_text(separator=' ', strip=True)
        except Exception:
            text = str(main_content)
        
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
        if not text:
            return ""
        
        # Remove HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive newlines and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove very short lines that are likely navigation or ads
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10 or line in ['', ' ', '\t']:  # Keep meaningful content
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
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
    
    def test_scraping(self, url: str) -> Dict[str, Any]:
        """Test scraping functionality with detailed output.
        
        Args:
            url: URL to test
            
        Returns:
            Detailed scraping results
        """
        print(f"Testing web scraping for: {url}")
        
        result = self.extract_content(url)
        
        if result["success"]:
            content = result["content"]
            metadata = result["metadata"]
            
            print(f"✅ Successfully scraped {len(content)} characters")
            print(f"Title: {metadata.get('title', 'No title')}")
            print(f"Domain: {metadata.get('domain', 'Unknown')}")
            print(f"Content preview: {content[:200]}...")
            
            return {
                "success": True,
                "content_length": len(content),
                "title": metadata.get('title', ''),
                "domain": metadata.get('domain', ''),
                "preview": content[:500]
            }
        else:
            print(f"❌ Failed to scrape: {result['error']}")
            return {
                "success": False,
                "error": result["error"]
            }
