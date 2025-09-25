"""
Tests for web scraping functionality.
"""

import pytest
from unittest.mock import Mock, patch
from src.rag_sample.web_scraper import WebScraper


class TestWebScraper:
    """Test web scraper functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scraper = WebScraper()
    
    def test_web_scraper_initialization(self):
        """Test web scraper initialization."""
        assert self.scraper.timeout == 30
        assert self.scraper.max_content_length == 10 * 1024 * 1024
        assert hasattr(self.scraper, 'session')
    
    def test_is_valid_url_valid(self):
        """Test URL validation with valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://www.example.com/path",
            "http://subdomain.example.com:8080/path?query=value"
        ]
        
        for url in valid_urls:
            assert self.scraper.is_valid_url(url) is True
    
    def test_is_valid_url_invalid(self):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "example.com",
            "",
            None
        ]
        
        for url in invalid_urls:
            assert self.scraper.is_valid_url(url) is False
    
    def test_get_domain_from_url(self):
        """Test domain extraction from URLs."""
        test_cases = [
            ("https://example.com", "example.com"),
            ("https://www.example.com/path", "www.example.com"),
            ("http://subdomain.example.com:8080", "subdomain.example.com"),
            ("invalid-url", "unknown")
        ]
        
        for url, expected_domain in test_cases:
            result = self.scraper.get_domain_from_url(url)
            assert result == expected_domain
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        dirty_text = "  Hello   world  \n\n  "
        cleaned = self.scraper._clean_text(dirty_text)
        assert cleaned == "Hello world"
    
    def test_clean_text_html_entities(self):
        """Test HTML entity cleaning."""
        text_with_entities = "Hello &amp; world &lt;test&gt;"
        cleaned = self.scraper._clean_text(text_with_entities)
        assert "&amp;" not in cleaned
        assert "&lt;" not in cleaned
        assert "&gt;" not in cleaned
    
    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        assert self.scraper._clean_text("") == ""
        assert self.scraper._clean_text(None) == ""
    
    @patch('src.rag_sample.web_scraper.requests.Session.get')
    def test_extract_content_invalid_url(self, mock_get):
        """Test content extraction with invalid URL."""
        result = self.scraper.extract_content("invalid-url")
        assert result["success"] is False
        assert "Invalid URL format" in result["error"]
    
    @patch('src.rag_sample.web_scraper.requests.Session.get')
    def test_extract_content_request_error(self, mock_get):
        """Test content extraction with request error."""
        mock_get.side_effect = Exception("Connection error")
        
        result = self.scraper.extract_content("https://example.com")
        assert result["success"] is False
        assert "Request failed" in result["error"]
