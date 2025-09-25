# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modular architecture with separate document and retrieval engines
- Comprehensive test suite with pytest
- GPU acceleration support with PyTorch and CUDA
- Docker containerization with multi-stage builds
- Environment management with conda, pipenv, and poetry
- Code of Conduct and contribution guidelines
- Detailed GPU requirements documentation
- Project structure documentation

### Changed
- Refactored RAG engine into smaller, focused modules
- Improved error handling and logging
- Enhanced document retrieval with advanced filtering
- Updated dependencies to latest stable versions

### Fixed
- Rate limiting issues with Groq API
- Document retrieval threshold optimization
- Memory management for large document collections
- Cross-platform compatibility issues

## [0.1.0] - 2024-09-25

### Added
- Initial release of RAG Sample application
- Command-line interface with rich terminal output
- Document ingestion from PDF, TXT, and MD files
- Web scraping capabilities for URL content
- Vector similarity search with ChromaDB
- Conversation memory with token tracking
- Prompt management with YAML configuration
- Groq LLM integration
- HuggingFace embeddings support
- Configuration management with environment variables
- Virtual environment setup script
- Basic documentation and installation guide

### Features
- **Document Management**: Add, remove, and list documents
- **Web Scraping**: Extract content from URLs
- **Vector Search**: Semantic similarity search with configurable parameters
- **Conversation Memory**: Context-aware conversations with summarization
- **Prompt Engineering**: YAML-based prompt configuration
- **CLI Interface**: Rich terminal interface with command history
- **Configuration**: Flexible configuration with environment variables

### Technical Details
- **Language**: Python 3.8+
- **Dependencies**: LangChain, ChromaDB, Groq, HuggingFace
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Groq API with multiple model support
- **Web Scraping**: BeautifulSoup4 with lxml parser
