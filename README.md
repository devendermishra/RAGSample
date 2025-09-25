# RAG Sample

A sample command line (CLI) based conversational RAG (Retrieval-Augmented Generation) application that lets you chat with your documents.

## Features

- 📄 Document ingestion (PDF, TXT, MD files)
- 🔍 Vector-based document search using ChromaDB
- 💬 Interactive CLI chat interface
- 🤖 Groq LLM integration (Llama, Mixtral models)
- ⚙️ Configurable settings
- 🎨 Rich terminal interface
- 🧪 Comprehensive test suite
- 🐳 Docker containerization
- 🚀 GPU acceleration support
- 📦 Multiple environment management options

## Project Structure

```
RAGSample/
├── src/rag_sample/           # Main application code
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration management
│   ├── rag_engine.py        # Core RAG functionality
│   ├── document_manager.py  # Document processing and management
│   ├── retrieval_engine.py  # Vector similarity search
│   ├── conversation_memory.py # Chat history and summarization
│   ├── web_scraper.py       # URL content extraction
│   └── prompt_builder.py    # YAML-based prompt management
├── tests/                   # Test suite
│   ├── test_config.py       # Configuration tests
│   ├── test_document_manager.py # Document management tests
│   └── test_web_scraper.py  # Web scraping tests
├── config/                  # Configuration files
│   └── prompts.yaml        # Prompt templates
├── documents/               # Document storage directory
├── data/                    # Data storage
│   └── vector_db/          # ChromaDB vector database
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── requirements.lock        # Exact dependency versions
├── environment.yml         # Conda environment specification
├── Pipfile                 # Pipenv configuration
├── Dockerfile              # Docker containerization
├── docker-compose.yml      # Docker Compose configuration
├── setup.py                # Package installation
├── setup_venv.sh          # Virtual environment setup
├── .env.example            # Environment variables template
├── GPU_REQUIREMENTS.md     # GPU acceleration documentation
├── CODE_OF_CONDUCT.md      # Community guidelines
├── CHANGELOG.md            # Version history
└── README.md               # This file
```

### Directory Purposes

- **`src/rag_sample/`**: Core application modules with clear separation of concerns
- **`tests/`**: Comprehensive test suite covering all major functionality
- **`config/`**: YAML-based configuration files for prompts and settings
- **`documents/`**: User-uploaded documents for processing and indexing
- **`data/vector_db/`**: Persistent ChromaDB vector database storage
- **`logs/`**: Application logs for debugging and monitoring

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key (get one at https://console.groq.com/)

### Installation

#### **Quick Start (Recommended):**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RAGSample
   ```

2. **Run the setup script:**
   ```bash
   ./setup_venv.sh
   ```

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

5. **Configure your API key:**
   ```bash
   cp env.example .env
   # Edit .env and add your Groq API key
   ```

6. **Run the application:**
   ```bash
   rag-sample
   ```

#### **Manual Installation:**

If you encounter issues with the setup script:

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Configure and run:**
   ```bash
   cp env.example .env
   # Edit .env with your Groq API key
   rag-sample
   ```

#### **Troubleshooting:**

- **"externally-managed-environment" error:** Use virtual environment (see above)
- **"command not found: pip":** Use `python3 -m pip` instead
- **Permission errors:** Use virtual environment or `pip install --user`

For detailed troubleshooting, see [INSTALL.md](INSTALL.md).

## Usage

Once the application is running, you can:

1. **Start chatting:** Type your questions about the documents
2. **End session:** Type `quit`, `exit`, or `q` to exit
3. **Add more documents:** Place files in the `documents/` directory and restart

### Example Session

```
You: What is the main topic of the document?
Assistant: Based on the document content, the main topic is...

You: Can you summarize the key points?
Assistant: Here are the key points from the document:
1. Point one...
2. Point two...
```

## Configuration

The application can be configured through environment variables or a `.env` file:

- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_MODEL`: Model to use (default: llama3-8b-8192)
  - Available models: `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`
- `TEMPERATURE`: Response creativity (default: 0.7)
- `MAX_TOKENS`: Maximum response length (default: 1000)
- `VECTOR_DB_PATH`: Path to vector database (default: ./data/vector_db)
- `DOCUMENTS_PATH`: Path to documents directory (default: ./documents)
- `MAX_CONVERSATION_TOKENS`: Maximum tokens for conversation memory (default: 4000)
- `SUMMARIZATION_THRESHOLD`: When to summarize conversation (default: 0.8)
- `ENABLE_CONVERSATION_MEMORY`: Enable conversation context (default: true)
- `RETRIEVAL_TOP_K`: Number of documents to retrieve (default: 5)
- `RETRIEVAL_THRESHOLD`: Similarity threshold for document retrieval (default: 0.3)
- `ENABLE_RETRIEVAL_DEBUG`: Enable retrieval debugging (default: false)
- `USER_PROMPT`: User input prompt text (default: "You")
- `GOODBYE_MESSAGE`: Exit message (default: "Goodbye! Thanks for using RAG Sample.")
- `WELCOME_MESSAGE`: Welcome message (default: "Welcome to RAG Sample! Ask me anything about your documents.")

## Available Groq Models

The application supports several Groq models:

- **llama3-8b-8192**: Fast and efficient 8B parameter model (default)
- **llama3-70b-8192**: More capable 70B parameter model
- **mixtral-8x7b-32768**: Mixture of experts model with 8x7B parameters

You can switch models by:
1. Setting the `GROQ_MODEL` environment variable
2. Using the `--model` CLI option: `rag-sample --model llama3-70b-8192`

## Conversation Memory

The application now includes intelligent conversation memory that:

- **Maintains Context**: Remembers previous questions and answers in the session
- **Token Management**: Automatically tracks token usage and summarizes when needed
- **Smart Summarization**: When conversation exceeds token threshold, older messages are summarized
- **Memory Commands**: Use `/stats`, `/clear`, `/history` during chat

### Memory Commands

During a chat session, you can use these commands:

- `/stats` or `/status` - Show conversation statistics and token usage
- `/clear` or `/reset` - Clear the conversation memory
- `/history` or `/messages` - Show recent conversation messages
- `/add <file_path>` - Add a new document to the knowledge base
- `/docs` or `/documents` - List documents in the documents directory and vector store
- `/docstats` - Show document statistics from the vector database
- `/remove <document_identifier>` - Remove a document from the knowledge base
- `/listdocs` or `/list` - List all documents in the vector store
- `/cmdhistory` or `/cmds` - Show command history (last 20 commands)
- `/clearhistory` or `/clearhist` - Clear command history
- `/retrieval` or `/retrievalconfig` - Show retrieval configuration
- `/setretrieval <k> <threshold>` - Set retrieval parameters
- `/debug` or `/toggledebug` - Toggle retrieval debug mode
- `/ui` or `/uiconfig` - Show current UI configuration
- `/setui <user_prompt>|<goodbye_message>` - Set UI messages

### Example Session with Memory

```
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence...

You: Can you explain the types?
Assistant: Based on our previous discussion about machine learning, there are three main types...

You: /stats
Conversation Stats:
  total_messages: 4
  current_tokens: 1200
  max_tokens: 4000
  has_summary: false
  token_usage_percentage: 30.0
```

## Command History & Navigation

The CLI includes intelligent command history and navigation features:

### **Arrow Key Navigation:**
- **↑ (Up Arrow)** - Navigate to previous commands
- **↓ (Down Arrow)** - Navigate to next commands
- **Tab** - Auto-complete commands

### **Command History Features:**
- **Persistent History** - Commands are saved between sessions
- **Smart Deduplication** - Avoids duplicate consecutive commands
- **Tab Completion** - Auto-complete common commands
- **History Management** - View and clear command history

### **Example Navigation Session:**

```
You: /stats
Conversation Stats:
  total_messages: 4
  current_tokens: 1200
  max_tokens: 4000
  token_usage_percentage: 30.0

You: /cmdhistory
Command History (last 20):
   1. What is machine learning?
   2. Can you explain the types?
   3. /stats
   4. /cmdhistory

You: [Press ↑ to navigate to previous commands]
You: What is machine learning? [Previous command appears]
```

### **History Commands:**
- **`/cmdhistory`** or **`/cmds`** - Show last 20 commands
- **`/clearhistory`** or **`/clearhist`** - Clear command history

## UI Configuration

The RAG system allows you to customize the user interface messages for a personalized experience.

### UI Settings

- **`USER_PROMPT`**: The text displayed before user input (default: "You")
- **`GOODBYE_MESSAGE`**: Message shown when exiting the application
- **`WELCOME_MESSAGE`**: Message shown when starting the application

### UI Management Commands

- **`/ui`** or **`/uiconfig`** - Show current UI configuration
- **`/setui <user_prompt>|<goodbye_message>`** - Set UI messages

### Example UI Configuration

```bash
# View current UI settings
You: /ui
UI Configuration:
  User Prompt: 'You'
  Goodbye Message: 'Goodbye! Thanks for using RAG Sample.'
  Welcome Message: 'Welcome to RAG Sample! Ask me anything about your documents.'

# Customize UI messages
You: /setui 'User'|'See you later!'
✅ UI settings updated:
  User Prompt: 'User'
  Goodbye Message: 'See you later!'
```

### Environment Configuration

You can also set UI messages via environment variables:

```bash
# .env file
USER_PROMPT=User
GOODBYE_MESSAGE=See you later!
WELCOME_MESSAGE=Welcome to the RAG Assistant!
```

## Document Management

The application supports comprehensive document management including adding, listing, and removing documents from the knowledge base.

### **Document Removal:**

You can remove documents from the knowledge base using various identifiers:

#### **Remove by Filename:**
```bash
You: /remove manual.pdf
✅ Document 'manual.pdf' removed successfully!
```

#### **Remove by URL:**
```bash
You: /remove https://example.com/article
✅ Document 'https://example.com/article' removed successfully!
```

#### **Remove by Partial Match:**
```bash
You: /remove readytensor
✅ Document 'readytensor' removed successfully!
```

### **Document Listing:**

Use `/listdocs` to see all documents in the vector store with detailed information:

```
You: /listdocs
Documents in Vector Store:
   1. RAG Pipeline Implementation
      Source: https://app.readytensor.ai/publications/rag-pipeline-retrieval-augmented-generation-system-impl-using-langchain-chromadb-and-streamlit-cVoJzVn1gPOT
      Type: web_page
      Chunks: 15
      Domain: app.readytensor.ai

   2. Manual
      Source: ./documents/manual.pdf
      Type: pdf
      Chunks: 8
```

## Document Management

The application supports adding documents to the knowledge base in several ways:

### **Supported File Types:**
- **PDF files** (`.pdf`) - Extracts text from PDF documents
- **Text files** (`.txt`) - Plain text documents
- **Markdown files** (`.md`) - Markdown formatted documents
- **Web pages** (URLs) - Extracts text content from web pages

### **Adding Documents:**

#### **Method 1: Place files in documents directory**
```bash
# Copy documents to the documents directory
cp my-document.pdf documents/
cp notes.txt documents/
```

#### **Method 2: Use CLI commands during chat**
```
You: /add /path/to/document.pdf
✅ Document 'document.pdf' added successfully!

You: /add documents/new-file.txt
✅ Document 'new-file.txt' added successfully!

You: @https://example.com/article
✅ Web content from 'https://example.com/article' added successfully!
```

#### **Method 3: Programmatically**
```python
from rag_sample.rag_engine import RAGEngine

rag_engine = RAGEngine()

# Add local file
success = rag_engine.add_document("path/to/document.pdf")

# Add web content
success = rag_engine.add_document_from_url("https://example.com/article")
```

### **Document Commands:**

- **`/add <file_path>`** - Add a document to the knowledge base
- **`@<url>`** - Add web content from a URL
- **`/docs`** - List all documents in the documents directory
- **`/docstats`** - Show vector database statistics
- **`/remove <document_identifier>`** - Remove a document from the knowledge base
- **`/listdocs`** - List all documents in the vector store

### **Example Document Session:**

```
You: /docs
Documents Directory: ./documents
Found 3 supported documents:
  📄 manual.pdf (2,456,789 bytes)
  📄 notes.txt (15,234 bytes)
  📄 README.md (8,456 bytes)

Documents in Vector Store:
Found 4 documents in vector store:
   1. 🌐 RAG Pipeline Implementation
      Source: https://app.readytensor.ai/publications/rag-pipeline-retrieval-augmented-generation-system-impl-using-langchain-chromadb-and-streamlit-cVoJzVn1gPOT
      Type: web_page
      Chunks: 15
      Domain: app.readytensor.ai

   2. 📄 Manual
      Source: ./documents/manual.pdf
      Type: pdf
      Chunks: 8

You: /add new-document.pdf
Adding document: new-document.pdf...
✅ Document 'new-document.pdf' added successfully!

You: @https://app.readytensor.ai/publications/rag-pipeline-retrieval-augmented-generation-system-impl-using-langchain-chromadb-and-streamlit-cVoJzVn1gPOT
Scraping and adding web content from: https://app.readytensor.ai/publications/rag-pipeline-retrieval-augmented-generation-system-impl-using-langchain-chromadb-and-streamlit-cVoJzVn1gPOT...
✅ Web content from 'https://app.readytensor.ai/publications/rag-pipeline-retrieval-augmented-generation-system-impl-using-langchain-chromadb-and-streamlit-cVoJzVn1gPOT' added successfully!

You: /docstats
Document Statistics:
  total_documents: 5
  vector_db_path: ./data/vector_db
  documents_path: ./documents
  chunk_size: 1000
  chunk_overlap: 200

You: /listdocs
Documents in Vector Store:
   1. RAG Pipeline Implementation
      Source: https://app.readytensor.ai/publications/rag-pipeline-retrieval-augmented-generation-system-impl-using-langchain-chromadb-and-streamlit-cVoJzVn1gPOT
      Type: web_page
      Chunks: 15
      Domain: app.readytensor.ai

   2. Manual
      Source: ./documents/manual.pdf
      Type: pdf
      Chunks: 8

You: /remove manual.pdf
Removing document: manual.pdf...
Successfully removed 8 document chunks matching 'manual.pdf'
✅ Document 'manual.pdf' removed successfully!

You: /listdocs
Documents in Vector Store:
   1. RAG Pipeline Implementation
      Source: https://app.readytensor.ai/publications/rag-pipeline-retrieval-augmented-generation-system-impl-using-langchain-chromadb-and-streamlit-cVoJzVn1gPOT
      Type: web_page
      Chunks: 15
      Domain: app.readytensor.ai
```

## Contact Information

For questions, issues, or contributions, please contact:

- **Maintainer**: Devender Mishra
- **Email**: devender.mishra@example.com
- **GitHub**: [@devendermishra](https://github.com/devendermishra)
- **Issues**: [GitHub Issues](https://github.com/devendermishra/RAGSample/issues)

## License

Copyright (c) 2024 Devender Mishra. All rights reserved.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Development

### Project Structure

```
RAGSample/
├── src/
│   └── rag_sample/
│       ├── __init__.py
│       ├── cli.py          # Command line interface
│       ├── config.py       # Configuration management
│       └── rag_engine.py   # Core RAG functionality
├── documents/              # Place your documents here
├── data/                   # Vector database storage
├── requirements.txt        # Python dependencies
├── setup.py               # Package configuration
├── setup_venv.sh          # Environment setup script
└── README.md
```

### Running Tests

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Code formatting
black src/

# Linting
flake8 src/
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

If you encounter any issues or have questions, please open an issue on the repository.
