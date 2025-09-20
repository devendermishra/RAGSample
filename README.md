# RAG Sample

A sample command line (CLI) based conversational RAG (Retrieval-Augmented Generation) application that lets you chat with your documents.

## Features

- 📄 Document ingestion (PDF, TXT, MD files)
- 🔍 Vector-based document search using ChromaDB
- 💬 Interactive CLI chat interface
- 🤖 Groq LLM integration (Llama, Mixtral models)
- ⚙️ Configurable settings
- 🎨 Rich terminal interface

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key (get one at https://console.groq.com/)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RAGSample
   ```

2. **Set up the environment:**
   ```bash
   ./setup_venv.sh
   ```

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

4. **Configure your API key:**
   ```bash
   cp env.example .env
   # Edit .env and add your Groq API key
   ```

5. **Add documents:**
   ```bash
   # Place your PDF, TXT, or MD files in the documents/ directory
   mkdir -p documents
   cp your-document.pdf documents/
   ```

6. **Run the application:**
   ```bash
   # Option 1: Direct execution
   python -m rag_sample.cli
   
   # Option 2: Install and use CLI command
   pip install -e .
   rag-sample
   ```

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
- `/docs` or `/documents` - List documents in the documents directory
- `/docstats` - Show document statistics from the vector database

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

### **Example Document Session:**

```
You: /docs
Documents Directory: ./documents
Found 3 supported documents:
  📄 manual.pdf (2,456,789 bytes)
  📄 notes.txt (15,234 bytes)
  📄 README.md (8,456 bytes)

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
```

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
