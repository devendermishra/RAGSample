# RAGSample

A sample command line (CLI) based conversational RAG app. It lets you chat with your documents using AI.

## Features

- **Document Preparation**: Download and process articles from URLs (HTML and PDF support)
- **Vector Storage**: Store documents in a local ChromaDB vector database (full version) or simple JSON storage (demo)
- **Conversational AI**: Chat with your documents using OpenAI's GPT models (full version) or keyword-based search (demo)
- **Source Citations**: See which documents the AI used to answer your questions

## Quick Start (Demo Mode)

The application works in demo mode without any dependencies installation:

1. Clone this repository:
```bash
git clone https://github.com/devendermishra/RAGSample.git
cd RAGSample
```

2. Try the demo with a local HTML file:
```bash
# Create a sample HTML file
echo '<html><body><h1>AI Article</h1><p>Artificial Intelligence is the simulation of human intelligence in machines.</p></body></html>' > sample.html

# Prepare the document
python main.py prepare sample.html

# Chat with it
python main.py chat
```

## Full Installation (Advanced AI Features)

For full functionality with OpenAI GPT models and advanced vector search:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. Use with web URLs:
```bash
python main.py prepare "https://example.com/article1" "https://example.com/article2.pdf"
python main.py chat
```

## Usage

The application has two main commands: `prepare` and `chat`.

### 1. Prepare Documents

Download and prepare documents for RAG:

```bash
# Prepare from URLs (requires internet and full installation)
python main.py prepare "https://example.com/article1" "https://example.com/article2.pdf"

# Prepare from local files (works in demo mode)
python main.py prepare "article1.html" "document.txt"
```

Options:
- `--db-path`: Specify where to store the database (default: `./rag_db`)

### 2. Chat with Documents

Start an interactive chat session:

```bash
python main.py chat
```

Options:
- `--db-path`: Specify the path to your database (default: `./rag_db`)

### Example Workflow

```bash
# 1. Prepare some documents (demo mode with local file)
echo '<html><body><h1>Machine Learning Guide</h1><p>Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.</p></body></html>' > ml_guide.html
python main.py prepare ml_guide.html

# 2. Start chatting
python main.py chat
# Type: "What is machine learning?"
# Type: "quit" to exit
```

During chat:
- Type your questions about the documents
- Type `quit` or `exit` to end the session
- In demo mode: Uses simple keyword matching
- In full mode: Uses advanced AI with OpenAI GPT models

## Demo Mode vs Full Mode

| Feature | Demo Mode | Full Mode |
|---------|-----------|-----------|
| Dependencies | None (built-in Python only) | langchain, openai, chromadb, etc. |
| Document Processing | Basic HTML text extraction | Advanced with BeautifulSoup, PDF support |
| Storage | Simple JSON files | ChromaDB vector database |
| Search | Keyword matching | Semantic vector search |
| AI Response | Template-based | OpenAI GPT models |
| Internet Required | No (local files only) | Yes (for URLs and AI) |

## Environment Variables

Create a `.env` file with:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo  # Optional, default is gpt-3.5-turbo
```

## Supported Document Types

- **Demo Mode**: HTML files, plain text files
- **Full Mode**: Web pages (HTML), PDF files, any URL that returns text content

## Requirements

- **Demo Mode**: Python 3.8+
- **Full Mode**: Python 3.8+, OpenAI API key, Internet connection

## License

MIT License - see LICENSE file for details.
