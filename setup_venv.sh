#!/bin/bash

# RAG Sample - Virtual Environment Setup Script

echo "Setting up Python virtual environment for RAG Sample..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/vector_db
mkdir -p documents
mkdir -p logs

# Create sample .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating sample .env file..."
    cat > .env << EOF
# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192

# Application Settings
TEMPERATURE=0.7
MAX_TOKENS=1000

# Vector Database Settings
VECTOR_DB_PATH=./data/vector_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Document Processing
DOCUMENTS_PATH=./documents

# Conversation Memory Settings
MAX_CONVERSATION_TOKENS=4000
SUMMARIZATION_THRESHOLD=0.8
ENABLE_CONVERSATION_MEMORY=true

# Document Retrieval Settings
RETRIEVAL_TOP_K=5
RETRIEVAL_THRESHOLD=0.3
ENABLE_RETRIEVAL_DEBUG=false

# UI Settings
USER_PROMPT=You
GOODBYE_MESSAGE=Goodbye! Thanks for using RAG Sample.
WELCOME_MESSAGE=Welcome to RAG Sample! Ask me anything about your documents.
EOF
    echo "Sample .env file created. Please update with your actual Groq API key."
fi

echo ""
echo "Setup complete! To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Update .env file with your Groq API key"
echo "3. Add documents to the ./documents directory"
echo "4. Install the package in development mode: pip install -e ."
echo "5. Run the application: rag-sample"
echo ""
echo "Alternative: python -m rag_sample.cli"
