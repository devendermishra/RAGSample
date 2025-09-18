"""Demo chat module for RAG Sample (without external dependencies)."""

import os
import json
from pathlib import Path
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DemoRAGChatBot:
    """A demo conversational RAG chatbot (simplified version)."""
    
    def __init__(self, db_path: str):
        """Initialize the chatbot with a document database."""
        self.db_path = db_path
        self.documents = []
        self._load_documents()
    
    def _load_documents(self):
        """Load documents from storage."""
        documents_file = Path(self.db_path) / "documents.json"
        if not documents_file.exists():
            raise Exception(f"Documents file not found at {documents_file}")
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        logger.info(f"Loaded {len(self.documents)} document chunks")
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based document search (demo implementation)."""
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            content = doc['content'].lower()
            score = 0
            
            # Simple scoring: count keyword matches
            for word in query_words:
                score += content.count(word)
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
    
    def generate_answer(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate an answer based on relevant documents (demo implementation)."""
        if not relevant_docs:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Simple answer generation
        answer_parts = [
            f"Based on the documents, here's what I found about '{query}':\n"
        ]
        
        for i, doc in enumerate(relevant_docs[:2], 1):
            content_preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            source = doc['metadata']['title']
            answer_parts.append(f"{i}. From '{source}': {content_preview}\n")
        
        return "\n".join(answer_parts)
    
    def chat(self, question: str) -> Dict:
        """Process a chat message and return response."""
        # Search for relevant documents
        relevant_docs = self.search_documents(question)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_docs)
        
        return {
            "answer": answer,
            "source_documents": relevant_docs
        }


def start_chat(db_path: str):
    """Start an interactive chat session (demo version)."""
    try:
        # Check if OpenAI API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print("🔑 OpenAI API key found! For full functionality, install the required dependencies:")
            print("   pip install langchain langchain-openai langchain-community chromadb")
            print("   Then the app will use advanced AI capabilities.\n")
        else:
            print("ℹ️  Running in demo mode (no OpenAI API key found)")
            print("   For full AI capabilities, set OPENAI_API_KEY environment variable")
            print("   and install required dependencies.\n")
        
        # Initialize chatbot
        chatbot = DemoRAGChatBot(db_path)
        
        print("🤖 RAG ChatBot (Demo Mode) initialized!")
        print("💡 This demo uses simple keyword matching for document search.")
        print("📝 Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process the question
                print("🤔 Searching documents...")
                response = chatbot.chat(user_input)
                
                # Display response
                print(f"\n🤖 Bot: {response['answer']}")
                
                # Show sources if available
                if response['source_documents']:
                    print(f"\n📖 Sources used ({len(response['source_documents'])} documents):")
                    for i, doc in enumerate(response['source_documents'], 1):
                        title = doc['metadata']['title']
                        source = doc['metadata']['source']
                        print(f"   {i}. {title} ({source})")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to initialize chat: {e}")
        raise