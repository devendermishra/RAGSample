"""Chat module for RAG Sample."""

import os
from typing import List
import logging
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGChatBot:
    """A conversational RAG chatbot."""
    
    def __init__(self, db_path: str):
        """Initialize the chatbot with a vector database."""
        self.db_path = db_path
        self.vectorstore = None
        self.chain = None
        self.memory = None
        self._setup()
    
    def _setup(self):
        """Set up the RAG chain."""
        # Check for OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise Exception(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable. "
                "You can copy .env.example to .env and add your API key."
            )
        
        # Load vector store
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Initialize LLM
        model_name = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            openai_api_key=api_key
        )
        
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create the conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
    
    def chat(self, question: str) -> dict:
        """Process a chat message and return response."""
        if not self.chain:
            raise Exception("Chat chain not initialized")
        
        # Get response from the chain
        response = self.chain({"question": question})
        
        return {
            "answer": response["answer"],
            "source_documents": response.get("source_documents", [])
        }
    
    def get_relevant_sources(self, question: str, k: int = 3) -> List[dict]:
        """Get relevant source documents for a question."""
        if not self.vectorstore:
            return []
        
        docs = self.vectorstore.similarity_search(question, k=k)
        sources = []
        
        for doc in docs:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", "Unknown")
            })
        
        return sources


def start_chat(db_path: str):
    """Start an interactive chat session."""
    try:
        # Initialize chatbot
        chatbot = RAGChatBot(db_path)
        
        print("🤖 RAG ChatBot initialized! Ask me anything about your documents.")
        print("💡 Type 'sources' to see relevant sources for your last question.")
        print("📝 Type 'quit' or 'exit' to end the session.\n")
        
        last_question = None
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'sources':
                    if last_question:
                        print("\n📚 Relevant sources for your last question:")
                        sources = chatbot.get_relevant_sources(last_question)
                        for i, source in enumerate(sources, 1):
                            print(f"\n{i}. {source['title']}")
                            print(f"   Source: {source['source']}")
                            print(f"   Preview: {source['content']}")
                    else:
                        print("❓ No previous question to show sources for.")
                    print()
                    continue
                
                if not user_input:
                    continue
                
                # Process the question
                print("🤔 Thinking...")
                response = chatbot.chat(user_input)
                
                # Display response
                print(f"\n🤖 Bot: {response['answer']}")
                
                # Show sources if available
                if response['source_documents']:
                    print(f"\n📖 Based on {len(response['source_documents'])} source(s)")
                    for i, doc in enumerate(response['source_documents'][:2], 1):
                        source = doc.metadata.get('source', 'Unknown')
                        title = doc.metadata.get('title', 'Unknown')
                        print(f"   {i}. {title} ({source})")
                
                print()
                last_question = user_input
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to initialize chat: {e}")
        raise