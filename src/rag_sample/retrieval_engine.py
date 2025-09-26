"""
Document retrieval functionality for RAG system.
"""

from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Config
from .logging_config import get_logger

logger = get_logger(__name__)


def _is_content_relevant(doc: Document, question: str) -> bool:
    """Check if document content is relevant to the question using Ready Tensor techniques."""
    # Basic relevance checks
    question_lower = question.lower()
    content_lower = doc.page_content.lower()

    # Check for key terms overlap
    question_terms = set(question_lower.split())
    content_terms = set(content_lower.split())

    # Calculate basic relevance score
    overlap = len(question_terms.intersection(content_terms))
    relevance_score = overlap / len(question_terms) if question_terms else 0

    # Apply minimum relevance threshold
    return relevance_score >= 0.1  # 10% term overlap minimum


class RetrievalEngine:
    """Handles document retrieval using vector similarity search."""
    
    def __init__(self, config: Config, embeddings: HuggingFaceEmbeddings, collection):
        """Initialize retrieval engine.
        
        Args:
            config: Configuration object
            embeddings: Embeddings model
            collection: ChromaDB collection
        """
        self.config = config
        self.embeddings = embeddings
        self.collection = collection
    
    def retrieve_documents(self, question: str) -> List[Document]:
        """Retrieve relevant documents using Ready Tensor advanced techniques.
        
        Args:
            question: User's question
            
        Returns:
            List of relevant documents
        """
        try:
            # Generate embedding for the question
            question_embedding = self.embeddings.embed_query(question)
            
            # Query ChromaDB collection
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=self.config.retrieval_top_k * 2,  # Get more candidates for filtering
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to documents with scores
            docs_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    doc = Document(page_content=doc_text, metadata=metadata)
                    docs_with_scores.append((doc, distance))
            
            # Apply Ready Tensor's filtering and ranking
            filtered_docs = self._filter_and_rank_documents(docs_with_scores, question)
            
            return filtered_docs
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def _filter_and_rank_documents(self, docs_with_scores: List[tuple], question: str) -> List[Document]:
        """Filter and rank documents using Ready Tensor techniques."""
        filtered_docs = []
        
        for doc, score in docs_with_scores:
            # Apply similarity threshold
            if score <= self.config.retrieval_threshold:
                # Apply Ready Tensor's content relevance check
                if _is_content_relevant(doc, question):
                    filtered_docs.append(doc)
        
        # Limit to top_k results
        return filtered_docs[:self.config.retrieval_top_k]
