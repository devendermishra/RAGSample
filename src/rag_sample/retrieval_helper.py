from langchain_core.documents import Document


def is_content_relevant(doc: Document, question: str) -> bool:
    """
    Check if content is relevant to the question using Ready Tensor techniques.
    Args:
        doc: Document to check
        question: User's question
    Returns:
        True if relevant, False otherwise
    """
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
