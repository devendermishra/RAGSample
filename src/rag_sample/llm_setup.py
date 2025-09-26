"""
LLM Setup utility for supporting multiple LLM providers.
"""

import os
from typing import Optional, Any
from .logging_config import get_logger
from .exceptions import ConfigurationError

logger = get_logger(__name__)


def setup_llm(model_name: Optional[str] = None, temperature: float = 0.7, 
              max_tokens: int = 1000, tool: Optional[str] = None) -> Any:
    """
    Setup LLM based on available API keys and model name.
    
    Priority order for API keys:
    1. OPENAI_API_KEY
    2. GOOGLE_API_KEY  
    3. GROQ_API_KEY
    
    Args:
        model_name: Model name from LLM_MODEL env variable or CLI input
        temperature: Temperature for response generation
        max_tokens: Maximum tokens for response
        tool: Desired provider (case-insensitive). One of: 'openai', 'gemini'/'google', 'groq'.
        
    Returns:
        Initialized LLM instance
        
    Raises:
        ConfigurationError: If no valid API key is found
    """
    # Get model name from environment or use provided
    env_model = os.getenv("LLM_MODEL")
    final_model = model_name or env_model
    
    # Normalize requested tool if provided
    requested_tool = tool.lower() if isinstance(tool, str) else None
    if requested_tool == "google":
        requested_tool = "gemini"

    # Check for API keys in priority order
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    # If a tool is explicitly requested, honor it (and validate key)
    if requested_tool in {"openai", "gemini", "groq"}:
        if requested_tool == "openai":
            if not openai_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY is required when tool=openai", "MISSING_API_KEY"
                )
            logger.info("Using OpenAI API (forced by tool parameter)")
            return _setup_openai_llm(openai_key, final_model, temperature, max_tokens)
        elif requested_tool == "gemini":
            if not google_key:
                raise ConfigurationError(
                    "GOOGLE_API_KEY is required when tool=gemini/google", "MISSING_API_KEY"
                )
            logger.info("Using Google/Gemini API (forced by tool parameter)")
            return _setup_gemini_llm(google_key, final_model, temperature, max_tokens)
        elif requested_tool == "groq":
            if not groq_key:
                raise ConfigurationError(
                    "GROQ_API_KEY is required when tool=groq", "MISSING_API_KEY"
                )
            logger.info("Using Groq API (forced by tool parameter)")
            return _setup_groq_llm(groq_key, final_model, temperature, max_tokens)

    if requested_tool and requested_tool not in {"openai", "gemini", "groq"}:
        raise ConfigurationError(
            "Invalid tool specified. Use one of: openai, gemini (or google), groq",
            "INVALID_TOOL",
        )

    # Auto-select based on env priority if tool not specified
    # Try OpenAI first
    if openai_key:
        logger.info("Using OpenAI API")
        return _setup_openai_llm(openai_key, final_model, temperature, max_tokens)
    
    # Try Google/Gemini second
    if google_key:
        logger.info("Using Google/Gemini API")
        return _setup_gemini_llm(google_key, final_model, temperature, max_tokens)
    
    # Try Groq as fallback
    if groq_key:
        logger.info("Using Groq API")
        return _setup_groq_llm(groq_key, final_model, temperature, max_tokens)
    
    # No API key found
    error_msg = (
        "No valid API key found. Please set one of the following environment variables:\n"
        "  - OPENAI_API_KEY (for OpenAI models)\n"
        "  - GOOGLE_API_KEY (for Gemini models)\n"
        "  - GROQ_API_KEY (for Groq models)\n"
        "Priority order: OPENAI_API_KEY > GOOGLE_API_KEY > GROQ_API_KEY"
    )
    logger.error(error_msg)
    raise ConfigurationError(error_msg, "MISSING_API_KEY")


def _setup_openai_llm(api_key: str, model_name: Optional[str], 
                      temperature: float, max_tokens: int) -> Any:
    """Setup OpenAI LLM."""
    try:
        from langchain_openai import ChatOpenAI
        
        # Use default model if not specified
        model = model_name or "gpt-4o-mini"
        
        logger.info(f"Initializing OpenAI model: {model}")
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except ImportError:
        error_msg = "langchain-openai package not installed. Install with: pip install langchain-openai"
        logger.error(error_msg)
        raise ConfigurationError(error_msg, "MISSING_DEPENDENCY")


def _setup_gemini_llm(api_key: str, model_name: Optional[str], 
                      temperature: float, max_tokens: int) -> Any:
    """Setup Google Gemini LLM."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Use default model if not specified
        model = model_name or "gemini-2.0-flash"
        
        logger.info(f"Initializing Gemini model: {model}")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    except ImportError:
        error_msg = "langchain-google-genai package not installed. Install with: pip install langchain-google-genai"
        logger.error(error_msg)
        raise ConfigurationError(error_msg, "MISSING_DEPENDENCY")


def _setup_groq_llm(api_key: str, model_name: Optional[str], 
                    temperature: float, max_tokens: int) -> Any:
    """Setup Groq LLM."""
    try:
        from langchain_groq import ChatGroq
        
        # Use default model if not specified
        model = model_name or "llama3-8b-8192"
        
        logger.info(f"Initializing Groq model: {model}")
        return ChatGroq(
            groq_api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except ImportError:
        error_msg = "langchain-groq package not installed. Install with: pip install langchain-groq"
        logger.error(error_msg)
        raise ConfigurationError(error_msg, "MISSING_DEPENDENCY")


def get_available_providers() -> list:
    """
    Get list of available LLM providers based on installed packages.
    
    Returns:
        List of available provider names
    """
    providers = []
    
    try:
        import langchain_openai
        providers.append("openai")
    except ImportError:
        pass
    
    try:
        import langchain_google_genai
        providers.append("gemini")
    except ImportError:
        pass
    
    try:
        import langchain_groq
        providers.append("groq")
    except ImportError:
        pass
    
    return providers


def get_provider_from_api_key() -> Optional[str]:
    """
    Get the provider name based on available API keys.
    
    Returns:
        Provider name or None if no API key found
    """
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    elif os.getenv("GROQ_API_KEY"):
        return "groq"
    return None
