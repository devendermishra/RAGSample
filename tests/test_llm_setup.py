#!/usr/bin/env python3
"""
Test script for LLM setup functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_sample.llm_setup import setup_llm, get_available_providers, get_provider_from_api_key


def test_llm_setup():
    """Test LLM setup with different providers."""
    print("🔍 Testing LLM Setup Functionality")
    print("=" * 50)
    
    # Check available providers
    print(f"📦 Available providers: {get_available_providers()}")
    
    # Check current provider
    current_provider = get_provider_from_api_key()
    print(f"🔑 Current provider: {current_provider or 'None (no API key found)'}")
    
    # Test LLM setup
    try:
        print("\n🚀 Testing LLM setup...")
        llm = setup_llm(
            model_name="gpt-4o-mini",  # This will be overridden by env variables
            temperature=0.7,
            max_tokens=1000
        )
        print(f"✅ LLM setup successful: {type(llm).__name__}")
        
        # Test a simple prompt
        print("\n💬 Testing LLM with simple prompt...")
        response = llm.invoke("Hello, how are you?")
        print(f"🤖 Response: {response.content if hasattr(response, 'content') else response}")
        
    except Exception as e:
        print(f"❌ LLM setup failed: {str(e)}")
        print("\n💡 Make sure you have at least one API key set:")
        print("   - OPENAI_API_KEY")
        print("   - GOOGLE_API_KEY") 
        print("   - GROQ_API_KEY")


def show_environment_info():
    """Show current environment configuration."""
    print("\n🌍 Environment Configuration:")
    print("-" * 30)
    
    env_vars = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY", 
        "GROQ_API_KEY",
        "LLM_MODEL"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys for security
            if "API_KEY" in var:
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"  {var}: {masked_value}")
            else:
                print(f"  {var}: {value}")
        else:
            print(f"  {var}: Not set")


if __name__ == "__main__":
    show_environment_info()
    test_llm_setup()
