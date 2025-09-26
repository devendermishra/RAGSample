#!/usr/bin/env python3
"""
Test script to verify tool override functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_sample.llm_setup import setup_llm


def test_tool_override():
    """Test that tool parameter overrides environment priority."""
    print("🧪 Testing Tool Override Functionality")
    print("=" * 50)
    
    # Test case 1: Force Groq even when OpenAI key is present
    print("\n📋 Test Case 1: Force Groq with OpenAI key present")
    print("-" * 40)
    
    # Set up test environment
    original_openai = os.getenv("OPENAI_API_KEY")
    original_groq = os.getenv("GROQ_API_KEY")
    
    try:
        # Set both keys
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["GROQ_API_KEY"] = "test-groq-key"
        
        # Test 1: Without tool parameter (should use OpenAI due to priority)
        print("Testing auto-selection (should prefer OpenAI):")
        try:
            llm = setup_llm(model_name="llama3-8b-8192", tool=None)
            print(f"✅ Auto-selection result: {type(llm).__name__}")
        except Exception as e:
            print(f"❌ Auto-selection failed: {str(e)}")
        
        # Test 2: Force Groq with tool parameter
        print("\nTesting forced Groq selection:")
        try:
            llm = setup_llm(model_name="llama3-8b-8192", tool="groq")
            print(f"✅ Forced Groq result: {type(llm).__name__}")
        except Exception as e:
            print(f"❌ Forced Groq failed: {str(e)}")
        
        # Test 3: Force OpenAI with tool parameter
        print("\nTesting forced OpenAI selection:")
        try:
            llm = setup_llm(model_name="gpt-4o-mini", tool="openai")
            print(f"✅ Forced OpenAI result: {type(llm).__name__}")
        except Exception as e:
            print(f"❌ Forced OpenAI failed: {str(e)}")
            
    finally:
        # Restore original environment
        if original_openai:
            os.environ["OPENAI_API_KEY"] = original_openai
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
            
        if original_groq:
            os.environ["GROQ_API_KEY"] = original_groq
        elif "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
    
    # Test case 2: Invalid tool parameter
    print("\n📋 Test Case 2: Invalid tool parameter")
    print("-" * 40)
    try:
        llm = setup_llm(tool="invalid-tool")
        print("❌ Should have failed with invalid tool")
    except Exception as e:
        print(f"✅ Correctly failed with invalid tool: {str(e)}")
    
    # Test case 3: Tool specified but no corresponding API key
    print("\n📋 Test Case 3: Tool specified but no API key")
    print("-" * 40)
    try:
        # Clear all API keys
        for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
        
        llm = setup_llm(tool="openai")
        print("❌ Should have failed with missing API key")
    except Exception as e:
        print(f"✅ Correctly failed with missing API key: {str(e)}")


def show_current_environment():
    """Show current environment configuration."""
    print("\n🌍 Current Environment:")
    print("-" * 20)
    
    env_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "LLM_MODEL"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if "API_KEY" in var:
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"  {var}: {masked_value}")
            else:
                print(f"  {var}: {value}")
        else:
            print(f"  {var}: Not set")


if __name__ == "__main__":
    show_current_environment()
    test_tool_override()
    print("\n✅ Tool override testing completed!")
