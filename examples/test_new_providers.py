#!/usr/bin/env python3
"""
Test script for new OpenRouter and Groq providers in Memvid
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memvid import LLMClient


def test_provider_availability():
    """Test which providers are available"""
    print("Testing provider availability...")
    print(f"Available providers: {LLMClient.list_available_providers()}")
    print(f"API key status: {LLMClient.check_api_keys()}")
    print()


def test_openrouter():
    """Test OpenRouter provider"""
    print("Testing OpenRouter provider...")
    
    # Check if OpenRouter API key is available
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("To test OpenRouter, set: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize client with OpenRouter
        client = LLMClient(provider='openrouter', model='google/gemini-2.0-flash-exp:free')
        
        # Test simple chat
        messages = [
            {"role": "user", "content": "Hello! Please respond with exactly 'OpenRouter is working!' if you can see this."}
        ]
        
        response = client.chat(messages)
        print(f"✅ OpenRouter response: {response}")
        
    except Exception as e:
        print(f"❌ OpenRouter test failed: {e}")
    
    print()


def test_groq():
    """Test Groq provider"""
    print("Testing Groq provider...")
    
    # Check if Groq API key is available
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY not found in environment variables")
        print("To test Groq, set: export GROQ_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize client with Groq
        client = LLMClient(provider='groq', model='llama3-70b-8192')
        
        # Test simple chat
        messages = [
            {"role": "user", "content": "Hello! Please respond with exactly 'Groq is working!' if you can see this."}
        ]
        
        response = client.chat(messages)
        print(f"✅ Groq response: {response}")
        
    except Exception as e:
        print(f"❌ Groq test failed: {e}")
    
    print()


def test_with_chat():
    """Test providers with MemvidChat (if video/index files exist)"""
    print("Testing with MemvidChat...")
    
    # Look for example video/index files
    video_files = []
    for file in os.listdir('.'):
        if file.endswith('.mp4') or file.endswith('.mkv'):
            index_file = file.replace('.mp4', '_index.json').replace('.mkv', '_index.json')
            if os.path.exists(index_file):
                video_files.append((file, index_file))
    
    if not video_files:
        print("❌ No video/index file pairs found for testing MemvidChat")
        print("Create a memory first using: python examples/build_memory.py")
        return
    
    video_file, index_file = video_files[0]
    print(f"Using files: {video_file}, {index_file}")
    
    # Test with different providers
    for provider in ['openrouter', 'groq']:
        api_key_env = f"{provider.upper()}_API_KEY"
        if not os.getenv(api_key_env):
            print(f"❌ {api_key_env} not set, skipping {provider}")
            continue
            
        try:
            from memvid import MemvidChat
            
            chat = MemvidChat(
                video_file=video_file,
                index_file=index_file,
                llm_provider=provider
            )
            
            response = chat.chat("What is this memory about?")
            print(f"✅ {provider} with MemvidChat: {response[:100]}...")
            
        except Exception as e:
            print(f"❌ {provider} with MemvidChat failed: {e}")
    
    print()


def main():
    """Main test function"""
    print("🧪 Testing new OpenRouter and Groq providers for Memvid")
    print("=" * 60)
    
    test_provider_availability()
    test_openrouter()
    test_groq()
    test_with_chat()
    
    print("✅ Testing complete!")
    print("\nTo use the new providers:")
    print("1. Set environment variables:")
    print("   export OPENROUTER_API_KEY='your-openrouter-key'")
    print("   export GROQ_API_KEY='your-groq-key'")
    print("2. Use in your code:")
    print("   client = LLMClient(provider='openrouter')")
    print("   client = LLMClient(provider='groq')")
    print("3. Or with file_chat.py:")
    print("   python examples/file_chat.py --provider openrouter --input-dir docs/")
    print("   python examples/file_chat.py --provider groq --input-dir docs/")


if __name__ == "__main__":
    main() 