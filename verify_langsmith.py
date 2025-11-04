#!/usr/bin/env python3
"""
LangSmith Setup Verification Script

Run this to verify your LangSmith configuration is working correctly.
"""

import os
import sys
from dotenv import load_dotenv

def check_langsmith_setup():
    """Check if LangSmith is properly configured"""
    
    print("🔍 LangSmith Setup Verification")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    checks_passed = 0
    checks_total = 5
    
    # Check 1: LANGCHAIN_TRACING_V2
    print("\n1️⃣  Checking LANGCHAIN_TRACING_V2...")
    tracing = os.getenv('LANGCHAIN_TRACING_V2', '').lower()
    if tracing == 'true':
        print("   ✅ LANGCHAIN_TRACING_V2 is enabled")
        checks_passed += 1
    else:
        print(f"   ❌ LANGCHAIN_TRACING_V2 is '{tracing}' (should be 'true')")
        print("   💡 Set LANGCHAIN_TRACING_V2=true in your .env file")
    
    # Check 2: LANGCHAIN_API_KEY
    print("\n2️⃣  Checking LANGCHAIN_API_KEY...")
    api_key = os.getenv('LANGCHAIN_API_KEY', '')
    if api_key and api_key.startswith('lsv2_'):
        print(f"   ✅ LANGCHAIN_API_KEY is set ({api_key[:20]}...)")
        checks_passed += 1
    elif api_key:
        print(f"   ⚠️  LANGCHAIN_API_KEY is set but doesn't start with 'lsv2_'")
        print("   💡 Make sure you're using a LangSmith API key")
    else:
        print("   ❌ LANGCHAIN_API_KEY is not set")
        print("   💡 Get your API key from https://smith.langchain.com/settings")
    
    # Check 3: LANGCHAIN_ENDPOINT
    print("\n3️⃣  Checking LANGCHAIN_ENDPOINT...")
    endpoint = os.getenv('LANGCHAIN_ENDPOINT', '')
    if endpoint == 'https://api.smith.langchain.com':
        print("   ✅ LANGCHAIN_ENDPOINT is correct")
        checks_passed += 1
    else:
        print(f"   ❌ LANGCHAIN_ENDPOINT is '{endpoint}'")
        print("   💡 Set LANGCHAIN_ENDPOINT=https://api.smith.langchain.com")
    
    # Check 4: LANGCHAIN_PROJECT
    print("\n4️⃣  Checking LANGCHAIN_PROJECT...")
    project = os.getenv('LANGCHAIN_PROJECT', '')
    if project:
        print(f"   ✅ LANGCHAIN_PROJECT is set to '{project}'")
        checks_passed += 1
    else:
        print("   ⚠️  LANGCHAIN_PROJECT is not set (will use 'default')")
        print("   💡 Set LANGCHAIN_PROJECT=Concept-Map-Generator in .env")
        checks_passed += 0.5
    
    # Check 5: langsmith package
    print("\n5️⃣  Checking langsmith package...")
    try:
        import langsmith
        print(f"   ✅ langsmith package installed (version: {langsmith.__version__})")
        checks_passed += 1
    except ImportError:
        print("   ❌ langsmith package not installed")
        print("   💡 Run: pip install langsmith")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("✅ LangSmith is fully configured!")
        print("🚀 You can now run: python main_universal.py")
        print("📊 View traces at: https://smith.langchain.com")
        return True
    elif checks_passed >= 3:
        print("⚠️  LangSmith is partially configured")
        print("💡 Fix the issues above for full functionality")
        return False
    else:
        print("❌ LangSmith is not properly configured")
        print("📚 See LANGSMITH_SETUP.md for detailed instructions")
        return False

if __name__ == "__main__":
    success = check_langsmith_setup()
    sys.exit(0 if success else 1)
