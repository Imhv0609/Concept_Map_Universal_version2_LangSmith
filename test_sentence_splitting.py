"""
Test script for improved sentence splitting
===========================================
Tests handling of:
1. No space after period (e.g., "Sentence1.Sentence2")
2. Titles (Mr., Mrs., Dr., etc.)
3. Abbreviations (U.S., Ph.D., etc.)
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from timeline_mapper import split_into_sentences


def test_sentence_splitting():
    """Test various edge cases in sentence splitting"""
    
    print("=" * 70)
    print("🧪 SENTENCE SPLITTING TEST")
    print("=" * 70)
    print()
    
    test_cases = [
        {
            "name": "No space after period",
            "input": "This is sentence one.This is sentence two.This is sentence three.",
            "expected_count": 3
        },
        {
            "name": "Titles (Mr., Mrs., Dr.)",
            "input": "Mr. Smith visited Mrs. Johnson. Dr. Brown was there too.",
            "expected_count": 2
        },
        {
            "name": "Mixed titles and no space",
            "input": "Dr. Einstein worked hard.He discovered relativity.Mr. Bohr agreed.",
            "expected_count": 3
        },
        {
            "name": "Abbreviations (U.S., Ph.D., etc.)",
            "input": "The U.S. has many Ph.D. holders. They work hard etc.",
            "expected_count": 2
        },
        {
            "name": "Normal spacing",
            "input": "First sentence. Second sentence. Third sentence.",
            "expected_count": 3
        },
        {
            "name": "Complex mix",
            "input": "Dr. Smith from the U.S. met Mr. Jones.They discussed Ph.D. programs.Prof. Lee joined later etc.",
            "expected_count": 3
        },
        {
            "name": "Question and exclamation marks",
            "input": "What is this?This is amazing!Who would have thought?",
            "expected_count": 3
        },
        {
            "name": "Real-world example",
            "input": "Photosynthesis converts light to energy.Plants use chlorophyll.Dr. Calvin discovered the cycle.",
            "expected_count": 3
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 70)
        print(f"Input: \"{test_case['input']}\"")
        print()
        
        result = split_into_sentences(test_case['input'])
        
        print(f"Found {len(result)} sentences:")
        for j, sentence in enumerate(result, 1):
            print(f"  {j}. \"{sentence}\"")
        
        print()
        
        if len(result) == test_case['expected_count']:
            print(f"✅ PASSED: Expected {test_case['expected_count']} sentences, got {len(result)}")
        else:
            print(f"❌ FAILED: Expected {test_case['expected_count']} sentences, got {len(result)}")
            all_passed = False
        
        print()
        print("=" * 70)
        print()
    
    # Final summary
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Check output above")
    
    print()
    return all_passed


if __name__ == "__main__":
    success = test_sentence_splitting()
    sys.exit(0 if success else 1)
