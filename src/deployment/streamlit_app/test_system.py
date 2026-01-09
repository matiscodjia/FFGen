#!/usr/bin/env python3
"""
Test script for the cache system
"""

import os
from deepseek_caller import DeepSeekCaller
from stats_logger import StatsLogger
from config import DEEPSEEK_API_KEY

def test_deepseek_connection():
    """Test if DeepSeek API is accessible"""
    print("=" * 70)
    print("🧪 TEST 1: DeepSeek API Connection")
    print("=" * 70)

    if not DEEPSEEK_API_KEY:
        print(" DEEPSEEK_API_KEY not found in environment")
        print("   Set it with: export DEEPSEEK_API_KEY='your-key'")
        return False

    try:
        caller = DeepSeekCaller()
        is_connected = caller.test_connection()

        if is_connected:
            print(" DeepSeek API is accessible")
            return True
        else:
            print(" DeepSeek API test failed")
            return False

    except Exception as e:
        print(f" Error: {e}")
        return False

def test_feedback_generation():
    """Test feedback generation"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Feedback Generation")
    print("=" * 70)

    caller = DeepSeekCaller()

    test_context = {
        "theme": "Prime Number Check",
        "difficulty": "beginner",
        "error_category": "Incorrect Base Case Return Value",
        "instructions": "Write a function to check if a number is prime",
        "code": """
int is_prime(int n) {
    if (n <= 1) {
        return 1;  // Bug: should return 0
    }
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}
""",
        "test_cases_scope": ["Input: 1", "Input: 2", "Input: 17"],
        "failed_tests": ["Input: 1 (returns 1, expected 0)"]
    }

    print(" Generating feedback...")
    result = caller.generate_feedback(test_context)

    if result.get('feedback'):
        print(" Feedback generated successfully!")
        print("\n Metrics:")
        print(f"   Tokens (prompt): {result['tokens_prompt']}")
        print(f"   Tokens (completion): {result['tokens_completion']}")
        print(f"   Total tokens: {result['tokens_total']}")
        print(f"   Generation time: {result['generation_time_ms']:.0f} ms")
        print("\n Feedback:")
        print(f"   {result['feedback'][:200]}...")
        return True
    else:
        print(f" Error: {result.get('error')}")
        return False

def test_stats_logger():
    """Test stats logging"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Stats Logger")
    print("=" * 70)

    logger = StatsLogger()

    # Test query log
    test_query = {
        "query_id": "test-123",
        "status": "hit",
        "similarity_score": 0.15,
        "confidence": 0.95,
        "response_time_ms": 123.45,
        "theme": "Test Theme",
        "error_category": "Test Error",
        "difficulty": "beginner",
        "deepseek_tokens": 0,
        "cache_size": 100
    }

    try:
        logger.log_query(test_query)
        print(" Query logged successfully")

        # Read back
        stats = logger.read_stats(limit=1)
        if stats:
            print(f" Read back: {stats[-1]['query_id']}")
        else:
            print("  No stats found (empty file)")

        return True

    except Exception as e:
        print(f" Error: {e}")
        return False

def main():
    print(" TESTING CACHE SYSTEM COMPONENTS")
    print()

    results = []

    # Test 1: API Connection
    results.append(("DeepSeek API", test_deepseek_connection()))

    # Test 2: Feedback Generation
    if results[0][1]:  # Only if API works
        results.append(("Feedback Generation", test_feedback_generation()))
    else:
        print("\n⏭  Skipping feedback generation test (API unavailable)")

    # Test 3: Stats Logger
    results.append(("Stats Logger", test_stats_logger()))

    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"{status:12} {test_name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    print()
    print(f"Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print(" All tests passed! System is ready.")
        return 0
    else:
        print("  Some tests failed. Check configuration.")
        return 1

if __name__ == "__main__":
    exit(main())
