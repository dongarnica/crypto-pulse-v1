#!/usr/bin/env python3
"""
Simple test runner to debug hanging issues.
"""

import os
import sys
import subprocess
import time

def run_single_test(test_path):
    """Run a single test file and report results."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {test_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test with timeout
        result = subprocess.run(
            [sys.executable, test_path],
            cwd="/workspaces/crypto-refactor",
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        duration = time.time() - start_time
        
        print(f"Exit Code: {result.returncode}")
        print(f"Duration: {duration:.2f}s")
        
        if result.stdout:
            print(f"\nSTDOUT:")
            print(result.stdout[:1000])  # First 1000 chars
            
        if result.stderr:
            print(f"\nSTDERR:")
            print(result.stderr[:1000])  # First 1000 chars
            
        return {
            'test_path': test_path,
            'exit_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ TEST TIMEOUT after 30 seconds")
        return {
            'test_path': test_path,
            'exit_code': -1,
            'duration': 30.0,
            'stdout': '',
            'stderr': 'Timeout after 30 seconds',
            'success': False
        }
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")
        return {
            'test_path': test_path,
            'exit_code': -2,
            'duration': time.time() - start_time,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }

def main():
    """Run diagnostic tests."""
    
    print("🔍 CRYPTO TRADING BOT - DIAGNOSTIC TEST RUNNER")
    print("=" * 80)
    
    # List of tests to run
    tests = [
        "tests/test_config_usage.py",
        "tests/test_crypto.py",
        "tests/test_historical_data.py",
        "tests/test_alpaca_client.py",
        "tests/test_logging_comprehensive.py"
    ]
    
    results = []
    
    for test_path in tests:
        full_path = f"/workspaces/crypto-refactor/{test_path}"
        if os.path.exists(full_path):
            result = run_single_test(test_path)
            results.append(result)
        else:
            print(f"⚠️ Test not found: {test_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"Tests Run: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    print(f"\n📋 DETAILED RESULTS:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['test_path']:<40} ({result['duration']:.2f}s)")
        if not result['success']:
            error_line = result['stderr'].split('\n')[0] if result['stderr'] else "Unknown error"
            print(f"    └─ {error_line}")

if __name__ == "__main__":
    main()
