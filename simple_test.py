#!/usr/bin/env python3

print("Starting simple test...")

try:
    import sys
    print("✓ sys imported")
    
    import os  
    print("✓ os imported")
    
    import time
    print("✓ time imported")
    
    # Test basic file operations
    if os.path.exists('/workspaces/crypto-refactor'):
        print("✓ workspace exists")
    else:
        print("✗ workspace missing")
        
    # Test our modules
    sys.path.append('/workspaces/crypto-refactor')
    
    from utils.logging_utils import get_logger
    print("✓ get_logger imported")
    
    logger = get_logger('test')
    print("✓ logger created")
    
    logger.info("Test log message")
    print("✓ logger works")
    
    print("🎉 ALL BASIC TESTS PASSED")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
