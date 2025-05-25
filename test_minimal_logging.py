#!/usr/bin/env python3
"""
Minimal logging test to isolate hanging issues.
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_minimal_logging():
    """Test minimal logging functionality."""
    print("🔍 MINIMAL LOGGING TEST")
    print("="*50)
    
    try:
        # Test basic logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Basic logging test successful")
        print("✅ Basic logging works")
        
        # Test logging utils import
        try:
            from utils.logging_utils import get_logger
            logger2 = get_logger(__name__)
            logger2.info("Custom logging utility test successful")
            print("✅ Custom logging utils work")
        except Exception as e:
            print(f"❌ Custom logging utils failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Minimal logging test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_minimal_logging()
    if success:
        print("🎉 Minimal logging test passed!")
        sys.exit(0)
    else:
        print("❌ Minimal logging test failed!")
        sys.exit(1)
