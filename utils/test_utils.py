"""
Test utilities for handling common test scenarios and rate limiting.
"""
import time
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

def handle_rate_limit(func: Callable, max_retries: int = 3, delay: float = 2.0) -> Any:
    """
    Handle rate limiting by retrying with exponential backoff.
    
    Args:
        func: Function to call
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        
    Returns:
        Function result or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            result = func()
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "too many requests" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    return None
            else:
                # Non-rate-limit error, don't retry
                logger.error(f"Non-rate-limit error: {e}")
                raise e
    
    return None

def is_rate_limited_error(error: Exception) -> bool:
    """Check if an error is due to rate limiting."""
    error_msg = str(error).lower()
    return "429" in error_msg or "too many requests" in error_msg

def skip_if_rate_limited(func: Callable) -> Any:
    """
    Decorator to skip tests gracefully if rate limited.
    
    Returns a test result indicating the test was skipped due to rate limiting.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if is_rate_limited_error(e):
                logger.warning(f"Test skipped due to rate limiting: {func.__name__}")
                return {
                    'passed': 0,
                    'failed': 0,
                    'skipped': 1,
                    'error': f"Rate limited: {str(e)}"
                }
            else:
                raise e
    return wrapper

def create_mock_api_response(data: dict) -> dict:
    """Create a mock API response for testing."""
    return {
        'status_code': 200,
        'data': data,
        'success': True
    }

def format_test_result(test_name: str, passed: bool, details: str = "", error: str = "") -> dict:
    """Format a standardized test result."""
    return {
        'test_name': test_name,
        'passed': passed,
        'failed': not passed,
        'details': details,
        'error': error,
        'timestamp': time.time()
    }
