#!/usr/bin/env python3
"""
Enhanced comprehensive test runner for the crypto trading bot.
Includes all enhanced test modules with structured output reporting.
"""

import os
import sys
import time
import subprocess
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_default_logging, get_logger

class EnhancedTestRunner:
    """Enhanced test runner that executes comprehensive test modules."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.test_results = []
        
    def run_test_module(self, module_name: str, module_path: str) -> Dict[str, Any]:
        """Run a test module and capture its results."""
        self.logger.info(f"Running test module: {module_name}")
        
        print(f"\n{'='*80}")
        print(f"🧪 RUNNING MODULE: {module_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Run the test module as a subprocess
            result = subprocess.run(
                [sys.executable, module_path],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            duration = time.time() - start_time
            
            # Parse the result
            test_result = {
                'module_name': module_name,
                'module_path': module_path,
                'duration': duration,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0,
                'failed': result.returncode != 0
            }
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            # Print status
            if result.returncode == 0:
                status_icon = "✅"
                status_text = "PASSED"
                self.logger.info(f"Module {module_name} PASSED in {duration:.3f}s")
            else:
                status_icon = "❌"
                status_text = "FAILED"
                self.logger.error(f"Module {module_name} FAILED in {duration:.3f}s")
            
            print(f"\n{status_icon} MODULE {status_text}: {module_name}")
            print(f"   Duration: {duration:.3f}s")
            print(f"   Exit Code: {result.returncode}")
            
            self.test_results.append(test_result)
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = "Test module timed out after 60 seconds"
            
            test_result = {
                'module_name': module_name,
                'module_path': module_path,
                'duration': duration,
                'exit_code': -1,
                'stdout': '',
                'stderr': error_msg,
                'passed': False,
                'failed': True
            }
            
            print(f"\n❌ MODULE TIMEOUT: {module_name}")
            print(f"   Duration: {duration:.3f}s")
            print(f"   Error: {error_msg}")
            
            self.logger.error(f"Module {module_name} TIMED OUT after {duration:.3f}s")
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            test_result = {
                'module_name': module_name,
                'module_path': module_path,
                'duration': duration,
                'exit_code': -1,
                'stdout': '',
                'stderr': error_msg,
                'passed': False,
                'failed': True
            }
            
            print(f"\n❌ MODULE ERROR: {module_name}")
            print(f"   Duration: {duration:.3f}s")
            print(f"   Error: {error_msg}")
            
            self.logger.error(f"Module {module_name} ERROR after {duration:.3f}s: {error_msg}")
            self.test_results.append(test_result)
            return test_result
    
    def run_enhanced_test_suite(self) -> Dict[str, Any]:
        """Run the enhanced comprehensive test suite."""
        
        print("="*100)
        print("🚀 CRYPTO TRADING BOT - ENHANCED COMPREHENSIVE TEST SUITE")
        print("="*100)
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define enhanced test modules
        test_modules = [
            ("Configuration Usage Tests", "tests/test_config_usage.py"),
            ("Historical Data Tests", "tests/test_historical_data.py"),
            ("Mountain Time Tests", "tests/test_mountain_time.py"),
            ("Crypto Data Tests", "tests/test_crypto.py"),
            ("Alpaca Positions Tests", "tests/test_alpaca_positions.py"),
            ("Alpaca Client Tests", "tests/test_alpaca_client.py"),
            ("Alpaca Comprehensive Tests", "tests/test_alpaca_comprehensive.py"),
            ("Logging Comprehensive Tests", "tests/test_logging_safe.py")
        ]
        
        start_time = time.time()
        
        self.logger.info(f"Starting enhanced test suite with {len(test_modules)} modules")
        print(f"Running {len(test_modules)} test modules...")
        
        # Run all test modules
        for module_name, module_path in test_modules:
            try:
                # Check if the module file exists
                full_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    module_path
                )
                
                if os.path.exists(full_path):
                    self.run_test_module(module_name, full_path)
                else:
                    self.logger.warning(f"Test module not found: {full_path}")
                    print(f"\n⚠️ MODULE NOT FOUND: {module_name}")
                    print(f"   Path: {module_path}")
                    
                    # Add a failed result for missing module
                    self.test_results.append({
                        'module_name': module_name,
                        'module_path': module_path,
                        'duration': 0,
                        'exit_code': -1,
                        'stdout': '',
                        'stderr': f"Module not found: {full_path}",
                        'passed': False,
                        'failed': True
                    })
                    
            except KeyboardInterrupt:
                self.logger.warning("Test suite interrupted by user")
                print("\n⚠️ Test suite interrupted by user")
                break
        
        total_duration = time.time() - start_time
        
        # Calculate results
        passed_modules = [r for r in self.test_results if r['passed']]
        failed_modules = [r for r in self.test_results if r['failed']]
        
        # Print comprehensive summary
        print("\n" + "="*100)
        print("📊 ENHANCED TEST SUITE RESULTS")
        print("="*100)
        
        print(f"Total Modules Run: {len(self.test_results)}")
        print(f"Modules Passed: {len(passed_modules)}")
        print(f"Modules Failed: {len(failed_modules)}")
        print(f"Success Rate: {(len(passed_modules) / len(self.test_results) * 100):.1f}%" if self.test_results else "N/A")
        print(f"Total Duration: {total_duration:.3f}s")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 80)
        
        for result in self.test_results:
            status_icon = "✅" if result['passed'] else "❌"
            print(f"{status_icon} {result['module_name']:<40} ({result['duration']:.3f}s)")
            
            if result['failed'] and result['stderr']:
                error_lines = result['stderr'].split('\n')
                for line in error_lines[:2]:  # Show first 2 error lines
                    if line.strip():
                        print(f"    └─ {line.strip()}")
        
        # Failed modules summary
        if failed_modules:
            print(f"\n❌ FAILED MODULES SUMMARY:")
            print("-" * 50)
            for result in failed_modules:
                print(f"• {result['module_name']}")
                if result['stderr']:
                    error_line = result['stderr'].split('\n')[0]
                    print(f"  Error: {error_line}")
        
        print("\n" + "="*100)
        
        if len(failed_modules) == 0:
            print("🎉 ALL ENHANCED TESTS PASSED! The crypto trading bot is fully functional.")
        elif len(passed_modules) > 0:
            print(f"⚠️ PARTIAL SUCCESS: {len(passed_modules)}/{len(self.test_results)} modules passed.")
            print("   Please review failed modules and fix any issues.")
        else:
            print("🚨 ALL MODULES FAILED: Please check the configuration and error messages above.")
        
        print("="*100)
        
        return {
            'total_modules': len(self.test_results),
            'passed_modules': len(passed_modules),
            'failed_modules': len(failed_modules),
            'total_duration': total_duration,
            'results': self.test_results
        }


def main():
    """Main function to run the enhanced test suite."""
    # Setup logging
    setup_default_logging()
    
    try:
        # Initialize enhanced test runner
        runner = EnhancedTestRunner()
        
        # Run enhanced test suite
        final_results = runner.run_enhanced_test_suite()
        
        # Exit with appropriate code
        if final_results['failed_modules'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Enhanced test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
