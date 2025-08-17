#!/usr/bin/env python3
"""
Master script to run all dataset testing scripts
Runs tests for all four CSV files and provides a comprehensive summary
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_test_script(script_name, description):
    """Run a single test script and capture its output"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"SCRIPT: {script_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the script and capture output
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check if script ran successfully
        success = result.returncode == 0
        
        elapsed_time = time.time() - start_time
        
        return {
            'script': script_name,
            'description': description,
            'success': success,
            'return_code': result.returncode,
            'elapsed_time': elapsed_time,
            'output': result.stdout,
            'error': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {script_name} took too long to run (>5 minutes)")
        return {
            'script': script_name,
            'description': description,
            'success': False,
            'return_code': -1,
            'elapsed_time': 300,
            'output': '',
            'error': 'Script timed out after 5 minutes'
        }
    except Exception as e:
        print(f"❌ ERROR running {script_name}: {e}")
        return {
            'script': script_name,
            'description': description,
            'success': False,
            'return_code': -1,
            'elapsed_time': time.time() - start_time,
            'output': '',
            'error': str(e)
        }

def main():
    """Main function to run all tests"""
    print("🚀 DATASET VALIDATION SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Define all tests to run
    tests = [
        ("test_paired_dataset.py", "Paired Dataset (mask + past + present images)"),
        ("test_unpaired_pairs.py", "Unpaired Pairs (past + present images only)"),
        ("test_val_dataset.py", "Validation Dataset (mask + past + present images)"),
        ("test_combined_dataset.py", "Combined Dataset (labeled + unlabeled samples)")
    ]
    
    # Check if all test scripts exist
    missing_scripts = []
    for script_name, _ in tests:
        if not os.path.exists(script_name):
            missing_scripts.append(script_name)
    
    if missing_scripts:
        print(f"\n❌ ERROR: Missing test scripts: {missing_scripts}")
        print("Please ensure all test scripts are in the current directory.")
        return
    
    # Run all tests
    results = []
    total_start_time = time.time()
    
    for script_name, description in tests:
        result = run_test_script(script_name, description)
        results.append(result)
    
    total_elapsed_time = time.time() - total_start_time
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total execution time: {total_elapsed_time:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = len(results) - successful_tests
    
    print(f"\nTest Results:")
    print(f"  ✅ Successful: {successful_tests}/{len(results)}")
    print(f"  ❌ Failed: {failed_tests}/{len(results)}")
    print(f"  📈 Success Rate: {(successful_tests/len(results))*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status} {result['description']}")
        print(f"    Time: {result['elapsed_time']:.2f}s | Return Code: {result['return_code']}")
        if result['error']:
            print(f"    Error: {result['error']}")
    
    # Overall assessment
    print(f"\n{'='*80}")
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED! Your datasets are ready for use.")
    else:
        print(f"⚠️  {failed_tests} test(s) failed. Please review the errors above.")
        print("Recommendations:")
        print("  1. Check file paths in the CSV files")
        print("  2. Ensure all referenced files exist")
        print("  3. Verify image files are not corrupted")
        print("  4. Check for NaN/Inf values in image data")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
