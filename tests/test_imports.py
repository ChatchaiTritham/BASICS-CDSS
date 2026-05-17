"""
Comprehensive module import test for BASICS-CDSS
Tests all 7 core modules and their submodules
"""
import sys
import traceback

def import_module(module_name):
    """Test importing a module and return status."""
    try:
        module = __import__(module_name, fromlist=[''])
        items = [item for item in dir(module) if not item.startswith('_')]
        print(f"[OK] {module_name:30s} - {len(items):3d} exported items")
        return True, len(items)
    except Exception as e:
        print(f"[FAIL] {module_name:30s} - Error: {str(e)}")
        traceback.print_exc()
        return False, 0

def main():
    print("=" * 80)
    print("BASICS-CDSS Module Import Test")
    print("=" * 80)
    print()

    # Test main package
    print("1. Main Package Import")
    print("-" * 80)
    success, count = import_module('basics_cdss')
    print()

    # Test all submodules
    print("2. Submodule Imports")
    print("-" * 80)

    modules_to_test = [
        'basics_cdss.scenario',
        'basics_cdss.metrics',
        'basics_cdss.governance',
        'basics_cdss.visualization',
        'basics_cdss.temporal',
        'basics_cdss.causal',
        'basics_cdss.multiagent',
    ]

    results = {}
    total_items = 0

    for module_name in modules_to_test:
        success, count = import_module(module_name)
        results[module_name] = (success, count)
        total_items += count

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    passed = sum(1 for s, _ in results.values() if s)
    total = len(results)

    print(f"Modules tested: {total}")
    print(f"Modules passed: {passed}")
    print(f"Modules failed: {total - passed}")
    print(f"Total exported items: {total_items}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print()

    if passed == total:
        print("[SUCCESS] All modules imported successfully!")
        return 0
    else:
        print("[FAILURE] Some modules failed to import")
        return 1

if __name__ == '__main__':
    sys.exit(main())
