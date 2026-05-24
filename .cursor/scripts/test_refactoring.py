"""
Тесты для проверки рефакторинга кода.
Проверяет импорты, экспорты и функциональность модулей.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_module_imports():
    """Проверка импорта всех рефакторенных модулей."""
    print("\n" + "="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    tests = []
    
    # Test utils module
    try:
        from modules.utils import collapse_consecutive_duplicates
        print("✓ modules.utils imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"✗ modules.utils import failed: {e}")
        tests.append(False)
    
    # Test chat_utils module
    try:
        from modules.chat_utils import normalize_chat_history, add_to_chat_history
        print("✓ modules.chat_utils imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"✗ modules.chat_utils import failed: {e}")
        tests.append(False)
    
    # Test visualization package
    try:
        from modules.visualization import (
            align_graphemes_to_phonemes,
            create_side_by_side_comparison,
            create_colored_text,
            create_text_comparison_view,
            create_raw_phonemes_display,
            create_validation_comparison,
            create_detailed_report,
            create_simple_phoneme_comparison,
            create_text_with_sources_display,
            create_dual_model_comparison,
            create_triple_model_comparison,
            create_quadruple_model_comparison,
        )
        print("✓ modules.visualization package imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"✗ modules.visualization import failed: {e}")
        import traceback
        traceback.print_exc()
        tests.append(False)
    
    print(f"\nPassed: {sum(tests)}/{len(tests)}")
    return all(tests)


def test_utils_functionality():
    """Проверка функциональности modules.utils."""
    print("\n" + "="*80)
    print("TEST 2: Utils Functionality")
    print("="*80)
    
    from modules.utils import collapse_consecutive_duplicates
    
    tests = [
        (["a", "a", "b", "b", "c"], ["a", "b", "c"], "Basic collapse"),
        ([], [], "Empty list"),
        (["a"], ["a"], "Single element"),
        (["a", "b", "a"], ["a", "b", "a"], "Same element not consecutive"),
        (["a", "", "a"], ["a"], "Skip empty strings (empty string removes following duplicate)"),
    ]
    
    passed = 0
    for input_list, expected, description in tests:
        result = collapse_consecutive_duplicates(input_list)
        if result == expected:
            print(f"✓ {description}: {input_list} → {result}")
            passed += 1
        else:
            print(f"✗ {description}: {input_list} → {result} (expected {expected})")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_chat_utils_functionality():
    """Проверка функциональности modules.chat_utils."""
    print("\n" + "="*80)
    print("TEST 3: Chat Utils Functionality")
    print("="*80)
    
    from modules.chat_utils import normalize_chat_history, add_to_chat_history
    
    # Test normalize_chat_history
    tests_normalize = [
        (None, [], "None input"),
        ([], [], "Empty list"),
        ([{"role": "user", "content": "hi"}], [{"role": "user", "content": "hi"}], "Dict format"),
        ([["hi", "hello"]], [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "Tuple format"),
    ]
    
    passed = 0
    for input_val, expected, description in tests_normalize:
        result = normalize_chat_history(input_val)
        if result == expected:
            print(f"✓ normalize_chat_history - {description}")
            passed += 1
        else:
            print(f"✗ normalize_chat_history - {description}: got {result}, expected {expected}")
    
    # Test add_to_chat_history
    history = [{"role": "user", "content": "hi"}]
    result = add_to_chat_history(history, "test", "response")
    expected_len = 3  # original + user + assistant
    
    if len(result) == expected_len and result[-1]["content"] == "response":
        print(f"✓ add_to_chat_history: correctly added messages")
        passed += 1
    else:
        print(f"✗ add_to_chat_history: failed")
        passed += 0
    
    total_tests = len(tests_normalize) + 1
    print(f"\nPassed: {passed}/{total_tests}")
    return passed == total_tests


def test_visualization_exports():
    """Проверка экспорта функций из visualization."""
    print("\n" + "="*80)
    print("TEST 4: Visualization Exports")
    print("="*80)
    
    try:
        from modules.visualization import __all__
        from modules import visualization
        
        # Check that __all__ functions are callable
        failed = []
        for func_name in __all__:
            func = getattr(visualization, func_name, None)
            if func is None or not callable(func):
                failed.append(func_name)
            else:
                print(f"✓ {func_name} is exported and callable")
        
        if failed:
            print(f"\n✗ Failed exports: {failed}")
            return False
        else:
            print(f"\n✓ All {len(__all__)} functions are exported correctly")
            return True
    except Exception as e:
        print(f"✗ Visualization export check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_syntax_check():
    """Проверка синтаксиса всех рефакторенных файлов."""
    print("\n" + "="*80)
    print("TEST 5: Syntax Check")
    print("="*80)
    
    import py_compile
    
    files_to_check = [
        "modules/utils.py",
        "modules/chat_utils.py",
        "modules/visualization/__init__.py",
        "modules/visualization/helpers.py",
        "modules/visualization/html_generators.py",
        "modules/visualization/report_generators.py",
        "modules/visualization/multi_model_comparison.py",
    ]
    
    failed = []
    for file_path in files_to_check:
        full_path = project_root / file_path
        try:
            py_compile.compile(str(full_path), doraise=True)
            print(f"✓ {file_path} - syntax OK")
        except py_compile.PyCompileError as e:
            print(f"✗ {file_path} - syntax error: {e}")
            failed.append(file_path)
    
    print(f"\nPassed: {len(files_to_check) - len(failed)}/{len(files_to_check)}")
    return len(failed) == 0


def run_all_tests():
    """Запуск всех тестов рефакторинга."""
    print("\n" + "="*80)
    print("REFACTORING TESTS")
    print("="*80)
    print("Проверка импортов, экспортов и функциональности рефакторенных модулей")
    print("="*80)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Utils Functionality", test_utils_functionality),
        ("Chat Utils Functionality", test_chat_utils_functionality),
        ("Visualization Exports", test_visualization_exports),
        ("Syntax Check", test_syntax_check),
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                failed_tests += 1
        except Exception as e:
            failed_tests += 1
            print(f"✗ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print("="*80)
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
