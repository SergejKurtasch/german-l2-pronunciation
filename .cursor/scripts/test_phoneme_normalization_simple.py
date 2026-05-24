"""
Simple unit tests for phoneme normalization.
Tests the normalizer directly without loading large dictionaries.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.phoneme_normalizer import get_phoneme_normalizer


def test_phoneme_mappings():
    """Test that phoneme mappings work correctly."""
    print("\n" + "="*80)
    print("TEST 1: Phoneme Mappings (ˑ→ː, ʀ→ʁ, ˀ→ʔ, ʧ→tʃ, g→ɡ)")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    tests = [
        ("aˑ", "aː", "Half-long to long"),
        ("ʀaː", "ʁaː", "Uvular trill to fricative"),
        ("ˀa", "ʔa", "Glottal stop modifier to glottal stop"),
        ("ʧa", "tʃa", "Affricate ʧ to tʃ"),
        ("ga", "ɡa", "Latin g to IPA g"),
    ]
    
    passed = 0
    for input_str, expected, description in tests:
        result = normalizer.normalize_phoneme_string(input_str, source='dictionary')
        if result == expected:
            print(f"✓ {description}: '{input_str}' → '{result}'")
            passed += 1
        else:
            print(f"✗ {description}: '{input_str}' → '{result}' (expected '{expected}')")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_invalid_pattern_rejection():
    """Test that invalid patterns are rejected."""
    print("\n" + "="*80)
    print("TEST 2: Invalid Pattern Rejection")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    tests = [
        ("ˈkɔmpɑQNOU?", "", "QNOU pattern"),
        ("ˈkɔmpɑQOU", "", "QOU pattern"),
        ("ABC123", "", "3+ capital letters"),
        ("ˈʔaːlə", "ʔaːlə", "Valid pattern (should pass)"),
    ]
    
    passed = 0
    for input_str, expected, description in tests:
        result = normalizer.normalize_phoneme_string(input_str, source='dictionary')
        if result == expected:
            print(f"✓ {description}: '{input_str}' → '{result}'")
            passed += 1
        else:
            print(f"✗ {description}: '{input_str}' → '{result}' (expected '{expected}')")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_diacritic_removal():
    """Test that diacritics not in model are removed."""
    print("\n" + "="*80)
    print("TEST 3: Diacritic Removal")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    tests = [
        ("a̯", "a", "Remove ̯ (U+032F)"),
        ("a͡", "a", "Remove ͡ (U+0361)"),
        ("a͜", "a", "a͜", "Remove ͜ (U+035C)"),
        ("ă", "a", "Remove ̆ (U+0306)"),
        ("aː", "aː", "Keep ː (in model)"),
    ]
    
    passed = 0
    for input_str, expected, description in tests:
        result = normalizer.normalize_phoneme_string(input_str, source='dictionary')
        if result == expected:
            print(f"✓ {description}: '{input_str}' → '{result}'")
            passed += 1
        else:
            print(f"✗ {description}: '{input_str}' → '{result}' (expected '{expected}')")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_stress_mark_removal():
    """Test that stress marks are removed."""
    print("\n" + "="*80)
    print("TEST 4: Stress Mark Removal")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    tests = [
        ("ˈaː", "aː", "Remove primary stress ˈ"),
        ("ˌaː", "aː", "Remove secondary stress ˌ"),
        ("ˈˌaː", "aː", "Remove both stress marks"),
    ]
    
    passed = 0
    for input_str, expected, description in tests:
        result = normalizer.normalize_phoneme_string(input_str, source='dictionary')
        if result == expected:
            print(f"✓ {description}: '{input_str}' → '{result}'")
            passed += 1
        else:
            print(f"✗ {description}: '{input_str}' → '{result}' (expected '{expected}')")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_character_removal():
    """Test that characters not in model are removed."""
    print("\n" + "="*80)
    print("TEST 5: Character Removal")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    tests = [
        ("~aː", "aː", "Remove ~ placeholder"),
        ("'aː", "aː", "Remove apostrophe '"),
        ("ʧaː", "tʃaː", "Convert ʧ to tʃ"),
        ("ʀaː", "ʁaː", "Convert ʀ to ʁ"),
    ]
    
    passed = 0
    for input_str, expected, description in tests:
        result = normalizer.normalize_phoneme_string(input_str, source='dictionary')
        if result == expected:
            print(f"✓ {description}: '{input_str}' → '{result}'")
            passed += 1
        else:
            print(f"✗ {description}: '{input_str}' → '{result}' (expected '{expected}')")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_phoneme_list_normalization():
    """Test normalization of phoneme lists."""
    print("\n" + "="*80)
    print("TEST 6: Phoneme List Normalization")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    # Test list normalization
    input_list = ["ˈa", "pf", "ə", "l"]
    expected = ["a", "pf", "ə", "l"]  # Stress removed, but 'pf' kept as is
    
    result = normalizer.normalize_phoneme_list(input_list, source='dictionary')
    
    print(f"Input:    {input_list}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    
    if result == expected:
        print("✓ PASSED: Phoneme list correctly normalized")
        return True
    else:
        print("✗ FAILED: Phoneme list normalization failed")
        return False


def run_all_tests():
    """Run all phoneme normalization tests."""
    print("\n" + "="*80)
    print("PHONEME NORMALIZATION TESTS (Simple Version)")
    print("="*80)
    print("Testing normalizer directly without loading DSL dictionary")
    print("="*80)
    
    tests = [
        test_phoneme_mappings,
        test_invalid_pattern_rejection,
        test_diacritic_removal,
        test_stress_mark_removal,
        test_character_removal,
        test_phoneme_list_normalization,
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test in tests:
        try:
            if test():
                passed_tests += 1
            else:
                failed_tests += 1
        except Exception as e:
            failed_tests += 1
            print(f"✗ ERROR in {test.__name__}: {e}")
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
