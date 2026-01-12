"""
Unit tests for phoneme normalization.
Tests the implementation according to the plan specifications.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.phoneme_normalizer import get_phoneme_normalizer
from modules.g2p_module import DSLG2P


def test_affricate_separation():
    """Test that affricates are separated into individual phonemes."""
    print("\n" + "="*80)
    print("TEST 1: Affricate Separation (pf)")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    # Test input: /ˈapfəl/ (Apfel)
    # Expected: ['a', 'p', 'f', 'ə', 'l']
    test_input = "/ˈapfəl/"
    
    # Simulate DSL parsing
    cleaned = test_input.strip('/').replace('ˈ', '')
    
    # Parse manually to see what happens
    from modules.g2p_module import DSLG2P
    dsl = DSLG2P(
        dsl_path=project_root / "data" / "dictionaries" / "de_ipa.dsl"
    )
    
    # Normalize using the DSL method
    result = dsl._normalize_ipa_transcription(test_input)
    
    print(f"Input:    {test_input}")
    print(f"Result:   {result}")
    print(f"Expected: ['a', 'p', 'f', 'ə', 'l']")
    
    # Check that pf is separated
    assert 'pf' not in result, "Affricate 'pf' should be separated"
    assert 'p' in result and 'f' in result, "Both 'p' and 'f' should be present"
    
    print("✓ PASSED: Affricate correctly separated")
    return True


def test_half_long_conversion():
    """Test that half-long marker ˑ is converted to ː."""
    print("\n" + "="*80)
    print("TEST 2: Half-Long to Long Conversion (ˑ → ː)")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    # Test input: /kaˑfə/ (with half-long)
    # Expected: ['k', 'aː', 'f', 'ə']
    test_input = "/kaˑfə/"
    
    dsl = DSLG2P(
        dsl_path=project_root / "data" / "dictionaries" / "de_ipa.dsl"
    )
    result = dsl._normalize_ipa_transcription(test_input)
    
    print(f"Input:    {test_input}")
    print(f"Result:   {result}")
    print(f"Expected: ['k', 'aː', 'f', 'ə']")
    
    # Check that ˑ is converted to ː
    assert 'ˑ' not in str(result), "Half-long marker ˑ should be converted"
    assert 'aː' in result, "Long vowel 'aː' should be present"
    
    print("✓ PASSED: Half-long correctly converted to long")
    return True


def test_aspirated_preservation():
    """Test that aspirated consonants are preserved as 2 characters."""
    print("\n" + "="*80)
    print("TEST 3: Aspirated Consonant Preservation (tʰ)")
    print("="*80)
    
    # Test input: /tʰoːn/
    # Expected: ['tʰ', 'oː', 'n']
    test_input = "/tʰoːn/"
    
    dsl = DSLG2P(
        dsl_path=project_root / "data" / "dictionaries" / "de_ipa.dsl"
    )
    result = dsl._normalize_ipa_transcription(test_input)
    
    print(f"Input:    {test_input}")
    print(f"Result:   {result}")
    print(f"Expected: ['tʰ', 'oː', 'n']")
    
    # Check that tʰ is kept as single unit
    assert 'tʰ' in result, "Aspirated consonant 'tʰ' should be preserved"
    
    print("✓ PASSED: Aspirated consonant preserved")
    return True


def test_diphthong_normalization():
    """Test that diphthongs with non-syllabic marker are normalized."""
    print("\n" + "="*80)
    print("TEST 4: Diphthong Normalization (aʊ̯ → aʊ)")
    print("="*80)
    
    # Test input: /haʊ̯s/
    # Expected: ['h', 'aʊ', 's']
    test_input = "/haʊ̯s/"
    
    dsl = DSLG2P(
        dsl_path=project_root / "data" / "dictionaries" / "de_ipa.dsl"
    )
    result = dsl._normalize_ipa_transcription(test_input)
    
    print(f"Input:    {test_input}")
    print(f"Result:   {result}")
    print(f"Expected: ['h', 'aʊ', 's']")
    
    # Check that ̯ is removed
    result_str = ''.join(result)
    assert '̯' not in result_str, "Non-syllabic marker ̯ should be removed"
    assert 'aʊ' in result, "Diphthong 'aʊ' should be preserved"
    
    print("✓ PASSED: Diphthong correctly normalized")
    return True


def test_invalid_pattern_rejection():
    """Test that invalid patterns (QNOU?) are rejected."""
    print("\n" + "="*80)
    print("TEST 5: Invalid Pattern Rejection (QNOU?)")
    print("="*80)
    
    normalizer = get_phoneme_normalizer()
    
    # Test input: /ˈkɔmpɑQNOU?/
    # Expected: empty or rejected
    test_input = "/ˈkɔmpɑQNOU?/"
    
    # Normalize using the normalizer directly
    result = normalizer.normalize_phoneme_string(test_input, source='dictionary')
    
    print(f"Input:    {test_input}")
    print(f"Result:   '{result}'")
    print(f"Expected: '' (empty, rejected)")
    
    # Check that result is empty (rejected)
    assert result == "", "Invalid pattern should be rejected (empty string)"
    
    print("✓ PASSED: Invalid pattern correctly rejected")
    return True


def test_glottal_stop_preservation():
    """Test that glottal stop ʔ is preserved."""
    print("\n" + "="*80)
    print("TEST 6: Glottal Stop Preservation (ʔ)")
    print("="*80)
    
    # Test input: /ˈʔaːlə/
    # Expected: ['ʔ', 'aː', 'l', 'ə']
    test_input = "/ˈʔaːlə/"
    
    dsl = DSLG2P(
        dsl_path=project_root / "data" / "dictionaries" / "de_ipa.dsl"
    )
    result = dsl._normalize_ipa_transcription(test_input)
    
    print(f"Input:    {test_input}")
    print(f"Result:   {result}")
    print(f"Expected: ['ʔ', 'aː', 'l', 'ə']")
    
    # Check that ʔ is preserved
    assert 'ʔ' in result, "Glottal stop 'ʔ' should be preserved"
    
    print("✓ PASSED: Glottal stop preserved")
    return True


def run_all_tests():
    """Run all phoneme normalization tests."""
    print("\n" + "="*80)
    print("PHONEME NORMALIZATION TESTS")
    print("="*80)
    print(f"Testing implementation according to plan specifications")
    print("="*80)
    
    tests = [
        test_affricate_separation,
        test_half_long_conversion,
        test_aspirated_preservation,
        test_diphthong_normalization,
        test_invalid_pattern_rejection,
        test_glottal_stop_preservation,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((test.__name__, f"ERROR: {str(e)}"))
            print(f"✗ ERROR: {e}")
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if errors:
        print("\nFailed tests:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")
    
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
