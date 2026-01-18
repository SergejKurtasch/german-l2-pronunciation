#!/usr/bin/env python3
"""
Script to compare CTC and MFA validation results in detail.
Checks if results are truly identical or if there are differences.
"""

import json
import sys
from pathlib import Path

def extract_metrics_from_notebook(notebook_path):
    """Extract metrics from notebook output cells."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    metrics = {}
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            for output in outputs:
                if output.get('output_type') == 'stream':
                    text = ''.join(output.get('text', []))
                    
                    # Extract key metrics
                    if 'PER Improvement:' in text:
                        for line in text.split('\n'):
                            if 'PER Improvement:' in line:
                                try:
                                    value = line.split('PER Improvement:')[1].strip().rstrip('%')
                                    metrics['per_improvement'] = float(value)
                                except:
                                    pass
                            if 'Accuracy Before:' in line:
                                try:
                                    value = line.split('Accuracy Before:')[1].strip().rstrip('%')
                                    metrics['accuracy_before'] = float(value)
                                except:
                                    pass
                            if 'Accuracy After:' in line:
                                try:
                                    value = line.split('Accuracy After:')[1].strip().rstrip('%')
                                    metrics['accuracy_after'] = float(value)
                                except:
                                    pass
                            if 'Accuracy Improvement:' in line:
                                try:
                                    value = line.split('Accuracy Improvement:')[1].strip().rstrip('%')
                                    metrics['accuracy_improvement'] = float(value)
                                except:
                                    pass
                            if 'Total validated pairs:' in line:
                                try:
                                    value = line.split('Total validated pairs:')[1].strip()
                                    metrics['total_validated'] = int(value)
                                except:
                                    pass
                            if 'Total corrected errors:' in line:
                                try:
                                    value = line.split('Total corrected errors:')[1].strip()
                                    metrics['total_corrected'] = int(value)
                                except:
                                    pass
                            if 'Correction rate:' in line:
                                try:
                                    value = line.split('Correction rate:')[1].strip().rstrip('%')
                                    metrics['correction_rate'] = float(value)
                                except:
                                    pass
                    
                    # Also check for baseline metrics
                    if 'PER: ' in text and 'BASELINE' in text:
                        for line in text.split('\n'):
                            if 'PER: ' in line and '%' in line:
                                try:
                                    value = line.split('PER:')[1].strip().rstrip('%')
                                    metrics['per_before'] = float(value)
                                except:
                                    pass
                            if 'Accuracy: ' in line and '%' in line:
                                try:
                                    value = line.split('Accuracy:')[1].strip().rstrip('%')
                                    if 'accuracy_before' not in metrics:
                                        metrics['accuracy_before_baseline'] = float(value)
                                except:
                                    pass
    
    return metrics

def main():
    base_path = Path(__file__).parent / "notebooks"
    ctc_path = base_path / 'ctc_validation_phoneme_analysis.ipynb'
    mfa_path = base_path / 'mfa_validation_phoneme_analysis.ipynb'
    
    print("=" * 80)
    print("COMPARING CTC vs MFA VALIDATION RESULTS")
    print("=" * 80)
    print()
    
    ctc_metrics = extract_metrics_from_notebook(ctc_path)
    mfa_metrics = extract_metrics_from_notebook(mfa_path)
    
    print("CTC Metrics:")
    print(json.dumps(ctc_metrics, indent=2))
    print()
    
    print("MFA Metrics:")
    print(json.dumps(mfa_metrics, indent=2))
    print()
    
    print("=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    
    all_keys = set(ctc_metrics.keys()) | set(mfa_metrics.keys())
    differences = []
    identical = []
    
    for key in sorted(all_keys):
        ctc_val = ctc_metrics.get(key, 'N/A')
        mfa_val = mfa_metrics.get(key, 'N/A')
        
        if ctc_val == 'N/A' or mfa_val == 'N/A':
            status = "MISSING"
            print(f"{key:30s} | CTC: {str(ctc_val):15s} | MFA: {str(mfa_val):15s} | {status}")
        elif isinstance(ctc_val, (int, float)) and isinstance(mfa_val, (int, float)):
            diff = abs(ctc_val - mfa_val)
            if diff < 0.01:  # Consider identical if difference < 0.01
                status = "IDENTICAL"
                identical.append(key)
            else:
                status = f"DIFFERENT (diff: {diff:.4f})"
                differences.append((key, ctc_val, mfa_val, diff))
            print(f"{key:30s} | CTC: {ctc_val:15.4f} | MFA: {mfa_val:15.4f} | {status}")
        else:
            if ctc_val == mfa_val:
                status = "IDENTICAL"
                identical.append(key)
            else:
                status = "DIFFERENT"
                differences.append((key, ctc_val, mfa_val, None))
            print(f"{key:30s} | CTC: {str(ctc_val):15s} | MFA: {str(mfa_val):15s} | {status}")
    
    print()
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Identical metrics: {len(identical)}")
    print(f"Different metrics: {len(differences)}")
    
    if differences:
        print()
        print("DIFFERENCES FOUND:")
        for key, ctc_val, mfa_val, diff in differences:
            if diff is not None:
                print(f"  {key}: CTC={ctc_val}, MFA={mfa_val}, diff={diff:.6f}")
            else:
                print(f"  {key}: CTC={ctc_val}, MFA={mfa_val}")
    else:
        print()
        print("⚠️  WARNING: All metrics are identical!")
        print("This is suspicious - the results should differ if different alignment methods are used.")
        print("Possible causes:")
        print("  1. Results were copied from one notebook to another")
        print("  2. Both notebooks use the same alignment method")
        print("  3. The alignment method doesn't affect the final validation results")
        print("  4. The notebooks were run on the same data with cached results")

if __name__ == '__main__':
    main()
