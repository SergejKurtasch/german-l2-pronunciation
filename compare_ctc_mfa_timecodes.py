#!/usr/bin/env python3
"""
Script to compare timecodes between CTC and MFA alignment methods.
Loads saved data from both notebooks and creates comparison visualizations.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add project to path
notebook_dir = Path('/Volumes/SSanDisk/SpeechRec-German-diagnostic/notebooks')
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root))

# Try to load data from notebooks
# We'll need to extract segment data from the notebooks

def extract_segments_from_notebook(notebook_path, method_name):
    """Extract segment timecodes from notebook output."""
    print(f"\n{'='*80}")
    print(f"Extracting {method_name} segments from notebook...")
    print(f"{'='*80}")
    
    # This is a placeholder - in reality, we'd need to run the notebooks
    # or load saved data. For now, we'll create a script that can be run
    # in the notebooks to save comparison data.
    
    return None

def create_timecode_comparison_script():
    """Create a script that can be added to both notebooks to compare timecodes."""
    
    script = '''
# =============================================================================
# COMPARISON: CTC vs MFA Timecode Analysis
# =============================================================================
# This cell compares timecodes between CTC and MFA alignment methods
# Add this to both notebooks after segment extraction

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def compare_timecodes_ctc_mfa(df_ctc, df_mfa, sample_size=50):
    """
    Compare timecodes between CTC and MFA alignment.
    
    Args:
        df_ctc: DataFrame from CTC notebook with 'recognized_segments'
        df_mfa: DataFrame from MFA notebook with 'recognized_segments'
        sample_size: Number of files to compare
    """
    
    # Sample files for comparison
    sample_indices = df_ctc.index[:sample_size]
    
    timecode_differences = []
    segment_counts = {'ctc': [], 'mfa': []}
    duration_differences = []
    
    for idx in sample_indices:
        if idx not in df_mfa.index:
            continue
            
        ctc_segments = df_ctc.loc[idx, 'recognized_segments']
        mfa_segments = df_mfa.loc[idx, 'recognized_segments']
        
        if not ctc_segments or not mfa_segments:
            continue
        
        # Count segments
        segment_counts['ctc'].append(len(ctc_segments))
        segment_counts['mfa'].append(len(mfa_segments))
        
        # Compare timecodes for matching phonemes
        # We'll match by position (assuming same phoneme sequence)
        recognized_ctc = df_ctc.loc[idx, 'recognized_phonemes']
        recognized_mfa = df_mfa.loc[idx, 'recognized_phonemes']
        
        if len(recognized_ctc) != len(recognized_mfa):
            continue
        
        # Compare segments for each phoneme
        min_len = min(len(ctc_segments), len(mfa_segments))
        for i in range(min_len):
            if i < len(ctc_segments) and i < len(mfa_segments):
                ctc_seg = ctc_segments[i]
                mfa_seg = mfa_segments[i]
                
                if hasattr(ctc_seg, 'start_time') and hasattr(mfa_seg, 'start_time'):
                    start_diff = abs(ctc_seg.start_time - mfa_seg.start_time)
                    end_diff = abs(ctc_seg.end_time - mfa_seg.end_time)
                    duration_ctc = ctc_seg.end_time - ctc_seg.start_time
                    duration_mfa = mfa_seg.end_time - mfa_seg.start_time
                    duration_diff = abs(duration_ctc - duration_mfa)
                    
                    timecode_differences.append({
                        'file_idx': idx,
                        'phoneme_idx': i,
                        'phoneme': recognized_ctc[i] if i < len(recognized_ctc) else '?',
                        'start_diff': start_diff,
                        'end_diff': end_diff,
                        'duration_diff': duration_diff,
                        'ctc_start': ctc_seg.start_time,
                        'mfa_start': mfa_seg.start_time,
                        'ctc_end': ctc_seg.end_time,
                        'mfa_end': mfa_seg.end_time,
                        'ctc_duration': duration_ctc,
                        'mfa_duration': duration_mfa
                    })
                    
                    duration_differences.append(duration_diff)
    
    return {
        'timecode_differences': timecode_differences,
        'segment_counts': segment_counts,
        'duration_differences': duration_differences
    }

# Create visualization function
def plot_timecode_comparison(comparison_data, save_path=None):
    """Plot comparison graphs."""
    
    timecode_diffs = comparison_data['timecode_differences']
    segment_counts = comparison_data['segment_counts']
    duration_diffs = comparison_data['duration_differences']
    
    if not timecode_diffs:
        print("No timecode differences found. Segments may be identical or data not available.")
        return
    
    df_diffs = pd.DataFrame(timecode_diffs)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CTC vs MFA Timecode Comparison', fontsize=16, fontweight='bold')
    
    # 1. Start time differences
    axes[0, 0].hist(df_diffs['start_diff'] * 1000, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Start Time Difference (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Start Time Differences')
    axes[0, 0].axvline(df_diffs['start_diff'].mean() * 1000, color='red', linestyle='--', 
                       label=f'Mean: {df_diffs["start_diff"].mean()*1000:.2f} ms')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. End time differences
    axes[0, 1].hist(df_diffs['end_diff'] * 1000, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('End Time Difference (ms)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of End Time Differences')
    axes[0, 1].axvline(df_diffs['end_diff'].mean() * 1000, color='red', linestyle='--',
                       label=f'Mean: {df_diffs["end_diff"].mean()*1000:.2f} ms')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Duration differences
    axes[0, 2].hist(df_diffs['duration_diff'] * 1000, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_xlabel('Duration Difference (ms)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Duration Differences')
    axes[0, 2].axvline(df_diffs['duration_diff'].mean() * 1000, color='red', linestyle='--',
                       label=f'Mean: {df_diffs["duration_diff"].mean()*1000:.2f} ms')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Segment count comparison
    axes[1, 0].bar(['CTC', 'MFA'], 
                   [np.mean(segment_counts['ctc']), np.mean(segment_counts['mfa'])],
                   color=['blue', 'green'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Average Number of Segments')
    axes[1, 0].set_title('Average Segment Count per File')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Scatter: CTC vs MFA start times
    sample_df = df_diffs.sample(min(1000, len(df_diffs)))
    axes[1, 1].scatter(sample_df['ctc_start'] * 1000, sample_df['mfa_start'] * 1000,
                      alpha=0.5, s=10)
    max_val = max(sample_df['ctc_start'].max(), sample_df['mfa_start'].max()) * 1000
    axes[1, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect match')
    axes[1, 1].set_xlabel('CTC Start Time (ms)')
    axes[1, 1].set_ylabel('MFA Start Time (ms)')
    axes[1, 1].set_title('Start Time: CTC vs MFA')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Scatter: CTC vs MFA durations
    axes[1, 2].scatter(sample_df['ctc_duration'] * 1000, sample_df['mfa_duration'] * 1000,
                      alpha=0.5, s=10, color='orange')
    max_dur = max(sample_df['ctc_duration'].max(), sample_df['mfa_duration'].max()) * 1000
    axes[1, 2].plot([0, max_dur], [0, max_dur], 'r--', label='Perfect match')
    axes[1, 2].set_xlabel('CTC Duration (ms)')
    axes[1, 2].set_ylabel('MFA Duration (ms)')
    axes[1, 2].set_title('Duration: CTC vs MFA')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\\n✓ Saved comparison plot to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\\n{'='*80}")
    print("TIMECODE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Total segments compared: {len(df_diffs)}")
    print(f"\\nStart Time Differences:")
    print(f"  Mean: {df_diffs['start_diff'].mean()*1000:.2f} ms")
    print(f"  Median: {df_diffs['start_diff'].median()*1000:.2f} ms")
    print(f"  Std: {df_diffs['start_diff'].std()*1000:.2f} ms")
    print(f"  Max: {df_diffs['start_diff'].max()*1000:.2f} ms")
    print(f"\\nEnd Time Differences:")
    print(f"  Mean: {df_diffs['end_diff'].mean()*1000:.2f} ms")
    print(f"  Median: {df_diffs['end_diff'].median()*1000:.2f} ms")
    print(f"  Std: {df_diffs['end_diff'].std()*1000:.2f} ms")
    print(f"  Max: {df_diffs['end_diff'].max()*1000:.2f} ms")
    print(f"\\nDuration Differences:")
    print(f"  Mean: {df_diffs['duration_diff'].mean()*1000:.2f} ms")
    print(f"  Median: {df_diffs['duration_diff'].median()*1000:.2f} ms")
    print(f"  Std: {df_diffs['duration_diff'].std()*1000:.2f} ms")
    print(f"  Max: {df_diffs['duration_diff'].max()*1000:.2f} ms")
    print(f"\\nSegment Counts:")
    print(f"  CTC average: {np.mean(segment_counts['ctc']):.1f} segments/file")
    print(f"  MFA average: {np.mean(segment_counts['mfa']):.1f} segments/file")
    print(f"  Difference: {abs(np.mean(segment_counts['ctc']) - np.mean(segment_counts['mfa'])):.1f} segments/file")
    print(f"{'='*80}")

# Usage example (to be added to notebooks):
# comparison_data = compare_timecodes_ctc_mfa(df_sample_ctc, df_sample_mfa, sample_size=100)
# plot_timecode_comparison(comparison_data, save_path='timecode_comparison.png')
'''
    
    return script

if __name__ == '__main__':
    script = create_timecode_comparison_script()
    
    output_path = Path('/Volumes/SSanDisk/SpeechRec-German-diagnostic/notebooks/timecode_comparison_code.py')
    output_path.write_text(script)
    
    print(f"✓ Created timecode comparison script: {output_path}")
    print("\nThis script can be imported into both notebooks to compare timecodes.")
    print("Add the code to a new cell in both notebooks after segment extraction.")
