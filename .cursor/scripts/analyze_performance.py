#!/usr/bin/env python3
"""
Analyze performance logs to show timing improvements and ASR engine usage.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

def analyze_logs(log_file_path: str):
    """Analyze performance logs and show timing breakdown."""
    log_path = Path(log_file_path)
    
    if not log_path.exists():
        print(f"Log file not found: {log_file_path}")
        return
    
    print("=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Read and parse logs
    performance_logs = []
    asr_logs = []
    g2p_logs = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if not line.strip():
                    continue
                log_entry = json.loads(line)
                
                # Filter performance logs
                if log_entry.get('runId') == 'performance':
                    performance_logs.append(log_entry)
                    
                    # Check for ASR-related logs
                    location = log_entry.get('location', '')
                    message = log_entry.get('message', '')
                    if 'asr' in location.lower() or 'asr' in message.lower():
                        asr_logs.append(log_entry)
                    
                    # Check for G2P-related logs
                    if 'g2p' in location.lower():
                        g2p_logs.append(log_entry)
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                continue
    
    if not performance_logs:
        print("No performance logs found.")
        return
    
    # Find process_pronunciation start/end pairs
    runs = []
    current_run = None
    
    for log in performance_logs:
        location = log.get('location', '')
        message = log.get('message', '')
        
        if 'process_pronunciation:start' in location:
            if current_run:
                runs.append(current_run)
            current_run = {
                'start': log,
                'stages': [],
                'total_time': None
            }
        elif 'process_pronunciation:end' in location:
            if current_run:
                current_run['end'] = log
                current_run['total_time'] = log.get('elapsed_ms', 0)
                runs.append(current_run)
                current_run = None
        elif current_run:
            # Track stage timings
            elapsed = log.get('elapsed_ms', 0)
            if elapsed > 0:  # Only track stages with actual time
                current_run['stages'].append({
                    'location': location,
                    'message': message,
                    'elapsed_ms': elapsed,
                    'data': log.get('data', {})
                })
    
    if not runs:
        print("No complete runs found in logs.")
        return
    
    # Analyze runs
    print(f"Found {len(runs)} complete run(s)\n")
    
    for i, run in enumerate(runs, 1):
        print(f"--- Run {i} ---")
        print(f"Total time: {run['total_time']:.1f} ms ({run['total_time']/1000:.2f} seconds)")
        print()
        
        # Show major stages
        print("Major stages:")
        stages_by_category = defaultdict(list)
        
        for stage in run['stages']:
            location = stage['location']
            elapsed = stage['elapsed_ms']
            
            # Categorize stages
            if 'asr' in location.lower():
                stages_by_category['ASR'].append((stage, elapsed))
            elif 'g2p' in location.lower() or 'dictionary' in location.lower():
                stages_by_category['G2P'].append((stage, elapsed))
            elif 'phoneme_recognition' in location.lower() or 'decode' in location.lower():
                stages_by_category['Phoneme Recognition'].append((stage, elapsed))
            elif 'forced_align' in location.lower() or 'alignment' in location.lower():
                stages_by_category['Alignment'].append((stage, elapsed))
            elif 'visualization' in location.lower() or 'viz' in location.lower():
                stages_by_category['Visualization'].append((stage, elapsed))
            elif elapsed > 100:  # Other significant stages
                stages_by_category['Other'].append((stage, elapsed))
        
        for category, stage_list in stages_by_category.items():
            total_category_time = sum(elapsed for _, elapsed in stage_list)
            if total_category_time > 0:
                print(f"  {category}: {total_category_time:.1f} ms")
                # Show details for significant stages
                for stage, elapsed in stage_list:
                    if elapsed > 50:  # Show stages > 50ms
                        location_short = stage['location'].split(':')[-1] if ':' in stage['location'] else stage['location']
                        print(f"    - {location_short}: {elapsed:.1f} ms")
        print()
    
    # Compare runs if multiple
    if len(runs) > 1:
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print()
        
        first_run = runs[0]
        last_run = runs[-1]
        
        improvement = first_run['total_time'] - last_run['total_time']
        improvement_pct = (improvement / first_run['total_time']) * 100 if first_run['total_time'] > 0 else 0
        
        print(f"First run:  {first_run['total_time']:.1f} ms ({first_run['total_time']/1000:.2f} seconds)")
        print(f"Last run:   {last_run['total_time']:.1f} ms ({last_run['total_time']/1000:.2f} seconds)")
        print(f"Improvement: {improvement:.1f} ms ({improvement_pct:.1f}% faster)")
        print()
    
    # Check ASR engine usage
    print("=" * 80)
    print("ASR ENGINE DETECTION")
    print("=" * 80)
    print()
    
    # Look for ASR timing in runs
    asr_times = []
    asr_engines_found = []
    asr_devices_found = []
    for run in runs:
        for stage in run['stages']:
            if 'asr' in stage['location'].lower() and 'completed' in stage['message'].lower():
                asr_times.append(stage['elapsed_ms'])
                data = stage.get('data', {})
                recognized_text = data.get('recognized_text', '')
                asr_engine = data.get('asr_engine', 'unknown')
                asr_device = data.get('asr_device', 'unknown')
                asr_engines_found.append(asr_engine)
                asr_devices_found.append(asr_device)
                print(f"ASR time: {stage['elapsed_ms']:.1f} ms")
                if asr_engine != 'unknown':
                    print(f"  Engine: {asr_engine.upper()}")
                if asr_device != 'unknown':
                    device_display = asr_device.upper()
                    if asr_device == 'mps':
                        device_display = "MPS (Apple Silicon GPU)"
                    elif asr_device == 'cuda':
                        device_display = "CUDA (NVIDIA GPU)"
                    elif asr_device == 'cpu':
                        device_display = "CPU"
                    elif asr_device == 'macos_native':
                        device_display = "macOS Native Framework"
                    print(f"  Device: {device_display}")
                if recognized_text:
                    print(f"  Recognized: \"{recognized_text}\"")
                print()
    
    # Determine actual engine and device
    if asr_engines_found:
        unique_engines = set(asr_engines_found)
        if 'unknown' in unique_engines:
            unique_engines.remove('unknown')
        if unique_engines:
            print(f"Detected ASR engine(s): {', '.join(unique_engines).upper()}")
        else:
            print("ASR engine not logged in performance data (check console output)")
    else:
        print("No ASR runs found in logs")
    
    # Show device information
    if asr_devices_found:
        unique_devices = set(asr_devices_found)
        if 'unknown' in unique_devices:
            unique_devices.remove('unknown')
        if unique_devices:
            device_labels = []
            for d in unique_devices:
                if d == 'mps':
                    device_labels.append("MPS (Apple Silicon GPU)")
                elif d == 'cuda':
                    device_labels.append("CUDA (NVIDIA GPU)")
                elif d == 'cpu':
                    device_labels.append("CPU")
                elif d == 'macos_native':
                    device_labels.append("macOS Native Framework")
                else:
                    device_labels.append(d.upper())
            print(f"Detected device(s): {', '.join(device_labels)}")
        print()
    
    # Estimate engine based on timing
    if asr_times:
        avg_asr_time = sum(asr_times) / len(asr_times)
        print(f"Average ASR time: {avg_asr_time:.1f} ms ({avg_asr_time/1000:.2f} seconds)")
        
        # macOS Speech is typically 1-3 seconds, Whisper medium is 5-8 seconds
        if not asr_engines_found or all(e == 'unknown' for e in asr_engines_found):
            if avg_asr_time < 3000:
                print("→ Estimated: macOS Speech (fast)")
            elif avg_asr_time < 5000:
                print("→ Estimated: macOS Speech or Whisper small/tiny")
            else:
                print("→ Estimated: Whisper (medium or larger)")
        print()
    
    # Check console output for ASR initialization messages
    print("Note: Check console output for ASR initialization messages:")
    print("  - 'ASR recognizer (macOS Speech) initialized' → macOS Speech")
    print("  - 'ASR recognizer (Whisper ...) initialized' → Whisper")
    print("  - 'macOS Speech not available, using fallback' → Fallback to Whisper")
    print()
    
    # G2P analysis
    print("=" * 80)
    print("G2P PERFORMANCE")
    print("=" * 80)
    print()
    
    g2p_times = []
    for run in runs:
        for stage in run['stages']:
            if 'g2p' in stage['location'].lower() and 'completed' in stage['message'].lower():
                g2p_times.append(stage['elapsed_ms'])
                print(f"G2P time: {stage['elapsed_ms']:.1f} ms")
    
    if g2p_times:
        avg_g2p_time = sum(g2p_times) / len(g2p_times)
        print(f"Average G2P time: {avg_g2p_time:.1f} ms")
        
        if avg_g2p_time < 100:
            print("→ G2P cache is working! (very fast)")
        elif avg_g2p_time < 1000:
            print("→ G2P is fast (likely using cache)")
        else:
            print("→ G2P is slow (may need dictionary caching)")
        print()

if __name__ == '__main__':
    log_file = '/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log'
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    analyze_logs(log_file)
