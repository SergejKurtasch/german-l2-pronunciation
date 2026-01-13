"""
Visualization module for displaying pronunciation analysis results.
"""

from typing import List, Dict, Tuple, Optional


def collapse_consecutive_duplicates(phonemes: List[str]) -> List[str]:
    """
    Collapse consecutive duplicate phonemes (same logic as CTC collapse).
    This ensures that expected and recognized phonemes are processed consistently.
    
    Args:
        phonemes: List of phoneme strings
        
    Returns:
        List of phonemes with consecutive duplicates collapsed
    """
    if not phonemes:
        return phonemes
    
    collapsed = []
    prev_phoneme = None
    
    for phoneme in phonemes:
        # Skip empty phonemes
        if not phoneme or not phoneme.strip():
            continue
        
        # If different from previous, add it
        if phoneme != prev_phoneme:
            collapsed.append(phoneme)
            prev_phoneme = phoneme
        # If same as previous, skip it (CTC collapse)
    
    return collapsed


def create_side_by_side_comparison(
    expected_phonemes: List[str],
    recognized_phonemes: List[str],
    aligned_pairs: List[Tuple[Optional[str], Optional[str]]]
) -> str:
    """
    Create side-by-side comparison of expected and recognized phonemes using alignment.
    Shows two rows: expected phonemes on top, recognized phonemes below with color coding.
    
    Args:
        expected_phonemes: List of expected phoneme strings (not used, kept for compatibility)
        recognized_phonemes: List of recognized phoneme strings (not used, kept for compatibility)
        aligned_pairs: List of aligned pairs from Needleman-Wunsch [(expected, recognized), ...]
        
    Returns:
        HTML string with two-row aligned comparison
        
    Color coding:
        - Green: phonemes match
        - Yellow: expected phoneme is missing (gap in recognized)
        - Red: phonemes differ
    """
    if not aligned_pairs:
        return "<div style='color: gray;'>No alignment data available.</div>"
    
    html = "<div style='font-family: monospace; font-size: 16px; line-height: 2.2; padding: 15px; background: #f8f9fa; border-radius: 5px;' data-block-id='side-by-side-comparison'>"
    
    # Add legend at the top
    html += "<div style='margin-bottom: 15px; padding: 10px; background: #fff; border-radius: 4px; font-size: 12px;'>"
    html += "<strong>Legend:</strong> "
    html += "<span style='color: green; margin-left: 10px;'>● Match</span> "
    html += "<span style='color: orange; margin-left: 10px;'>● Missing</span> "
    html += "<span style='color: red; margin-left: 10px;'>● Mismatch</span> "
    html += "<span style='color: blue; margin-left: 10px;'>● Extra</span> "
    html += "<span style='color: #999; margin-left: 10px;'>Double space = Word boundary</span>"
    html += "</div>"
    
    # Build two rows: expected (top) and recognized (bottom)
    expected_row = []
    recognized_row = []
    
    for idx, (expected_ph, recognized_ph) in enumerate(aligned_pairs):
        # Check if this is a word boundary position (either side has '||')
        is_word_boundary = (expected_ph == '||' or recognized_ph == '||')
        
        # Handle word boundary markers - show as double space
        # If one side is '||', both sides should show double space (even if other is None)
        if not is_word_boundary:
            if expected_ph is None:
                exp_display = '-'
            else:
                exp_display = expected_ph
            
            if recognized_ph is None:
                rec_display = '-'
            else:
                rec_display = recognized_ph
        
        # Determine color based on alignment
        if expected_ph == recognized_ph and expected_ph is not None:
            # Perfect match (including word boundaries)
            color = 'green'
        elif is_word_boundary:
            # Word boundary mismatch (one side has '||', other has None or different)
            if expected_ph == '||' and recognized_ph == '||':
                color = 'green'  # Both are boundaries - perfect match
            elif expected_ph == '||' and recognized_ph is None:
                color = 'orange'  # Missing word boundary in recognized
            elif recognized_ph == '||' and expected_ph is None:
                color = 'blue'  # Extra word boundary in recognized
            else:
                color = 'orange'  # Word boundary mismatch
        elif recognized_ph is None:
            # Gap in recognized sequence (missing phoneme)
            color = 'orange'
        elif expected_ph is None:
            # Gap in expected sequence (extra phoneme)
            color = 'blue'
        else:
            # Mismatch
            color = 'red'
        
        # Add to rows with no gaps between phonemes
        # Use inline-block with font-size: 0 on parent to remove gaps, then restore font-size inside
        # Both rows use same font size and weight for consistency
        if is_word_boundary:
            # Word boundary - add double space using two separate spans with min-width
            # Use regular space character with min-width to ensure visibility
            # Each span has min-width to create visible space even with font-size: 0 on parent
            expected_row.append(f"<span style='display: inline-block; color: {color}; font-size: 16px; font-weight: normal; min-width: 0.5em; white-space: pre;'> </span><span style='display: inline-block; color: {color}; font-size: 16px; font-weight: normal; min-width: 0.5em; white-space: pre;'> </span>")
            recognized_row.append(f"<span style='display: inline-block; color: {color}; font-size: 16px; font-weight: normal; min-width: 0.5em; white-space: pre;'> </span><span style='display: inline-block; color: {color}; font-size: 16px; font-weight: normal; min-width: 0.5em; white-space: pre;'> </span>")
        else:
            # Regular phoneme - add colored background, no margin/padding that creates gaps
            expected_row.append(f"<span style='display: inline-block; color: {color}; background: {color}10; padding: 2px 0; font-size: 16px; font-weight: normal;'>{exp_display}</span>")
            recognized_row.append(f"<span style='display: inline-block; color: {color}; background: {color}15; padding: 2px 0; font-size: 16px; font-weight: normal;'>{rec_display}</span>")
    
    # Split long sequences into chunks to fit screen width
    # Improved algorithm: calculate optimal chunk size based on container width
    # This algorithm dynamically calculates the number of phonemes that can fit on one line
    # based on the estimated container width and average phoneme width
    
    # Estimate average phoneme width (including inline-block spacing and padding)
    # Each phoneme span is approximately 25-40px wide (font-size 16px + padding 2px + inline-block spacing)
    # Shorter phonemes (1-2 chars) are ~25px, longer ones (3-4 chars) are ~35-40px
    AVG_PHONEME_WIDTH_PX = 32  # Average of short and long phonemes
    
    # Estimate container width dynamically
    # Chat bubble width varies: mobile ~600px, tablet ~800px, desktop ~1000-1200px
    # Use a more adaptive approach: calculate based on typical chat container
    # Most chat containers in Gradio are ~80-90% of viewport width, minus padding
    # For better distribution, use a larger estimate to maximize usage of available space
    ESTIMATED_CONTAINER_WIDTH_PX = 1000  # More generous estimate for better space utilization
    # Account for padding, margins, and label width (left/right padding + label)
    AVAILABLE_WIDTH_PX = ESTIMATED_CONTAINER_WIDTH_PX - 60  # 60px for padding/margins/label
    
    # Calculate optimal chunk size
    OPTIMAL_CHUNK_SIZE = max(40, int(AVAILABLE_WIDTH_PX / AVG_PHONEME_WIDTH_PX))
    
    # Use optimal chunk size with a small safety margin to account for variable phoneme widths
    # This ensures most sequences fit on one line while preventing overflow
    CHUNK_SIZE = max(40, OPTIMAL_CHUNK_SIZE - 3)  # Subtract 3 for safety margin
    
    total_pairs = len(expected_row)
    
    # Build HTML with multiple pairs of rows if needed
    for chunk_start in range(0, total_pairs, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_pairs)
        expected_chunk = expected_row[chunk_start:chunk_end]
        recognized_chunk = recognized_row[chunk_start:chunk_end]
        
        # Expected phonemes row - normal background
        # Use natural text wrapping like regular text, exactly like colored text display
        html += "<div style='margin-bottom: 5px; padding: 5px 8px; background: #ffffff; border-radius: 3px;'>"
        html += "<div style='color: #495057; font-size: 13px; font-weight: bold; margin-bottom: 2px; text-align: right;'>Expected phonemes:</div>"
        # Use exact same style as regular text: font-size: 18px, line-height: 1.3, margin: 5px 0
        # This allows natural text wrapping like regular text
        html += "<div style='font-size: 18px; line-height: 1.3; margin: 5px 0;'>" + "".join(expected_chunk) + "</div>"
        html += "</div>"
        
        # Recognized phonemes row - light gray background
        html += "<div style='margin-bottom: 10px; padding: 5px 8px; background: #e9ecef; border-radius: 3px;'>"
        html += "<div style='color: #495057; font-size: 13px; font-weight: bold; margin-bottom: 2px; text-align: right;'>Recognized phonemes:</div>"
        # Use exact same style as regular text for natural wrapping
        html += "<div style='font-size: 18px; line-height: 1.3; margin: 5px 0;'>" + "".join(recognized_chunk) + "</div>"
        html += "</div>"
    
    html += "</div>"
    
    return html


def create_colored_text(
    text: str,
    aligned_pairs: List[Dict],
    expected_phonemes_dict: Optional[List[Dict]] = None,
    aligned_pairs_tuples: Optional[List[Tuple[Optional[str], Optional[str]]]] = None
) -> str:
    """
    Create colored text visualization based on Expected phonemes colors.
    Uses a 3D mapping: (text_char, phoneme, color) for precise character-to-phoneme alignment.
    Text coloring matches the colors used for Expected phonemes in side-by-side comparison.
    
    Args:
        text: Original text
        aligned_pairs: List of dictionaries with alignment results (for backward compatibility)
        expected_phonemes_dict: List of expected phoneme dictionaries with 'text_char' and 'phoneme' fields
        aligned_pairs_tuples: List of aligned pairs (expected, recognized) tuples for color determination
        
    Returns:
        HTML string with colored text
    """
    # Use aligned_pairs_tuples if provided, otherwise extract from aligned_pairs
    if aligned_pairs_tuples is None:
        # Extract tuples from dict format
        aligned_pairs_tuples = []
        for pair in aligned_pairs:
            exp = pair.get('expected', None)
            rec = pair.get('recognized', None)
            aligned_pairs_tuples.append((exp, rec))
    
    if not expected_phonemes_dict or not aligned_pairs_tuples:
        # Fallback: return uncolored text
        return f"<div style='font-size: 18px; line-height: 1.8;'>{text}</div>"
    
    # Step 1: Build 3D mapping structure: (text_char, phoneme, color)
    # First, build list of expected phonemes from dict (before collapse) with position info
    expected_phonemes_list = []
    for ph_dict in expected_phonemes_dict:
        phoneme = ph_dict.get('phoneme', '')
        text_char = ph_dict.get('text_char', '')
        position = ph_dict.get('position', 0)
        if phoneme and phoneme != '||':  # Skip word boundaries
            expected_phonemes_list.append({
                'phoneme': phoneme,
                'text_char': text_char,
                'position': position
            })
    
    # Step 2: Apply CTC collapse to expected phonemes (same as in app.py)
    phoneme_strings = [ph_info['phoneme'] for ph_info in expected_phonemes_list]
    collapsed_phoneme_strings = collapse_consecutive_duplicates(phoneme_strings)
    
    # Step 3: Rebuild list with collapsed phonemes, preserving text_char and position info
    collapsed_expected_phonemes = []
    prev_phoneme = None
    for ph_info in expected_phonemes_list:
        ph = ph_info['phoneme']
        if ph != prev_phoneme:
            collapsed_expected_phonemes.append(ph_info)
            prev_phoneme = ph
    
    # Step 4: Build mapping from collapsed expected phoneme to its color
    collapsed_ph_to_color = {}
    aligned_idx = 0  # Current position in aligned_pairs
    
    # Go through collapsed expected phonemes in order and match them with aligned_pairs
    for collapsed_ph_idx, expected_ph_info in enumerate(collapsed_expected_phonemes):
        expected_ph = expected_ph_info['phoneme']
        
        # Find this phoneme in aligned_pairs starting from current position
        found = False
        while aligned_idx < len(aligned_pairs_tuples):
            exp_ph, rec_ph = aligned_pairs_tuples[aligned_idx]
            
            if exp_ph == expected_ph:
                # Found matching expected phoneme - determine its color
                is_word_boundary = (exp_ph == '||' or rec_ph == '||')
                
                if exp_ph == rec_ph:
                    color = 'green'  # Perfect match
                elif is_word_boundary:
                    if exp_ph == '||' and rec_ph == '||':
                        color = 'green'
                    elif exp_ph == '||' and rec_ph is None:
                        color = 'orange'
                    elif rec_ph == '||' and exp_ph is None:
                        color = 'blue'
                    else:
                        color = 'orange'
                elif rec_ph is None:
                    color = 'orange'  # Missing phoneme
                elif exp_ph is None:
                    color = 'blue'  # Extra phoneme
                else:
                    color = 'red'  # Mismatch
                
                # Map this collapsed phoneme to its color
                collapsed_ph_to_color[collapsed_ph_idx] = color
                aligned_idx += 1
                found = True
                break
            elif exp_ph is None or exp_ph == '||':
                # Gap or word boundary in aligned_pairs, skip
                aligned_idx += 1
            else:
                # Different phoneme - this shouldn't happen if collapse is consistent
                # But advance to avoid infinite loop
                aligned_idx += 1
        
        if not found:
            # Phoneme not found in aligned_pairs - use default color (shouldn't happen)
            collapsed_ph_to_color[collapsed_ph_idx] = 'gray'
    
    # Step 5: Build 3D mapping: char_position -> (phoneme, color)
    # Map each character position in text to its corresponding phoneme and color
    import re
    
    # Split text into tokens while preserving their positions
    tokens_with_positions = []
    for match in re.finditer(r"[\w']+|[^\w\s]", text):
        token = match.group()
        start_pos = match.start()
        tokens_with_positions.append({
            'token': token,
            'start_pos': start_pos,
            'end_pos': start_pos + len(token)
        })
    
    # Build mapping: char_index -> color
    char_to_color = {}
    
    # Group collapsed phonemes by their text_char (token)
    # Create mapping that handles both with and without punctuation
    phonemes_by_token = {}
    for i, ph_info in enumerate(collapsed_expected_phonemes):
        text_char = ph_info['text_char']
        # Normalize token for matching (remove punctuation for comparison)
        text_char_clean = re.sub(r'[^\w]', '', text_char)
        if text_char_clean not in phonemes_by_token:
            phonemes_by_token[text_char_clean] = []
        phonemes_by_token[text_char_clean].append({
            'index': i,
            'phoneme': ph_info['phoneme'],
            'text_char': text_char,
            'text_char_clean': text_char_clean
        })
    
    # For each token, find its phonemes and map characters to colors
    for token_info in tokens_with_positions:
        token = token_info['token']
        token_start = token_info['start_pos']
        token_end = token_info['end_pos']
        
        # Normalize token for matching (remove punctuation)
        token_clean = re.sub(r'[^\w]', '', token)
        
        # Find phonemes for this token using the mapping
        token_phonemes = phonemes_by_token.get(token_clean, [])
        
        if token_phonemes:
            # Get all characters in token (including punctuation)
            token_chars = list(token)
            # Get alphanumeric characters only for phoneme distribution
            alnum_chars = [c for c in token if c.isalnum()]
            
            if alnum_chars:
                # More accurate distribution: map phonemes to characters
                # Use proportional distribution but ensure all characters get a color
                num_phonemes = len(token_phonemes)
                num_chars = len(alnum_chars)
                
                # Calculate how many phonemes per character (can be fractional)
                if num_phonemes > 0 and num_chars > 0:
                    phonemes_per_char = num_phonemes / num_chars
                    
                    alnum_idx = 0
                    for i, char in enumerate(token_chars):
                        char_pos_in_text = token_start + i
                        
                        if char.isalnum():
                            # Map alphanumeric character to phoneme
                            # Use floor to ensure we don't go out of bounds
                            ph_idx = min(int(alnum_idx * phonemes_per_char), num_phonemes - 1)
                            collapsed_ph_idx = token_phonemes[ph_idx]['index']
                            if collapsed_ph_idx in collapsed_ph_to_color:
                                char_to_color[char_pos_in_text] = collapsed_ph_to_color[collapsed_ph_idx]
                            alnum_idx += 1
                        else:
                            # For punctuation within token, use color of the last phoneme
                            # This ensures punctuation at the end of words gets colored
                            last_ph_idx = token_phonemes[-1]['index']
                            if last_ph_idx in collapsed_ph_to_color:
                                char_to_color[char_pos_in_text] = collapsed_ph_to_color[last_ph_idx]
            else:
                # Token has no alphanumeric chars (only punctuation)
                # Use color of first phoneme if available
                if token_phonemes:
                    first_ph_idx = token_phonemes[0]['index']
                    if first_ph_idx in collapsed_ph_to_color:
                        for i, char in enumerate(token_chars):
                            char_pos_in_text = token_start + i
                            char_to_color[char_pos_in_text] = collapsed_ph_to_color[first_ph_idx]
    
    # Step 6: Generate HTML with colored text (without bold by default)
    html = "<div style='font-size: 18px; line-height: 1.3; margin: 5px 0;'>"
    
    for i, char in enumerate(text):
        if char.isspace():
            # Don't color spaces
            html += char
        elif i in char_to_color:
            color = char_to_color[i]
            html += f"<span style='color: {color};'>{char}</span>"
        else:
            # Character not mapped to any phoneme, leave uncolored
            html += char
    
    html += "</div>"
    return html


def create_text_comparison_view(
    expected_text: str,
    recognized_text: str,
    wer_result: Optional[Dict] = None
) -> str:
    """
    Create text comparison view when WER is too high.
    
    Args:
        expected_text: Expected text
        recognized_text: Recognized text from ASR
        wer_result: Optional WER calculation result
        
    Returns:
        HTML string with text comparison
    """
    # #region agent log
    import json, time
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"visualization.py:create_text_comparison_view_entry","message":"create_text_comparison_view called","data":{"expected_text":expected_text,"recognized_text":recognized_text,"wer_result":wer_result,"expected_is_none":expected_text is None,"recognized_is_none":recognized_text is None},"timestamp":int(time.time()*1000)})+'\n')
    # #endregion
    
    # Handle None values
    if expected_text is None:
        expected_text = ""
    if recognized_text is None:
        recognized_text = ""
    
    html = "<div style='padding: 15px; background: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;'>"
    html += "<h4 style='color: #856404; margin-top: 0;'>Text Comparison (High Word Error Rate)</h4>"
    
    if wer_result:
        html += f"<div style='margin-bottom: 15px; padding: 10px; background: #fff; border-radius: 4px;'>"
        html += f"<p style='margin: 0;'><strong>Word Error Rate (WER):</strong> {wer_result['wer']:.2%}</p>"
        html += f"<p style='margin: 5px 0 0 0;'><strong>Correct words:</strong> {wer_result['hits']} / {wer_result['total_reference_words']}</p>"
        html += f"<p style='margin: 5px 0 0 0;'><strong>Substitutions:</strong> {wer_result['substitutions']}, "
        html += f"<strong>Deletions:</strong> {wer_result['deletions']}, "
        html += f"<strong>Insertions:</strong> {wer_result['insertions']}</p>"
        html += "</div>"
    
    html += "<div style='margin-bottom: 15px; padding: 10px; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 4px;'>"
    html += "<p style='margin: 0; font-weight: bold; color: #2c3e50;'>Expected Text:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50; font-size: 16px;'>{expected_text}</p>"
    html += "</div>"
    
    html += "<div style='padding: 10px; background: #f0f8e8; border-left: 4px solid #27ae60; border-radius: 4px;'>"
    html += "<p style='margin: 0; font-weight: bold; color: #2c3e50;'>Recognized Text:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50; font-size: 16px;'>{recognized_text or 'N/A'}</p>"
    html += "</div>"
    
    html += "<div style='margin-top: 15px; padding: 10px; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 4px;'>"
    html += "<p style='margin: 0; color: #721c24;'><strong>Note:</strong> The recognized text differs significantly from the expected text. "
    html += "Phoneme-level analysis has been skipped. Please try to pronounce the expected text more accurately.</p>"
    html += "</div>"
    
    html += "</div>"
    
    # #region agent log
    import json, time
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"visualization.py:create_text_comparison_view_exit","message":"create_text_comparison_view completed","data":{"html_length":len(html)},"timestamp":int(time.time()*1000)})+'\n')
    # #endregion
    
    return html


def create_detailed_report(
    aligned_pairs: List[Dict],
    feedback_list: List[Dict],
    text: str,
    model2_name: Optional[str] = None,
    model2_aligned_pairs: Optional[List[Dict]] = None,
    model2_diagnostic_results: Optional[List[Dict]] = None,
    model3_name: Optional[str] = None,
    model3_aligned_pairs: Optional[List[Dict]] = None,
    model3_diagnostic_results: Optional[List[Dict]] = None,
    wer_result: Optional[Dict] = None,
    per_result: Optional[Dict] = None,
    recognized_text: Optional[str] = None
) -> str:
    """
    Create detailed report with feedback for one, two, or three models.
    
    Args:
        aligned_pairs: List of alignment results for model 1
        feedback_list: List of feedback dictionaries for model 1
        text: Original text
        model2_name: Optional name of second model
        model2_aligned_pairs: Optional alignment results for model 2
        model2_diagnostic_results: Optional diagnostic results for model 2
        model3_name: Optional name of third model
        model3_aligned_pairs: Optional alignment results for model 3
        model3_diagnostic_results: Optional diagnostic results for model 3
        wer_result: Optional WER calculation result
        per_result: Optional PER calculation result
        recognized_text: Optional recognized text from ASR
        
    Returns:
        HTML string with detailed report
    """
    html = "<div style='padding: 15px; background: #f9f9f9; border-radius: 5px;'>"
    html += "<h4 style='color: #333; margin-top: 0;'>Detailed Report</h4>"
    
    # Metrics section (WER and PER)
    if wer_result or per_result or recognized_text:
        html += "<div style='margin-bottom: 20px; padding: 10px; background: #e1f5fe; border-left: 4px solid #0288d1; border-radius: 4px;'>"
        html += "<h5 style='color: #2c3e50; margin-top: 0;'>Metrics</h5>"
        
        if recognized_text:
            html += f"<p style='color: #2c3e50;'><strong>Recognized text:</strong> {recognized_text}</p>"
        
        if wer_result:
            html += f"<p style='color: #2c3e50;'><strong>WER (Word Error Rate):</strong> {wer_result['wer']:.2%}</p>"
            html += f"<p style='color: #2c3e50; margin-left: 20px;'>"
            html += f"Substitutions: {wer_result['substitutions']}, "
            html += f"Deletions: {wer_result['deletions']}, "
            html += f"Insertions: {wer_result['insertions']}, "
            html += f"Correct: {wer_result['hits']} / {wer_result['total_reference_words']}"
            html += "</p>"
        
        if per_result:
            html += f"<p style='color: #2c3e50;'><strong>PER (Phoneme Error Rate):</strong> {per_result['per']:.2%}</p>"
            html += f"<p style='color: #2c3e50; margin-left: 20px;'>"
            html += f"Substitutions: {per_result['substitutions']}, "
            html += f"Deletions: {per_result['deletions']}, "
            html += f"Insertions: {per_result['insertions']}, "
            html += f"Total expected: {per_result['total_expected']}"
            html += "</p>"
        
        html += "</div>"
    
    # Model 1 Statistics (may be PRIMARY model if available)
    total = len(aligned_pairs)
    correct = sum(1 for p in aligned_pairs if p.get('is_correct', False))
    incorrect = sum(1 for p in aligned_pairs if not p.get('is_correct', False) and not p.get('is_missing', False) and not p.get('is_extra', False))
    missing = sum(1 for p in aligned_pairs if p.get('is_missing', False))
    extra = sum(1 for p in aligned_pairs if p.get('is_extra', False))
    
    # Determine model name - check if it's PRIMARY model by checking if model2_name is "Wav2Vec2 XLS-R"
    model1_display_name = "Wav2Vec2 XLS-R Model"
    if model2_name == "Wav2Vec2 XLS-R":
        # This means model1 is PRIMARY
        model1_display_name = "Wav2Vec2 XLSR-53 eSpeak (PRIMARY) Model"
    
    html += "<div style='margin-bottom: 20px; padding: 10px; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 4px;'>"
    html += f"<h5 style='color: #2c3e50; margin-top: 0;'>{model1_display_name}</h5>"
    html += f"<p style='color: #2c3e50;'><strong>Total phonemes:</strong> {total}</p>"
    html += f"<p style='color: #2c3e50;'><strong>Correct:</strong> {correct} ({correct/total*100:.1f}%)</p>"
    html += f"<p style='color: #2c3e50;'><strong>Incorrect:</strong> {incorrect} ({incorrect/total*100:.1f}%)</p>"
    html += f"<p style='color: #2c3e50;'><strong>Missing:</strong> {missing} ({missing/total*100:.1f}%)</p>"
    if extra > 0:
        html += f"<p style='color: #2c3e50;'><strong>Extra:</strong> {extra} ({extra/total*100:.1f}%)</p>"
    html += "</div>"
    
    # Model 2 Statistics (if available)
    if model2_name and model2_aligned_pairs is not None and model2_diagnostic_results is not None:
        total2 = len(model2_aligned_pairs)
        correct2 = sum(1 for p in model2_aligned_pairs if p.get('is_correct', False))
        incorrect2 = sum(1 for p in model2_aligned_pairs if not p.get('is_correct', False) and not p.get('is_missing', False) and not p.get('is_extra', False))
        missing2 = sum(1 for p in model2_aligned_pairs if p.get('is_missing', False))
        extra2 = sum(1 for p in model2_aligned_pairs if p.get('is_extra', False))
        
        html += "<div style='margin-bottom: 20px; padding: 10px; background: #f0f8e8; border-left: 4px solid #27ae60; border-radius: 4px;'>"
        html += f"<h5 style='color: #2c3e50; margin-top: 0;'>{model2_name} Model</h5>"
        html += f"<p style='color: #2c3e50;'><strong>Total phonemes:</strong> {total2}</p>"
        html += f"<p style='color: #2c3e50;'><strong>Correct:</strong> {correct2} ({correct2/total2*100:.1f}%)</p>"
        html += f"<p style='color: #2c3e50;'><strong>Incorrect:</strong> {incorrect2} ({incorrect2/total2*100:.1f}%)</p>"
        html += f"<p style='color: #2c3e50;'><strong>Missing:</strong> {missing2} ({missing2/total2*100:.1f}%)</p>"
        if extra2 > 0:
            html += f"<p style='color: #2c3e50;'><strong>Extra:</strong> {extra2} ({extra2/total2*100:.1f}%)</p>"
        html += "</div>"
    
    # Model 3 Statistics (if available)
    if model3_name and model3_aligned_pairs is not None and model3_diagnostic_results is not None:
        total3 = len(model3_aligned_pairs)
        correct3 = sum(1 for p in model3_aligned_pairs if p.get('is_correct', False))
        incorrect3 = sum(1 for p in model3_aligned_pairs if not p.get('is_correct', False) and not p.get('is_missing', False) and not p.get('is_extra', False))
        missing3 = sum(1 for p in model3_aligned_pairs if p.get('is_missing', False))
        extra3 = sum(1 for p in model3_aligned_pairs if p.get('is_extra', False))
        
        html += "<div style='margin-bottom: 20px; padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;'>"
        html += f"<h5 style='color: #2c3e50; margin-top: 0;'>{model3_name} Model</h5>"
        html += f"<p style='color: #2c3e50;'><strong>Total phonemes:</strong> {total3}</p>"
        html += f"<p style='color: #2c3e50;'><strong>Correct:</strong> {correct3} ({correct3/total3*100:.1f}%)</p>"
        html += f"<p style='color: #2c3e50;'><strong>Incorrect:</strong> {incorrect3} ({incorrect3/total3*100:.1f}%)</p>"
        html += f"<p style='color: #2c3e50;'><strong>Missing:</strong> {missing3} ({missing3/total3*100:.1f}%)</p>"
        if extra3 > 0:
            html += f"<p style='color: #2c3e50;'><strong>Extra:</strong> {extra3} ({extra3/total3*100:.1f}%)</p>"
        html += "</div>"
    
    # Combined Statistics (comparison of all available models)
    accuracy1 = (correct / total * 100) if total > 0 else 0
    model1_name_for_comparison = "Wav2Vec2 XLS-R"
    if model2_name == "Wav2Vec2 XLS-R":
        model1_name_for_comparison = "Wav2Vec2 XLSR-53 eSpeak (PRIMARY)"
    accuracies = [(model1_name_for_comparison, accuracy1)]
    
    if model2_name and model2_aligned_pairs is not None:
        accuracy2 = (correct2 / total2 * 100) if total2 > 0 else 0
        accuracies.append((model2_name, accuracy2))
    
    if model3_name and model3_aligned_pairs is not None:
        accuracy3 = (correct3 / total3 * 100) if total3 > 0 else 0
        accuracies.append((model3_name, accuracy3))
    
    if len(accuracies) > 1:
        html += "<div style='margin-bottom: 20px; padding: 10px; background: #e1f5fe; border-left: 4px solid #0288d1; border-radius: 4px;'>"
        html += "<h5 style='color: #2c3e50; margin-top: 0;'>Model Comparison</h5>"
        for model_name, model_accuracy in accuracies:
            html += f"<p style='color: #2c3e50;'><strong>{model_name} accuracy:</strong> {model_accuracy:.1f}%</p>"
        
        # Find best model
        best_model = max(accuracies, key=lambda x: x[1])
        other_models = [m for m in accuracies if m[0] != best_model[0]]
        if other_models:
            html += f"<p style='color: #2c3e50;'><em><strong>{best_model[0]}</strong> performs best ({best_model[1]:.1f}%)</em></p>"
            for other_name, other_acc in other_models:
                diff = best_model[1] - other_acc
                html += f"<p style='color: #2c3e50;'><em>Better than {other_name} by {diff:.1f}%</em></p>"
        html += "</div>"
    
    # Errors with feedback for Model 1
    errors = [f for f in feedback_list if not f.get('is_correct', False)]
    if errors:
        model1_error_title = "Wav2Vec2 XLS-R Errors and Feedback:"
        if model2_name == "Wav2Vec2 XLS-R":
            model1_error_title = "Wav2Vec2 XLSR-53 eSpeak (PRIMARY) Errors and Feedback:"
        html += f"<h5 style='color: #333;'>{model1_error_title}</h5>"
        html += "<ul style='list-style-type: none; padding-left: 0;'>"
        
        for i, error in enumerate(errors[:20]):  # Limit to first 20
            expected = error.get('expected', 'N/A')
            recognized = error.get('recognized', 'N/A')
            feedback = error.get('feedback_en', '')
            
            html += "<li style='margin-bottom: 10px; padding: 8px; background: #fff; border-left: 3px solid #e74c3c;'>"
            html += f"<strong>Error {i+1}:</strong> "
            if error.get('is_missing'):
                html += f"Missing phoneme '<code>{expected}</code>'"
            elif error.get('is_extra'):
                html += f"Extra phoneme '<code>{recognized}</code>'"
            else:
                html += f"Expected '<code>{expected}</code>', got '<code>{recognized}</code>'"
            
            if feedback:
                html += f"<br/><em style='color: #555;'>{feedback}</em>"
            html += "</li>"
        
        if len(errors) > 20:
            html += f"<li>... and {len(errors) - 20} more errors</li>"
        
        html += "</ul>"
    
    # Errors with feedback for Model 2 (if available)
    if model2_name and model2_diagnostic_results is not None:
        errors2 = [f for f in model2_diagnostic_results if not f.get('is_correct', False)]
        if errors2:
            html += f"<h5 style='color: #333;'>{model2_name} Errors and Feedback:</h5>"
            html += "<ul style='list-style-type: none; padding-left: 0;'>"
            
            for i, error in enumerate(errors2[:20]):  # Limit to first 20
                expected = error.get('expected', 'N/A')
                recognized = error.get('recognized', 'N/A')
                feedback = error.get('feedback_en', '')
                
                html += "<li style='margin-bottom: 10px; padding: 8px; background: #fff; border-left: 3px solid #e74c3c;'>"
                html += f"<strong>Error {i+1}:</strong> "
                if error.get('is_missing'):
                    html += f"Missing phoneme '<code>{expected}</code>'"
                elif error.get('is_extra'):
                    html += f"Extra phoneme '<code>{recognized}</code>'"
                else:
                    html += f"Expected '<code>{expected}</code>', got '<code>{recognized}</code>'"
                
                if feedback:
                    html += f"<br/><em style='color: #555;'>{feedback}</em>"
                html += "</li>"
            
            if len(errors2) > 20:
                html += f"<li>... and {len(errors2) - 20} more errors</li>"
            
            html += "</ul>"
    
    # Errors with feedback for Model 3 (if available)
    if model3_name and model3_diagnostic_results is not None:
        errors3 = [f for f in model3_diagnostic_results if not f.get('is_correct', False)]
        if errors3:
            html += f"<h5 style='color: #333;'>{model3_name} Errors and Feedback:</h5>"
            html += "<ul style='list-style-type: none; padding-left: 0;'>"
            
            for i, error in enumerate(errors3[:20]):  # Limit to first 20
                expected = error.get('expected', 'N/A')
                recognized = error.get('recognized', 'N/A')
                feedback = error.get('feedback_en', '')
                
                html += "<li style='margin-bottom: 10px; padding: 8px; background: #fff; border-left: 3px solid #e74c3c;'>"
                html += f"<strong>Error {i+1}:</strong> "
                if error.get('is_missing'):
                    html += f"Missing phoneme '<code>{expected}</code>'"
                elif error.get('is_extra'):
                    html += f"Extra phoneme '<code>{recognized}</code>'"
                else:
                    html += f"Expected '<code>{expected}</code>', got '<code>{recognized}</code>'"
                
                if feedback:
                    html += f"<br/><em style='color: #555;'>{feedback}</em>"
                html += "</li>"
            
            if len(errors3) > 20:
                html += f"<li>... and {len(errors3) - 20} more errors</li>"
            
            html += "</ul>"
    
    html += "</div>"
    return html


def create_simple_phoneme_comparison(
    expected_phonemes: List[str],
    recognized_phonemes: List[str]
) -> str:
    """
    Create simple phoneme comparison string.
    
    Args:
        expected_phonemes: List of expected phonemes
        recognized_phonemes: List of recognized phonemes
        
    Returns:
        HTML string with simple comparison
    """
    # #region agent log
    import json
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"visualization.py:416","message":"create_simple_phoneme_comparison entry","data":{"expected_phonemes":expected_phonemes,"recognized_phonemes":recognized_phonemes,"expected_count":len(expected_phonemes),"recognized_count":len(recognized_phonemes)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    # Filter out word boundary markers and replace with visible separator
    def format_phoneme(ph: str) -> str:
        return '|' if ph == '||' else ph
    
    expected_str = ' '.join(format_phoneme(ph) for ph in expected_phonemes)
    recognized_str = ' '.join(format_phoneme(ph) for ph in recognized_phonemes)
    
    # #region agent log
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"visualization.py:420","message":"after join operations","data":{"expected_str":expected_str,"recognized_str":recognized_str,"recognized_has_dash":'-' in recognized_str},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    html = "<div style='font-family: monospace; font-size: 14px;' data-block-id='recognized-phonemes'>"
    # Only show recognized phonemes, not expected (expected are shown in separate block)
    if recognized_str:
        html += f"<p>{recognized_str}</p>"
    html += "</div>"
    
    # #region agent log
    import re
    dash_positions = [m.start() for m in re.finditer(r'-', html)]
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"visualization.py:430","message":"before return html","data":{"html_full":html,"html_has_dash":'-' in html,"dash_count":html.count('-'),"dash_positions":dash_positions[:20],"recognized_str_in_html":recognized_str in html},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    return html


def create_text_with_sources_display(
    text: str,
    expected_phonemes_dict: List[Dict]
) -> str:
    """
    Create display showing original text with source information for each word.
    
    Args:
        text: Original German text
        expected_phonemes_dict: List of phoneme dictionaries with 'text_char' and 'source' fields
        
    Returns:
        HTML string with text and source labels
    """
    if not text or not expected_phonemes_dict:
        return "<div style='color: gray; padding: 10px;'>No text or phoneme information available.</div>"
    
    # Group phonemes by word (using text_char field)
    word_sources = {}
    for ph_info in expected_phonemes_dict:
        word = ph_info.get('text_char', '')
        source = ph_info.get('source', 'unknown')
        if word:
            # Normalize word for lookup (lowercase, remove punctuation)
            word_key = word.lower().strip(".,!?;:()\"")
            # Get the first source for this word (all phonemes of a word should have same source)
            if word_key not in word_sources:
                word_sources[word_key] = source
    
    # Map source names to display names and colors
    source_display = {
        'dsl': ('DSL', '#3498db'),      # Blue
        'mfa': ('MFA', '#27ae60'),      # Green
        'espeak': ('eSpeak', '#e67e22'), # Orange
        'unknown': ('Unknown', '#95a5a6') # Gray
    }
    
    # Split text into words (preserving punctuation and spaces)
    import re
    words = re.findall(r"[\w']+|[^\w\s]", text)
    
    html = "<div style='padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; margin-bottom: 15px;'>"
    html += "<h5 style='color: #2c3e50; margin-top: 0; margin-bottom: 15px;'>Original Text with Transcription Sources</h5>"
    
    html += "<div style='font-size: 18px; line-height: 2.0; margin-bottom: 10px;'>"
    
    for word in words:
        # Skip punctuation-only tokens
        if not word[0].isalnum():
            html += f"<span>{word}</span>"
            continue
        
        # Get source for this word (normalize for lookup)
        word_key = word.lower().strip(".,!?;:()\"")
        source = word_sources.get(word_key, 'unknown')
        display_name, color = source_display.get(source, ('Unknown', '#95a5a6'))
        
        # Create word with source badge
        html += f"<span style='position: relative; display: inline-block; margin-right: 8px;'>"
        html += f"<span style='font-weight: bold;'>{word}</span>"
        html += f"<span style='font-size: 10px; color: {color}; background: {color}20; padding: 2px 6px; border-radius: 3px; margin-left: 4px; vertical-align: super;'>{display_name}</span>"
        html += f"</span>"
    
    html += "</div>"
    
    # Add legend
    html += "<div style='margin-top: 15px; padding: 10px; background: #fff; border-radius: 4px; font-size: 12px;'>"
    html += "<strong>Legend:</strong> "
    for source_key, (display_name, color) in source_display.items():
        html += f"<span style='color: {color}; background: {color}20; padding: 2px 6px; border-radius: 3px; margin-left: 5px;'>{display_name}</span>"
    html += "</div>"
    
    html += "</div>"
    
    return html


def create_raw_phonemes_display(raw_phonemes: List[str]) -> str:
    """
    Create display for raw phonemes (before filtering).
    
    Args:
        raw_phonemes: List of raw phoneme strings (before filtering, may include '||' markers)
        
    Returns:
        HTML string with raw phonemes display
    """
    if not raw_phonemes:
        return "<div style='color: gray; padding: 10px;'>No raw phonemes available.</div>"
    
    # Format phonemes: replace '||' with visible separator
    def format_phoneme(ph: str) -> str:
        return '|' if ph == '||' else ph
    
    raw_str = ' '.join(format_phoneme(ph) for ph in raw_phonemes)
    
    html = "<div style='padding: 10px; background: #f0f0f0; border-radius: 5px; border-left: 4px solid #6c757d;'>"
    html += "<h5 style='color: #2c3e50; margin-top: 0;'>Raw Phonemes (Before Filtering)</h5>"
    html += f"<p style='color: #2c3e50; margin: 5px 0;'><strong>Total:</strong> {len(raw_phonemes)} phonemes</p>"
    html += f"<div style='font-family: monospace; font-size: 14px; background: #fff; padding: 10px; border-radius: 4px; margin-top: 10px;'>"
    html += f"<p style='margin: 0; word-break: break-all;'>{raw_str}</p>"
    html += "</div>"
    html += "<p style='color: #6c757d; font-size: 12px; margin-top: 10px; margin-bottom: 0;'>"
    html += "<em>These are all phonemes recognized by the model before filtering (whitelist and confidence filtering).</em>"
    html += "</p>"
    html += "</div>"
    
    return html


def create_dual_model_comparison(
    model1_name: str,
    model1_phonemes: List[str],
    model2_name: str,
    model2_phonemes: List[str]
) -> str:
    """
    Create comparison of phonemes from two different models.
    
    Args:
        model1_name: Name of the first model
        model1_phonemes: List of phonemes from first model
        model2_name: Name of the second model
        model2_phonemes: List of phonemes from second model
        
    Returns:
        HTML string with dual model comparison
    """
    model1_str = ' '.join(model1_phonemes) if model1_phonemes else '(none)'
    model2_str = ' '.join(model2_phonemes) if model2_phonemes else '(none)'
    
    html = "<div style='font-family: monospace; font-size: 14px;'>"
    html += f"<div style='margin-bottom: 15px; padding: 10px; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model1_name}:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model1_str}</p>"
    html += "</div>"
    
    html += f"<div style='padding: 10px; background: #f0f8e8; border-left: 4px solid #27ae60; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model2_name}:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model2_str}</p>"
    html += "</div>"
    
    html += "</div>"
    
    return html


def create_triple_model_comparison(
    model1_name: str,
    model1_phonemes: List[str],
    model2_name: str,
    model2_phonemes: List[str],
    model3_name: str,
    model3_phonemes: List[str]
) -> str:
    """
    Create comparison of phonemes from three different models.
    
    Args:
        model1_name: Name of the first model
        model1_phonemes: List of phonemes from first model
        model2_name: Name of the second model
        model2_phonemes: List of phonemes from second model
        model3_name: Name of the third model
        model3_phonemes: List of phonemes from third model
        
    Returns:
        HTML string with triple model comparison
    """
    model1_str = ' '.join(model1_phonemes) if model1_phonemes else '(none)'
    model2_str = ' '.join(model2_phonemes) if model2_phonemes else '(none)'
    model3_str = ' '.join(model3_phonemes) if model3_phonemes else '(none)'
    
    html = "<div style='font-family: monospace; font-size: 14px;'>"
    html += f"<div style='margin-bottom: 15px; padding: 10px; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model1_name}:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model1_str}</p>"
    html += "</div>"
    
    html += f"<div style='margin-bottom: 15px; padding: 10px; background: #f0f8e8; border-left: 4px solid #27ae60; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model2_name}:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model2_str}</p>"
    html += "</div>"
    
    html += f"<div style='padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model3_name}:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model3_str}</p>"
    html += "</div>"
    
    html += "</div>"
    
    return html


def create_quadruple_model_comparison(
    model0_name: str,
    model0_phonemes: List[str],
    model1_name: str,
    model1_phonemes: List[str],
    model2_name: str,
    model2_phonemes: List[str],
    model3_name: str,
    model3_phonemes: List[str]
) -> str:
    """
    Create comparison of phonemes from four different models.
    Model 0 is the PRIMARY model (shown first).
    
    Args:
        model0_name: Name of the primary (first) model
        model0_phonemes: List of phonemes from primary model
        model1_name: Name of the second model
        model1_phonemes: List of phonemes from second model
        model2_name: Name of the third model
        model2_phonemes: List of phonemes from third model
        model3_name: Name of the fourth model
        model3_phonemes: List of phonemes from fourth model
        
    Returns:
        HTML string with quadruple model comparison
    """
    model0_str = ' '.join(model0_phonemes) if model0_phonemes else '(none)'
    model1_str = ' '.join(model1_phonemes) if model1_phonemes else '(none)'
    model2_str = ' '.join(model2_phonemes) if model2_phonemes else '(none)'
    model3_str = ' '.join(model3_phonemes) if model3_phonemes else '(none)'
    
    html = "<div style='font-family: monospace; font-size: 14px;'>"
    # Primary model (first, highlighted)
    html += f"<div style='margin-bottom: 15px; padding: 10px; background: #ffe6f0; border-left: 4px solid #e91e63; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model0_name} (PRIMARY):</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model0_str}</p>"
    html += "</div>"
    
    html += f"<div style='margin-bottom: 15px; padding: 10px; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model1_name}:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model1_str}</p>"
    html += "</div>"
    
    html += f"<div style='margin-bottom: 15px; padding: 10px; background: #f0f8e8; border-left: 4px solid #27ae60; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model2_name}:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model2_str}</p>"
    html += "</div>"
    
    html += f"<div style='padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;'>"
    html += f"<p style='margin: 0; font-weight: bold; color: #2c3e50;'>{model3_name}:</p>"
    html += f"<p style='margin: 5px 0 0 0; color: #2c3e50;'>{model3_str}</p>"
    html += "</div>"
    
    html += "</div>"
    
    return html


def _extract_char_colors(html_content: str, text: str) -> Dict[int, str]:
    """
    Extract color for each character position from HTML.
    
    Args:
        html_content: HTML string with colored text
        text: Original text string
        
    Returns:
        Dictionary mapping character index to color (or None if no color)
    """
    import re
    char_colors = {}
    
    # Extract the inner content (remove outer div)
    inner_match = re.search(r'<div[^>]*>(.*)</div>', html_content, re.DOTALL)
    if not inner_match:
        return char_colors
    
    inner_html = inner_match.group(1)
    
    # Build a list of characters with their colors by parsing HTML sequentially
    text_pos = 0
    i = 0
    
    while i < len(inner_html) and text_pos < len(text):
        # Check for span tag
        span_match = re.match(r"<span[^>]*style=['\"]([^'\"]*)['\"][^>]*>(.)</span>", inner_html[i:], re.DOTALL)
        
        if span_match:
            style = span_match.group(1)
            char = span_match.group(2)
            
            # Extract color
            color_match = re.search(r'color:\s*([^;]+)', style)
            if color_match:
                color = color_match.group(1).strip()
                if text_pos < len(text) and text[text_pos] == char:
                    char_colors[text_pos] = color
                    text_pos += 1
            
            i += len(span_match.group(0))
        else:
            # Regular character (not in span, no color)
            char = inner_html[i]
            if text_pos < len(text) and text[text_pos] == char:
                # Character without color
                text_pos += 1
            i += 1
    
    return char_colors


def _apply_bold_to_changed_chars(html_content: str, text: str, changed_positions: set) -> str:
    """
    Apply bold font-weight to characters at changed positions.
    
    Args:
        html_content: HTML string with colored text
        text: Original text string
        changed_positions: Set of character indices that changed color
        
    Returns:
        Modified HTML with bold applied to changed characters
    """
    import re
    
    if not changed_positions:
        return html_content
    
    # Extract the inner content
    inner_match = re.search(r'(<div[^>]*>)(.*)(</div>)', html_content, re.DOTALL)
    if not inner_match:
        return html_content
    
    outer_start = inner_match.group(1)
    inner_html = inner_match.group(2)
    outer_end = inner_match.group(3)
    
    # Process HTML and add bold to changed characters
    text_pos = 0
    result_html = ""
    i = 0
    
    while i < len(inner_html) and text_pos < len(text):
        # Check for span tag
        span_match = re.match(r"(<span[^>]*style=['\"])([^'\"]*)(['\"][^>]*>)(.)(</span>)", inner_html[i:], re.DOTALL)
        
        if span_match:
            open_tag = span_match.group(1)
            style = span_match.group(2)
            quote_style = span_match.group(3)
            char = span_match.group(4)
            close_tag = span_match.group(5)
            
            # Check if this character position changed
            if text_pos in changed_positions:
                # Add or update bold and underline in style
                if 'font-weight' not in style:
                    style = style.rstrip('; ') + '; font-weight: bold'
                else:
                    style = re.sub(r'font-weight:\s*[^;]+', 'font-weight: bold', style)
                # Add underline
                if 'text-decoration' not in style:
                    style = style.rstrip('; ') + '; text-decoration: underline'
                else:
                    style = re.sub(r'text-decoration:\s*[^;]+', 'text-decoration: underline', style)
            
            # Rebuild span
            result_html += open_tag + style + quote_style + char + close_tag
            text_pos += 1
            i += len(span_match.group(0))
        else:
            # Regular character
            char = inner_html[i]
            if text_pos < len(text) and text[text_pos] == char:
                if text_pos in changed_positions:
                    # Character without span but changed - wrap in span with bold and underline
                    result_html += f"<span style='font-weight: bold; text-decoration: underline;'>{char}</span>"
                else:
                    result_html += char
                text_pos += 1
            else:
                result_html += char
            i += 1
    
    # Add remaining HTML
    result_html += inner_html[i:]
    
    # Rebuild outer div
    return outer_start + result_html + outer_end


def create_validation_comparison(
    text: str,
    before_validation_html: str,
    after_validation_html: str,
    enable_validation: bool
) -> str:
    """
    Create comparison view showing sentence before and after optional validation.
    Makes changed characters bold in both versions for easier comparison.
    
    Args:
        text: Original sentence text
        before_validation_html: HTML with colored text before validation (after Hagen-Faes model)
        after_validation_html: HTML with colored text after validation (if enabled)
        enable_validation: Whether validation was enabled
        
    Returns:
        HTML string with both versions side by side for comparison
    """
    if not enable_validation:
        # If validation is disabled, just return the after validation version (which is the same)
        return after_validation_html
    
    # Extract colors for each character position
    before_colors = _extract_char_colors(before_validation_html, text)
    after_colors = _extract_char_colors(after_validation_html, text)
    
    # Find positions where color changed
    changed_positions = set()
    for pos in range(len(text)):
        before_color = before_colors.get(pos)
        after_color = after_colors.get(pos)
        if before_color != after_color:
            changed_positions.add(pos)
    
    # Apply bold to changed characters in both versions for easier comparison
    before_html_with_bold = _apply_bold_to_changed_chars(before_validation_html, text, changed_positions)
    after_html_with_bold = _apply_bold_to_changed_chars(after_validation_html, text, changed_positions)
    
    html = "<div style='padding: 10px; background: #f8f9fa; border-radius: 5px;'>"
    
    # Before validation version (after Hagen-Faes model)
    html += "<div style='margin-bottom: 10px; padding: 7px; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 4px;'>"
    html += "<p style='margin: 0 0 5px 0; font-weight: bold; color: #2c3e50; font-size: 11px; text-align: right;'>Version 1: After Hagen-Faes Model (Before Validation)</p>"
    html += before_html_with_bold
    html += "</div>"
    
    # After validation version (with bold for changed characters)
    html += "<div style='padding: 7px; background: #f0f8e8; border-left: 4px solid #27ae60; border-radius: 4px;'>"
    html += "<p style='margin: 0 0 5px 0; font-weight: bold; color: #2c3e50; font-size: 11px; text-align: right;'>Version 2: After Optional Validation (Double Check)</p>"
    html += after_html_with_bold
    html += "</div>"
    
    html += "</div>"
    
    return html

