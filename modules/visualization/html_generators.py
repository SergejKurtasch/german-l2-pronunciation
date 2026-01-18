"""
HTML generators for visualization module.

Functions that generate HTML visualizations for pronunciation analysis.
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
from modules.utils import collapse_consecutive_duplicates
from modules.visualization.helpers import align_graphemes_to_phonemes, _extract_char_colors, _apply_bold_to_changed_chars

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
            # NEW: Use grapheme-phoneme alignment instead of linear interpolation
            try:
                # Extract phoneme list for this token
                phoneme_list = [ph['phoneme'] for ph in token_phonemes]
                
                # Align graphemes (characters) to phonemes
                grapheme_to_phoneme = align_graphemes_to_phonemes(token, phoneme_list)
                
                if grapheme_to_phoneme:
                    # Map each grapheme to its color based on phoneme alignment
                    char_offset = 0
                    for grapheme, phoneme_idx in grapheme_to_phoneme:
                        # Get the collapsed phoneme index for color lookup
                        if phoneme_idx < len(token_phonemes):
                            collapsed_ph_idx = token_phonemes[phoneme_idx]['index']
                            
                            if collapsed_ph_idx in collapsed_ph_to_color:
                                color = collapsed_ph_to_color[collapsed_ph_idx]
                                
                                # Apply color to all characters in this grapheme
                                for i in range(len(grapheme)):
                                    char_pos_in_text = token_start + char_offset + i
                                    char_to_color[char_pos_in_text] = color
                        
                        char_offset += len(grapheme)
                else:
                    # Fallback: alignment failed, use word-level coloring with dominant color
                    word_colors = [collapsed_ph_to_color.get(ph['index'], 'gray') 
                                   for ph in token_phonemes 
                                   if ph['index'] in collapsed_ph_to_color]
                    
                    if word_colors:
                        # Use most common color (or first color if tie)
                        from collections import Counter
                        dominant_color = Counter(word_colors).most_common(1)[0][0]
                        
                        for i in range(len(token)):
                            char_pos_in_text = token_start + i
                            char_to_color[char_pos_in_text] = dominant_color
            
            except Exception as e:
                # Fallback on any error: use word-level coloring with dominant color
                import sys
                print(f"Warning: Grapheme-phoneme alignment failed for token '{token}': {e}", file=sys.stderr)
                
                word_colors = [collapsed_ph_to_color.get(ph['index'], 'gray') 
                               for ph in token_phonemes 
                               if ph['index'] in collapsed_ph_to_color]
                
                if word_colors:
                    from collections import Counter
                    dominant_color = Counter(word_colors).most_common(1)[0][0]
                    
                    for i in range(len(token)):
                        char_pos_in_text = token_start + i
                        char_to_color[char_pos_in_text] = dominant_color
    
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
    log_path = Path(__file__).parent.parent / '.cursor' / 'debug.log'
    with open(log_path, 'a') as f:
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
    log_path = Path(__file__).parent.parent / '.cursor' / 'debug.log'
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"visualization.py:create_text_comparison_view_exit","message":"create_text_comparison_view completed","data":{"html_length":len(html)},"timestamp":int(time.time()*1000)})+'\n')
    # #endregion
    
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
    from modules.visualization.helpers import format_phoneme
    
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
        before_validation_html: HTML with colored text before validation (after Hugging Face model)
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
    
    # Before validation version (after Hugging Face model)
    html += "<div style='margin-bottom: 10px; padding: 7px; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 4px;'>"
    html += "<p style='margin: 0 0 5px 0; font-weight: bold; color: #2c3e50; font-size: 11px; text-align: right;'>Version 1: After Hugging Face Model (Before Validation)</p>"
    html += before_html_with_bold
    html += "</div>"
    
    # After validation version (with bold for changed characters)
    html += "<div style='padding: 7px; background: #f0f8e8; border-left: 4px solid #27ae60; border-radius: 4px;'>"
    html += "<p style='margin: 0 0 5px 0; font-weight: bold; color: #2c3e50; font-size: 11px; text-align: right;'>Version 2: After Optional Validation (Double Check)</p>"
    html += after_html_with_bold
    html += "</div>"
    
    html += "</div>"
    
    return html


