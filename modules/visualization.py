"""
Visualization module for displaying pronunciation analysis results.
"""

from typing import List, Dict, Tuple, Optional


def create_side_by_side_comparison(
    expected_phonemes: List[str],
    recognized_phonemes: List[str],
    aligned_pairs: List[Tuple[Optional[str], Optional[str]]]
) -> str:
    """
    Create side-by-side comparison of expected and recognized phonemes.
    
    Args:
        expected_phonemes: List of expected phoneme strings
        recognized_phonemes: List of recognized phoneme strings
        aligned_pairs: List of aligned pairs from Needleman-Wunsch
        
    Returns:
        HTML string with side-by-side comparison
    """
    # #region agent log
    import json
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"visualization.py:8","message":"create_side_by_side_comparison entry","data":{"expected_phonemes_count":len(expected_phonemes),"recognized_phonemes_count":len(recognized_phonemes),"aligned_pairs_count":len(aligned_pairs),"aligned_pairs_preview":aligned_pairs[:10],"none_in_recognized":sum(1 for exp, rec in aligned_pairs if rec is None)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    html = "<div style='font-family: monospace; font-size: 16px; line-height: 1.8;' data-block-id='side-by-side-comparison'>"
    
    # Note: Expected phonemes are shown in the separate "Expected Phonemes" block
    # Here we only show recognized phonemes with color coding based on alignment
    
    # Build mapping: track which positions in recognized_phonemes are correctly matched
    # The key insight: aligned_pairs contains pairs (expected, recognized) in the order
    # determined by Needleman-Wunsch alignment. When recognized is not None, it corresponds
    # to a phoneme from the recognized_phonemes list in sequential order (gaps in expected
    # don't affect the order of recognized phonemes).
    
    # Initialize all positions as not matched
    match_status = [False] * len(recognized_phonemes)
    
    # Track current index in recognized_phonemes as we iterate through aligned_pairs
    # aligned_pairs may contain gaps (None values), so we need to track the index carefully
    recognized_list_idx = 0
    
    for expected, recognized in aligned_pairs:
        if recognized is not None:
            # This aligned pair has a recognized phoneme (not a gap in recognized sequence)
            # Check if we haven't exceeded the recognized_phonemes list
            if recognized_list_idx < len(recognized_phonemes):
                # The recognized phoneme from alignment should match the one in the list at current position
                # (assuming alignment is correct and order is preserved)
                current_recognized = recognized_phonemes[recognized_list_idx]
                
                # Verify that the recognized phoneme in alignment matches the one in the list at this position
                # This ensures we're comparing the right phoneme at the right position
                if recognized == current_recognized:
                    # Check if it also matches the expected phoneme (correct match)
                    if expected is not None and expected == recognized:
                        match_status[recognized_list_idx] = True
                # Move to next position in recognized_phonemes regardless of match
                # (we've processed this recognized phoneme)
                recognized_list_idx += 1
            # If we've exceeded the list, stop processing (shouldn't happen in correct alignment)
    
    html += "<div>"
    recognized_row_content = []
    # Show all recognized phonemes from the original list with color coding based on position
    for i, ph in enumerate(recognized_phonemes):
        # Check if this position was correctly matched in alignment
        is_matched = match_status[i] if i < len(match_status) else False
        color = 'green' if is_matched else 'red'
        html += f"<span style='color: {color}; padding: 2px 4px;'>{ph}</span> "
        recognized_row_content.append(ph)
    
    html += "</div>"
    
    html += "</div>"
    
    # #region agent log
    recognized_row_str = ' '.join(recognized_row_content)
    import re
    dash_positions = [m.start() for m in re.finditer(r'-', html)]
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"K","location":"visualization.py:46","message":"create_side_by_side_comparison before return","data":{"html_full":html,"recognized_row_str":recognized_row_str,"html_has_dash":'-' in html,"dash_count":html.count('-'),"dash_positions":dash_positions[:30],"recognized_row_dash_count":recognized_row_str.count('-')},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    return html


def create_colored_text(
    text: str,
    aligned_pairs: List[Dict],
    char_to_phoneme_map: Optional[Dict[int, int]] = None
) -> str:
    """
    Create colored text visualization.
    
    Args:
        text: Original text
        aligned_pairs: List of dictionaries with alignment results
        char_to_phoneme_map: Optional mapping from character index to phoneme index
        
    Returns:
        HTML string with colored text
    """
    # Create mapping if not provided
    if char_to_phoneme_map is None:
        char_to_phoneme_map = {}
        phoneme_idx = 0
        for i, char in enumerate(text):
            if char.isalnum() and not char.isspace():
                if phoneme_idx < len(aligned_pairs):
                    char_to_phoneme_map[i] = phoneme_idx
                    phoneme_idx += 1
    
    html = "<div style='font-size: 18px; line-height: 1.8;'>"
    
    for i, char in enumerate(text):
        if char.isspace():
            html += char
        elif i in char_to_phoneme_map:
            phoneme_idx = char_to_phoneme_map[i]
            if phoneme_idx < len(aligned_pairs):
                pair = aligned_pairs[phoneme_idx]
                is_correct = pair.get('is_correct', False)
                is_missing = pair.get('is_missing', False)
                
                if is_missing:
                    color = 'gray'
                elif is_correct:
                    color = 'green'
                else:
                    color = 'red'
                
                html += f"<span style='color: {color}; font-weight: bold;'>{char}</span>"
            else:
                html += char
        else:
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
    
    expected_str = ' '.join(expected_phonemes)
    recognized_str = ' '.join(recognized_phonemes)
    
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
        raw_phonemes: List of raw phoneme strings (before filtering)
        
    Returns:
        HTML string with raw phonemes display
    """
    if not raw_phonemes:
        return "<div style='color: gray; padding: 10px;'>No raw phonemes available.</div>"
    
    raw_str = ' '.join(raw_phonemes)
    
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

