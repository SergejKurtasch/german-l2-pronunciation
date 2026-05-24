"""
Report generators for visualization module.

Functions that generate detailed reports and analysis views.
"""

from typing import List, Dict, Optional
from pathlib import Path

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
    log_path = Path(__file__).parent.parent / '.cursor' / 'debug.log'
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"visualization.py:416","message":"create_simple_phoneme_comparison entry","data":{"expected_phonemes":expected_phonemes,"recognized_phonemes":recognized_phonemes,"expected_count":len(expected_phonemes),"recognized_count":len(recognized_phonemes)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    # Filter out word boundary markers and replace with visible separator
    from modules.visualization.helpers import format_phoneme
    
    expected_str = ' '.join(format_phoneme(ph) for ph in expected_phonemes)
    recognized_str = ' '.join(format_phoneme(ph) for ph in recognized_phonemes)
    
    # #region agent log
    log_path = Path(__file__).parent.parent / '.cursor' / 'debug.log'
    with open(log_path, 'a') as f:
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
    log_path = Path(__file__).parent.parent / '.cursor' / 'debug.log'
    with open(log_path, 'a') as f:
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

