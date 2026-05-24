"""
Multi-model comparison generators for visualization module.

Functions that compare phonemes from multiple models.
"""

from typing import List

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

