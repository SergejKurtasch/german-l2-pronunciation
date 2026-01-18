"""
Visualization module for displaying pronunciation analysis results.

This module is split into submodules for better organization:
- helpers: Helper functions for grapheme detection and alignment
- html_generators: Functions that generate HTML visualizations
- report_generators: Functions that generate detailed reports
- multi_model_comparison: Functions for comparing multiple models
"""

# Import all public functions for backward compatibility
from modules.visualization.helpers import (
    align_graphemes_to_phonemes,
)

# Import HTML generators
from modules.visualization.html_generators import (
    create_side_by_side_comparison,
    create_colored_text,
    create_text_comparison_view,
    create_raw_phonemes_display,
    create_validation_comparison,
)

# Import report generators
from modules.visualization.report_generators import (
    create_detailed_report,
    create_simple_phoneme_comparison,
    create_text_with_sources_display,
)

# Import multi-model comparison
from modules.visualization.multi_model_comparison import (
    create_dual_model_comparison,
    create_triple_model_comparison,
    create_quadruple_model_comparison,
)

__all__ = [
    # Helpers
    'align_graphemes_to_phonemes',
    # HTML generators
    'create_side_by_side_comparison',
    'create_colored_text',
    'create_text_comparison_view',
    'create_raw_phonemes_display',
    'create_validation_comparison',
    # Report generators
    'create_detailed_report',
    'create_simple_phoneme_comparison',
    'create_text_with_sources_display',
    # Multi-model comparison
    'create_dual_model_comparison',
    'create_triple_model_comparison',
    'create_quadruple_model_comparison',
]
