"""
Helper functions for visualization module.
"""

from typing import List, Dict, Tuple


def _detect_german_graphemes(word: str) -> List[Tuple[str, int, int]]:
    """
    Detect multi-character graphemes in German words.
    
    Args:
        word: German word (can be mixed case)
        
    Returns:
        List of (grapheme, start_pos, end_pos) tuples
        Example: "Ich" -> [('I', 0, 1), ('ch', 1, 3)]
    """
    word_lower = word.lower()
    graphemes = []
    i = 0
    
    # German digraphs and trigraphs that typically represent single phonemes
    # Ordered by priority (longer first, more common patterns prioritized)
    multi_char_graphemes = [
        'tsch',  # trigraph (rare but distinct)
        'sch',   # trigraph - very common
        'ch',    # digraph - very common, single phoneme
        'ie',    # digraph - long i sound
        'ei',    # digraph - diphthong
        'eu',    # digraph - diphthong
        'äu',    # digraph - diphthong
        'au',    # digraph - diphthong
        # Note: Removed 'pf', 'th', 'ng', 'ph', etc. as they often map to 2 phonemes
        # The alignment algorithm will handle these better as separate characters
    ]
    
    while i < len(word):
        matched = False
        
        # Try to match multi-character graphemes (longest first)
        for grapheme in multi_char_graphemes:
            if i + len(grapheme) <= len(word) and word_lower[i:i+len(grapheme)] == grapheme:
                # Keep original case from word
                graphemes.append((word[i:i+len(grapheme)], i, i+len(grapheme)))
                i += len(grapheme)
                matched = True
                break
        
        if not matched:
            # Single character
            graphemes.append((word[i], i, i+1))
            i += 1
    
    return graphemes


def align_graphemes_to_phonemes(
    word: str,
    phonemes: List[str],
    use_g2p: bool = False
) -> List[Tuple[str, int]]:
    """
    Align characters/graphemes in word to their corresponding phonemes using edit distance.
    
    This function creates a character-to-phoneme mapping that respects German orthography
    by detecting multi-character graphemes (ch, sch, ie, ei, etc.) and aligning them
    with phonemes using a simplified edit distance algorithm.
    
    Args:
        word: German word (e.g., "Apfel", "Ich", "Grundlagenstreit")
        phonemes: List of phonemes for this word (e.g., ['a', 'p', 'f', 'ə', 'l'])
        use_g2p: Use G2P to help with alignment (currently unused, reserved for future)
        
    Returns:
        List of (grapheme, phoneme_index) tuples mapping each character/grapheme to a phoneme index
        Example for "Ich" with phonemes ['ɪ', 'ç']:
            [('I', 0), ('ch', 1)]
        Example for "Apfel" with phonemes ['a', 'p', 'f', 'ə', 'l']:
            [('A', 0), ('p', 1), ('f', 2), ('e', 3), ('l', 4)]
    """
    if not word or not phonemes:
        return []
    
    # Detect graphemes (handles multi-character graphemes like 'ch', 'sch', etc.)
    graphemes = _detect_german_graphemes(word)
    
    # Simple heuristic alignment: distribute phonemes across graphemes
    # This works well when grapheme count is similar to phoneme count
    num_graphemes = len(graphemes)
    num_phonemes = len(phonemes)
    
    if num_graphemes == 0 or num_phonemes == 0:
        return []
    
    result = []
    
    if num_graphemes == num_phonemes:
        # Perfect case: one-to-one mapping
        for i, (grapheme, start, end) in enumerate(graphemes):
            result.append((grapheme, i))
    
    elif num_graphemes < num_phonemes:
        # More phonemes than graphemes (e.g., long vowels: "a" -> "aː")
        # Distribute phonemes across graphemes, some graphemes get multiple phonemes
        phonemes_per_grapheme = num_phonemes / num_graphemes
        
        for i, (grapheme, start, end) in enumerate(graphemes):
            # Map this grapheme to the corresponding phoneme index
            phoneme_idx = min(int(i * phonemes_per_grapheme), num_phonemes - 1)
            result.append((grapheme, phoneme_idx))
    
    else:
        # More graphemes than phonemes (e.g., silent letters or multi-char graphemes)
        # Use more sophisticated alignment
        
        # Create a simple dynamic programming alignment
        # dp[g][p] = cost of aligning first g graphemes with first p phonemes
        dp = [[float('inf')] * (num_phonemes + 1) for _ in range(num_graphemes + 1)]
        backtrack = [[None] * (num_phonemes + 1) for _ in range(num_graphemes + 1)]
        
        # Base case: empty alignment
        dp[0][0] = 0
        
        # Fill DP table
        for g in range(num_graphemes + 1):
            for p in range(num_phonemes + 1):
                if dp[g][p] == float('inf'):
                    continue
                
                # Option 1: Match grapheme[g] with phoneme[p]
                if g < num_graphemes and p < num_phonemes:
                    cost = 0  # Assume match (we could add similarity scoring here)
                    if dp[g][p] + cost < dp[g+1][p+1]:
                        dp[g+1][p+1] = dp[g][p] + cost
                        backtrack[g+1][p+1] = (g, p, 'match')
                
                # Option 2: Skip grapheme (silent letter or merged with previous)
                if g < num_graphemes:
                    cost = 1  # Small penalty for skipping
                    if dp[g][p] + cost < dp[g+1][p]:
                        dp[g+1][p] = dp[g][p] + cost
                        backtrack[g+1][p] = (g, p, 'skip_grapheme')
                
                # Option 3: Skip phoneme (one grapheme maps to multiple phonemes)
                # Less common, higher penalty
                if p < num_phonemes:
                    cost = 2
                    if dp[g][p] + cost < dp[g][p+1]:
                        dp[g][p+1] = dp[g][p] + cost
                        backtrack[g][p+1] = (g, p, 'skip_phoneme')
        
        # Backtrack to find alignment
        g, p = num_graphemes, num_phonemes
        alignment = []
        
        while g > 0 or p > 0:
            if backtrack[g][p] is None:
                break
            
            prev_g, prev_p, action = backtrack[g][p]
            
            if action == 'match':
                alignment.append((g-1, prev_p))
            elif action == 'skip_grapheme':
                # Map skipped grapheme to nearest phoneme
                alignment.append((g-1, max(0, prev_p - 1)))
            
            g, p = prev_g, prev_p
        
        alignment.reverse()
        
        # Convert alignment to result format
        for grapheme_idx, phoneme_idx in alignment:
            grapheme, start, end = graphemes[grapheme_idx]
            result.append((grapheme, phoneme_idx))
    
    return result


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


def format_phoneme(ph: str) -> str:
    """
    Format a single phoneme for display, replacing word boundary markers with visible separator.
    
    Args:
        ph: Phoneme string (may contain '||' for word boundaries)
        
    Returns:
        Formatted phoneme string ('|' for word boundaries, original otherwise)
    """
    return '|' if ph == '||' else ph


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
