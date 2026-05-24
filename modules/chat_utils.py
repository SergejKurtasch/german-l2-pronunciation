"""
Chat history management utilities.
"""

from typing import List, Optional


def normalize_chat_history(chat_history: Optional[List]) -> List:
    """
    Normalize chat history to the format expected by Gradio Chatbot.
    Converts old tuple format [user_msg, assistant_msg] to dict format.
    
    Args:
        chat_history: Chat history in any format (None, list of tuples, or list of dicts)
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    if chat_history is None:
        return []
    
    normalized = []
    for msg in chat_history:
        if isinstance(msg, dict):
            # Already in correct format
            normalized.append(msg)
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            # Old tuple format [user_msg, assistant_msg] - convert to dict format
            user_msg, assistant_msg = msg
            normalized.append({"role": "user", "content": user_msg})
            normalized.append({"role": "assistant", "content": assistant_msg})
        else:
            # Unknown format - skip
            continue
    
    return normalized


def add_to_chat_history(
    chat_history: Optional[List], 
    user_message: str, 
    assistant_message: str
) -> List:
    """
    Add a new message pair to chat history in the correct format.
    
    Args:
        chat_history: Current chat history (any format)
        user_message: User message text
        assistant_message: Assistant message text (HTML)
        
    Returns:
        Updated chat history in normalized format
    """
    normalized = normalize_chat_history(chat_history)
    normalized.append({"role": "user", "content": user_message})
    normalized.append({"role": "assistant", "content": assistant_message})
    return normalized
