import torch
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


def find_word_token_alignment(
    text: str, 
    tokenizer: PreTrainedTokenizer, 
    token_ids: torch.Tensor
) -> Dict[str, List[int]]:
    """
    Create alignment between words in text and their corresponding token 
    positions.
    
    Args:
        text: The decoded text string
        tokenizer: The tokenizer used to encode/decode
        token_ids: The token IDs for the text
        
    Returns:
        Dict mapping words to lists of token positions that comprise them
    """
    # Handle empty inputs
    if not text.strip() or len(token_ids) == 0:
        return {}
    
    # Convert token IDs to individual tokens
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    
    # Reconstruct text from tokens to handle tokenizer-specific formatting
    reconstructed_text = tokenizer.convert_tokens_to_string(tokens)
    
    # Split text into words (simple whitespace split)
    words = text.split()
    word_to_positions = {}
    
    # For each word, find which tokens contribute to it
    for word in words:
        word_positions = []
        
        # Try different approaches to find word-token alignment
        word_positions = _find_word_positions_fuzzy_match(
            word, tokens, tokenizer, reconstructed_text
        )
        
        if word_positions:
            word_to_positions[word] = word_positions
            
    return word_to_positions


def _find_word_positions_fuzzy_match(
    word: str, 
    tokens: List[str], 
    tokenizer: PreTrainedTokenizer,
    reconstructed_text: str
) -> List[int]:
    """
    Find token positions for a word using fuzzy matching approaches.
    """
    word_lower = word.lower()
    positions = []
    
    # Method 1: Direct token substring matching
    for i, token in enumerate(tokens):
        token_clean = token.replace('▁', '').replace('##', '').replace('Ġ', '')
        if (token_clean.lower() in word_lower or 
                word_lower in token_clean.lower()):
            positions.append(i)
    
    # Method 2: Sequential token reconstruction
    if not positions:
        positions = _find_positions_by_reconstruction(word, tokens, tokenizer)
    
    # Method 3: Character-level alignment as fallback
    if not positions:
        positions = _find_positions_by_char_alignment(
            word, tokens, tokenizer, reconstructed_text
        )
        
    return positions


def _find_positions_by_reconstruction(
    word: str, 
    tokens: List[str], 
    tokenizer: PreTrainedTokenizer
) -> List[int]:
    """
    Find token positions by trying to reconstruct the word from consecutive 
    tokens.
    """
    word_lower = word.lower()
    
    for start_idx in range(len(tokens)):
        # Limit search window
        max_end = min(start_idx + 6, len(tokens) + 1)
        for end_idx in range(start_idx + 1, max_end):
            token_subset = tokens[start_idx:end_idx]
            
            # Try to reconstruct text from this token subset
            try:
                # Convert back to IDs and then decode
                token_ids = tokenizer.convert_tokens_to_ids(token_subset)
                reconstructed = tokenizer.decode(
                    token_ids, skip_special_tokens=True
                )
                
                if reconstructed.lower().strip() == word_lower.strip():
                    return list(range(start_idx, end_idx))
                    
                # Also check if the word is contained in the reconstruction
                if word_lower in reconstructed.lower():
                    return list(range(start_idx, end_idx))
                    
            except Exception:
                continue
                
    return []


def _find_positions_by_char_alignment(
    word: str,
    tokens: List[str], 
    tokenizer: PreTrainedTokenizer,
    reconstructed_text: str
) -> List[int]:
    """
    Find token positions using character-level alignment as a fallback.
    """
    word_lower = word.lower()
    reconstructed_lower = reconstructed_text.lower()
    
    # Find word position in reconstructed text
    word_start = reconstructed_lower.find(word_lower)
    if word_start == -1:
        return []
        
    word_end = word_start + len(word)
    
    # Map character positions back to token positions
    char_to_token = _create_char_to_token_mapping(tokens, tokenizer)
    
    positions = set()
    for char_pos in range(word_start, word_end):
        if char_pos in char_to_token:
            positions.add(char_to_token[char_pos])
            
    return sorted(list(positions))


def _create_char_to_token_mapping(
    tokens: List[str], 
    tokenizer: PreTrainedTokenizer
) -> Dict[int, int]:
    """
    Create a mapping from character positions to token indices.
    """
    char_to_token = {}
    char_pos = 0
    
    for token_idx, token in enumerate(tokens):
        # Decode individual token to get its string representation
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Map all characters in this token to the token index
            for _ in range(len(token_str)):
                char_to_token[char_pos] = token_idx
                char_pos += 1
                
        except Exception:
            # Skip problematic tokens
            continue
            
    return char_to_token


def distribute_word_errors_to_tokens(
    word_errors: Dict[str, str],
    word_to_positions: Dict[str, List[int]],
    sequence_length: int,
    error_weights: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    Distribute word-level errors to token-level rewards.
    
    Args:
        word_errors: Dict mapping words to error types (e.g., {'word': 'major'})
        word_to_positions: Dict mapping words to token positions
        sequence_length: Length of the token sequence
        error_weights: Dict mapping error types to penalty weights
        
    Returns:
        Tensor of shape (sequence_length,) with token-level penalties
    """
    if error_weights is None:
        error_weights = {
            'major': -2.0,
            'minor': -1.0,
            'critical': -5.0
        }
    
    token_penalties = torch.zeros(sequence_length, dtype=torch.float32)
    
    for word, error_type in word_errors.items():
        if word in word_to_positions:
            penalty = error_weights.get(error_type, -1.0)
            positions = word_to_positions[word]
            
            # Distribute penalty across all tokens for this word
            for pos in positions:
                if 0 <= pos < sequence_length:
                    token_penalties[pos] += penalty
                    
    return token_penalties


def compute_token_level_reward_with_errors(
    data, 
    token_errors_list: List[Dict[str, str]], 
    base_scores: List[float],
    tokenizer: PreTrainedTokenizer,
    error_weights: Optional[Dict[str, float]] = None,
    return_dict: bool = False
):
    """
    Compute token-level rewards incorporating both base scores and word-level 
    errors.
    
    Args:
        data: DataProto containing batch information
        token_errors_list: List of word error dicts for each sequence
        base_scores: List of base reward scores for each sequence  
        tokenizer: The tokenizer for word-token alignment
        error_weights: Dict mapping error types to penalty weights
        return_dict: Whether to return additional info
        
    Returns:
        Token-level reward tensor or dict containing rewards and extra info
    """
    reward_tensor = torch.zeros_like(
        data.batch["responses"], dtype=torch.float32
    )
    error_info = {}
    
    prompt_ids = data.batch["prompts"]
    prompt_len = prompt_ids.shape[-1]
    attention_mask = data.batch["attention_mask"]
    valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
    
    for i in range(len(data)):
        # Get valid response tokens and decode
        response_ids = data.batch["responses"][i]
        valid_length = valid_response_lengths[i].item()
        valid_response_ids = response_ids[:valid_length]
        
        # Decode response to get text
        response_text = tokenizer.decode(
            valid_response_ids, skip_special_tokens=True
        )
        
        # Get word-to-token alignment
        word_to_positions = find_word_token_alignment(
            response_text, tokenizer, valid_response_ids
        )
        
        # Get base score (typically applied to last token)
        base_score = base_scores[i] if i < len(base_scores) else 0.0
        reward_tensor[i, valid_length - 1] = base_score
        
        # Apply word-level error penalties
        if i < len(token_errors_list) and token_errors_list[i]:
            word_errors = token_errors_list[i]
            token_penalties = distribute_word_errors_to_tokens(
                word_errors, word_to_positions, valid_length, error_weights
            )
            
            # Add penalties to the reward tensor
            reward_tensor[i, :valid_length] += token_penalties
            
            # Store error info for debugging
            error_info[i] = {
                'word_errors': word_errors,
                'word_to_positions': word_to_positions,
                'response_text': response_text
            }
    
    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": {"token_error_info": error_info}
        }
    else:
        return reward_tensor 