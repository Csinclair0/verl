# Token-Level Reward Distribution for GRPO Training

## Overview

This solution provides a comprehensive implementation for distributing word-level errors from QE models to token-level rewards in GRPO (Group Relative Policy Optimization) training for machine translation. The implementation addresses the key challenges of word-to-token alignment and subword tokenization.

## Problem Statement

You have:
1. A GRPO PPO trainer for machine translation
2. A batched reward function that returns word-level errors (e.g., `{'Hello': 'major', 'World': 'minor'}`)
3. Need to distribute these word-level penalties to specific tokens in the sequence

The challenges are:
- Response text vs token ID alignment
- Subword tokenization where words may span multiple tokens
- Integration with existing GRPO infrastructure

## Solution Components

### 1. Word-to-Token Alignment Utility (`verl/utils/reward_score/token_alignment.py`)

This module provides robust alignment between words and tokens using multiple strategies:

- **Direct substring matching**: Handles common tokenizer prefixes/suffixes
- **Sequential reconstruction**: Attempts to reconstruct words from consecutive tokens  
- **Character-level alignment**: Fallback method using character position mapping

Key functions:
- `find_word_token_alignment()`: Maps words to token positions
- `distribute_word_errors_to_tokens()`: Converts word errors to token penalties
- `compute_token_level_reward_with_errors()`: Complete pipeline for token-level rewards

### 2. Enhanced Batch Reward Manager (`verl/workers/reward_manager/batch_reward.py`)

Extended your existing batch reward manager with:
- `compute_batch_score_with_token_errors()`: New function that handles token-level distribution
- Backward compatibility with existing code
- Option to enable/disable token-level processing

### 3. New Token Error Batch Reward Manager (`verl/workers/reward_manager/token_error_batch.py`)

A complete reward manager specifically designed for token-level error distribution:

- Extends batch reward functionality
- Configurable error weights for different error types
- Automatic word-to-token alignment
- Runtime configuration options
- Backward compatibility with sequence-level rewards

## Usage

### Basic Configuration

In your PPO trainer configuration:

```yaml
reward_model:
  reward_manager: "token_error_batch"  # Use the new reward manager
  reward_kwargs:
    error_weights:
      major: -2.0
      minor: -1.0  
      critical: -5.0
      accuracy: -2.5
      fluency: -1.0
      terminology: -4.0
    use_token_level_errors: true
    reward_fn_key: "data_source"

algorithm:
  adv_estimator: "grpo"  # GRPO works well with token-level rewards
```

### Programmatic Usage

```python
from verl.workers.reward_manager import get_reward_manager_cls

# Initialize the reward manager
reward_manager_cls = get_reward_manager_cls("token_error_batch")
reward_manager = reward_manager_cls(
    tokenizer=your_tokenizer,
    num_examine=1,
    compute_score=your_compute_score_function,
    error_weights={
        'major': -2.0,
        'minor': -1.0,
        'critical': -5.0
    },
    use_token_level_errors=True
)

# Use with your data
reward_tensor = reward_manager(data)
```

### Runtime Configuration

```python
# Adjust error weights during training
reward_manager.set_error_weights({
    'major': -3.0,  # Increase penalty for major errors
    'minor': -0.5,  # Decrease penalty for minor errors
})

# Fall back to sequence-level rewards if needed
reward_manager.enable_token_level_errors(False)
```

## Integration with Existing Code

The solution is designed to integrate seamlessly with your existing setup:

1. **Minimal Changes**: Your existing PPO trainer configuration only needs the reward manager change
2. **Backward Compatibility**: Can fall back to sequence-level rewards
3. **Drop-in Replacement**: Works with existing data preprocessing pipelines
4. **Configurable**: Error weights and behavior can be tuned

## Error Weight Configuration

Different scenarios may require different penalty strategies:

### Conservative (for stable training)
```python
error_weights = {
    'major': -1.0,
    'minor': -0.5,
    'critical': -2.0
}
```

### Aggressive (for strong error feedback)
```python
error_weights = {
    'major': -5.0,
    'minor': -2.0,
    'critical': -10.0
}
```

### Task-Specific (for domain adaptation)
```python
error_weights = {
    'accuracy': -3.0,      # High penalty for accuracy errors
    'terminology': -5.0,   # Very high penalty for term errors  
    'fluency': -1.0,       # Moderate penalty for fluency
    'style': -0.5          # Light penalty for style issues
}
```

## How It Works

1. **QE Model Output**: Your QE model returns `(scores, token_errors)` where `token_errors` contains word-level error maps
2. **Word-Token Alignment**: The system automatically aligns words to token positions using the tokenizer
3. **Error Distribution**: Word-level errors are distributed to their corresponding tokens with configurable weights
4. **Reward Tensor**: A token-level reward tensor is created with base scores and error penalties
5. **GRPO Training**: The PPO trainer receives token-level rewards for fine-grained optimization

## Files Created/Modified

- ✅ `verl/utils/reward_score/token_alignment.py` - Core alignment utilities
- ✅ `verl/workers/reward_manager/token_error_batch.py` - New reward manager
- ✅ `verl/workers/reward_manager/batch_reward.py` - Enhanced with token-level support
- ✅ `verl/workers/reward_manager/__init__.py` - Updated exports
- ✅ `examples/token_level_reward_example.py` - Comprehensive examples

## Key Benefits

1. **Fine-grained Feedback**: Penalties are applied to specific error tokens
2. **Improved Learning**: GRPO can learn to avoid specific error patterns
3. **Flexible Configuration**: Error weights can be tuned for different tasks
4. **Robust Alignment**: Handles various tokenizer types and subword schemes
5. **Backward Compatible**: Can fall back to sequence-level rewards
6. **Production Ready**: Integrated with existing VERL infrastructure

## Testing and Validation

The implementation includes:
- Multiple alignment strategies for robustness
- Fallback mechanisms for edge cases
- Comprehensive examples and documentation
- Runtime configuration options
- Error handling and logging

## Next Steps

1. Update your PPO trainer configuration to use `"token_error_batch"` 
2. Configure error weights based on your QE model's error types
3. Test with a small dataset to validate the word-to-token alignment
4. Monitor training stability and adjust error weights as needed
5. Consider A/B testing against sequence-level rewards to measure improvement

This solution provides a robust, flexible, and production-ready approach to token-level reward distribution for GRPO training with machine translation tasks. 