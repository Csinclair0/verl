#!/usr/bin/env python3
"""
Example demonstrating token-level reward distribution for GRPO training
with machine translation and word-level errors.

This example shows how to:
1. Set up the TokenErrorBatchRewardManager
2. Configure error weights for different error types
3. Integrate with existing GRPO training pipelines
4. Handle word-to-token alignment for different tokenizers
"""

import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.workers.reward_manager import get_reward_manager_cls
from verl.utils.reward_score.token_alignment import (
    find_word_token_alignment,
    distribute_word_errors_to_tokens
)


def example_basic_usage():
    """Basic example of using token-level reward distribution."""
    print("=== Basic Token-Level Reward Distribution Example ===")
    
    # 1. Initialize tokenizer (use your actual tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Example translation and errors from QE model
    text = "Hello world, this is a test translation."
    token_ids = tokenizer.encode(text, return_tensors="pt")[0]
    
    # Example word-level errors from your QE model
    word_errors = {
        "Hello": "minor",    # Minor error in "Hello"
        "test": "major",     # Major error in "test"
        "translation": "critical"  # Critical error in "translation"
    }
    
    # 3. Find word-to-token alignment
    word_to_positions = find_word_token_alignment(text, tokenizer, token_ids)
    print(f"Text: {text}")
    print(f"Word-to-token mapping: {word_to_positions}")
    
    # 4. Distribute errors to tokens
    error_weights = {
        'major': -2.0,
        'minor': -1.0,
        'critical': -5.0
    }
    
    token_penalties = distribute_word_errors_to_tokens(
        word_errors, word_to_positions, len(token_ids), error_weights
    )
    print(f"Token penalties: {token_penalties}")
    
    # 5. Apply base reward (e.g., BLEU score) + token penalties
    base_reward = 15.0  # Example base reward
    final_rewards = torch.zeros_like(token_penalties)
    final_rewards[-1] = base_reward  # Apply base reward to last token
    final_rewards += token_penalties  # Add penalties for error tokens
    
    print(f"Final token-level rewards: {final_rewards}")
    print()


def example_reward_manager_usage():
    """Example of using the TokenErrorBatchRewardManager with GRPO."""
    print("=== TokenErrorBatchRewardManager Example ===")
    
    # 1. Get the reward manager class
    reward_manager_cls = get_reward_manager_cls("token_error_batch")
    
    # 2. Mock tokenizer and compute_score function for this example
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def mock_compute_score(data_sources, solution_strs, ground_truths, 
                           extra_infos):
        """Mock compute score function that returns scores and token errors."""
        scores = []
        token_errors_list = []
        
        for solution_str in solution_strs:
            # Mock scoring logic
            base_score = 20.0 - len(solution_str.split()) * 0.1  
            scores.append(base_score)
            
            # Mock word-level errors
            words = solution_str.split()
            token_errors = {}
            if len(words) > 3:
                token_errors[words[1]] = "minor"  # Second word has minor error
            if len(words) > 5:
                token_errors[words[3]] = "major"  # Fourth word has major error
                
            token_errors_list.append(token_errors)
        
        return scores, token_errors_list
    
    # 3. Initialize the reward manager
    error_weights = {
        'major': -3.0,
        'minor': -1.5,
        'critical': -6.0,
        'accuracy': -2.0,
        'fluency': -1.0,
        'terminology': -4.0
    }
    
    reward_manager = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=mock_compute_score,
        error_weights=error_weights,
        use_token_level_errors=True,
        reward_fn_key="data_source"
    )
    
    # 4. Create mock data batch
    batch_size = 2
    seq_len = 10
    
    # Mock tensors
    prompts = torch.randint(1, 1000, (batch_size, seq_len))
    responses = torch.randint(1, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len * 2)  # prompt + response
    
    # Mock non-tensor data
    non_tensor_batch = {
        "data_source": ["translation"] * batch_size,
        "extra_info": [
            {
                "include_reasoning": False,
                "include_context_in_metrics": True,
                "src": "en",
                "tgt": "de",
                "input": "Hello world",
                "domain": "general"
            }
        ] * batch_size
    }
    
    # Create mock reward_model info
    reward_model_info = [
        {"ground_truth": "Hallo Welt", 
         "reward_model": {"ground_truth": "Hallo Welt"}},
        {"ground_truth": "Guten Tag", 
         "reward_model": {"ground_truth": "Guten Tag"}}
    ]
    
    # 5. Create DataProto
    data = DataProto(
        batch={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask
        },
        non_tensor_batch=non_tensor_batch
    )
    
    # Add reward model info to individual items
    for i, item_info in enumerate(reward_model_info):
        data[i].non_tensor_batch.update(item_info)
    
    print(f"Input batch size: {batch_size}")
    print(f"Response length: {seq_len}")
    
    # 6. Compute token-level rewards
    try:
        reward_tensor = reward_manager(data, return_dict=False)
        print(f"Output reward tensor shape: {reward_tensor.shape}")
        print(f"Sample rewards for first sequence: {reward_tensor[0]}")
        
        # Show which tokens got penalties
        non_zero_positions = torch.nonzero(reward_tensor[0]).squeeze()
        if non_zero_positions.numel() > 0:
            print(f"Positions with non-zero rewards: "
                  f"{non_zero_positions.tolist()}")
        
    except Exception as e:
        print(f"Note: This example uses mock data, so actual reward "
              f"computation may fail: {e}")
        print("In real usage, ensure your data structure matches your "
              "QE model requirements.")
    
    print()


def example_configuration_options():
    """Example showing different configuration options."""
    print("=== Configuration Options Example ===")
    
    # Different error weight configurations for various use cases
    
    # Configuration 1: Conservative penalties
    conservative_weights = {
        'major': -1.0,
        'minor': -0.5,
        'critical': -2.0
    }
    
    # Configuration 2: Aggressive penalties
    aggressive_weights = {
        'major': -5.0,
        'minor': -2.0,
        'critical': -10.0,
        'accuracy': -3.0,
        'fluency': -1.5,
        'terminology': -6.0
    }
    
    # Configuration 3: Task-specific weights
    task_specific_weights = {
        'major': -2.5,
        'minor': -1.0,
        'critical': -7.5,
        'accuracy': -3.0,      # Penalize accuracy errors heavily
        'fluency': -1.0,       # Moderate penalty for fluency
        'terminology': -5.0,   # Heavy penalty for terminology errors
        'style': -0.5,         # Light penalty for style issues
    }
    
    print("Conservative weights:", conservative_weights)
    print("Aggressive weights:", aggressive_weights)
    print("Task-specific weights:", task_specific_weights)
    
    # Example of runtime configuration changes
    print("\n--- Runtime Configuration Example ---")
    print("You can change error weights during training:")
    print("reward_manager.set_error_weights(new_weights)")
    print("reward_manager.enable_token_level_errors(False)  # Fall back to "
          "sequence-level")
    print()


def example_integration_with_ppo():
    """Example showing how to integrate with PPO trainer configuration."""
    print("=== PPO Integration Example ===")
    
    config_example = """
# Example configuration for PPO trainer with token-level rewards

reward_model:
  reward_manager: "token_error_batch"  # Use our new reward manager
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

# The rest of your PPO configuration remains the same
algorithm:
  adv_estimator: "grpo"  # GRPO works well with token-level rewards
  
# Your model and data configurations...
"""
    
    print("Example YAML configuration:")
    print(config_example)
    
    integration_notes = """
Integration Notes:
1. Set reward_manager to "token_error_batch" in your config
2. Configure error_weights based on your QE model's error types
3. Ensure your data preprocessing includes word-level error information
4. The reward manager will automatically handle word-to-token alignment
5. GRPO will receive token-level rewards instead of sequence-level rewards
6. Monitor training stability - you may need to adjust error weights
"""
    
    print(integration_notes)


def main():
    """Run all examples."""
    print("Token-Level Reward Distribution Examples for GRPO")
    print("=" * 60)
    
    example_basic_usage()
    example_reward_manager_usage()
    example_configuration_options()
    example_integration_with_ppo()
    
    print("Examples completed! See the implementation files for more details:")
    print("- verl/utils/reward_score/token_alignment.py")
    print("- verl/workers/reward_manager/token_error_batch.py")
    print("- verl/workers/reward_manager/batch_reward.py")


if __name__ == "__main__":
    main() 