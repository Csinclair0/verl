# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.batch_reward import (
    compute_batch_score_with_token_errors
)


@register("token_error_batch")
class TokenErrorBatchRewardManager:
    """
    Reward manager that distributes word-level errors to token-level rewards
    for GRPO training with machine translation.
    
    This manager extends the batch reward functionality to handle word-level
    errors from QE models and distribute them to specific tokens in the 
    sequence.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score, 
        reward_fn_key="data_source",
        error_weights=None,
        use_token_level_errors=True,
        **reward_kwargs
    ):
        """
        Initialize the TokenErrorBatchRewardManager.

        Args:
            tokenizer: The tokenizer used to decode token IDs and align words 
                to tokens
            num_examine: Number of batches to examine for debugging
            compute_score: Function to compute reward scores  
            reward_fn_key: Key for accessing data source in non-tensor batch
            error_weights: Dict mapping error types to penalty weights
                e.g., {'major': -2.0, 'minor': -1.0, 'critical': -5.0}
            use_token_level_errors: Whether to distribute errors to token level
            **reward_kwargs: Additional keyword arguments for reward 
                computation
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.use_token_level_errors = use_token_level_errors
        
        # Default error weights if not provided
        if error_weights is None:
            error_weights = {
                'major': -2.0,
                'minor': -1.0, 
                'critical': -5.0,
                'accuracy': -1.5,
                'fluency': -1.0,
                'terminology': -2.5
            }
        self.error_weights = error_weights

    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute rewards with token-level error distribution.
        
        Args:
            data: DataProto containing batch information
            return_dict: Whether to return additional debugging information
            
        Returns:
            Token-level reward tensor or dict with additional info
        """
        # Check if rm_scores already exist
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Extract necessary information from data
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        
        # Decode responses for scoring
        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = data.batch["responses"][i][:valid_len]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            responses_str.append(response_str)

        # Get ground truths and extra info
        ground_truths = [
            item.non_tensor_batch["reward_model"].get("ground_truth", None) 
            for item in data
        ]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        # Compute scores with token-level error distribution
        if self.use_token_level_errors:
            reward_tensor = compute_batch_score_with_token_errors(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                data=data,
                tokenizer=self.tokenizer,
                error_weights=self.error_weights,
                use_token_level_errors=True,
                **self.reward_kwargs,
            )
            
            # Store the accuracy scores for backward compatibility
            sequence_scores = reward_tensor.sum(dim=-1)
            data.batch["acc"] = sequence_scores.to(prompt_ids.device)
            
        else:
            # Fall back to regular scoring without token-level distribution
            scores, token_errors = compute_batch_score_with_token_errors(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                data=data,
                tokenizer=self.tokenizer,
                error_weights=self.error_weights,
                use_token_level_errors=False,
                **self.reward_kwargs,
            )
            
            # Apply scores to last tokens (traditional approach)
            reward_tensor = torch.zeros_like(
                data.batch["responses"], dtype=torch.float32
            )
            for i, score in enumerate(scores):
                length = valid_response_lengths[i].item()
                reward_tensor[i, length - 1] = score
                
            data.batch["acc"] = torch.tensor(
                scores, dtype=torch.float32, device=prompt_ids.device
            )

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor

    def set_error_weights(self, error_weights: dict):
        """
        Update the error weights for penalty calculation.
        
        Args:
            error_weights: Dict mapping error types to penalty weights
        """
        self.error_weights = error_weights

    def enable_token_level_errors(self, enabled: bool = True):
        """
        Enable or disable token-level error distribution.
        
        Args:
            enabled: Whether to use token-level error distribution
        """
        self.use_token_level_errors = enabled 