# Copyright 2025 Individual Contributor: Mert Unsal
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

from collections import defaultdict
import random
import torch
# import wandb # Removed as wandb.log and wandb.Table are no longer used directly here

from verl import DataProto


class BatchRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def set_token_penalties(self, token_penalties: dict):
        """
        Set penalties for specific token IDs.
        Args:
            token_penalties: dict mapping token IDs to penalty values
        """
        self.token_penalties = token_penalties

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            # Apply token-level penalties if specified
            if hasattr(self, 'token_penalties'):
                token_level_scores = data.batch["rm_scores"]
                response_ids = data.batch["responses"]
                
                # Apply penalties for each token
                for token_id, penalty in self.token_penalties.items():
                    mask = (response_ids == token_id)
                    token_level_scores[mask] -= penalty
                
                data.batch["rm_scores"] = token_level_scores

            if return_dict:
                # Even if returning early, ensure a consistent structure if detailed dict is expected
                # However, in this branch, we don't have scores/tables computed by this manager.
                # So, we return only what's available. The caller should be aware.
                # Or, we assume if rm_scores is present, this detailed logging is skipped.
                # For now, sticking to minimal changes for this branch.
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data) # This is a list of scores
        rewards = [] # This will hold the primary numeric reward for each item
        
        wandb_tables_payload = [] # To store data for tables

        # Use different names for thresholds to avoid confusion with batch min/max
        low_score_threshold = min(scores) + 0.001 if scores else 0
        high_score_threshold = max(scores) - 0.001 if scores else 0
        
        random_index = -1
        if len(data) > 0:
            random_index = random.randint(0, len(data) - 1)

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            current_score = scores[i] # Can be a dict or a float/int

            numeric_reward_value = 0.0
            if isinstance(current_score, dict):
                numeric_reward_value = current_score.get("score", 0.0) # Ensure there's a default
                for key, value in current_score.items():
                    reward_extra_info[key].append(value)
            else:
                numeric_reward_value = current_score

            rewards.append(numeric_reward_value)
            reward_tensor[i, length - 1] = numeric_reward_value

            # Common data for tables
            response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
            prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
            ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
            table_columns = ["Prompt", "Response", "Ground Truth", "Score", "Type"]

            if i == random_index:
                table_data = [[
                    prompt_str,
                    response_str,
                    ground_truth,
                    current_score, # Log the original score (could be dict)
                    "Random"
                ]]
                wandb_tables_payload.append({
                    "name": "prompt_response_data_random",
                    "columns": table_columns,
                    "data": table_data
                })
            
            if numeric_reward_value < low_score_threshold:
                table_data = [[
                    prompt_str,
                    response_str,
                    ground_truth,
                    current_score,
                    "Low"
                ]]
                wandb_tables_payload.append({
                    "name": "prompt_response_data_low_score",
                    "columns": table_columns,
                    "data": table_data
                })
                
            if numeric_reward_value > high_score_threshold:
                table_data = [[
                    prompt_str,
                    response_str,
                    ground_truth,
                    current_score,
                    "High"
                ]]  
                wandb_tables_payload.append({
                    "name": "prompt_response_data_high_score",
                    "columns": table_columns,
                    "data": table_data
                })
                            
        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        aggregated_batch_metrics = {}
        if rewards: # Use numeric rewards for aggregation
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            aggregated_batch_metrics["batch_mean_reward"] = (
                rewards_tensor.mean().item()
            )
            aggregated_batch_metrics["batch_min_reward"] = (
                rewards_tensor.min().item()
            )
            aggregated_batch_metrics["batch_max_reward"] = (
                rewards_tensor.max().item()
            )
            if len(rewards_tensor) > 1:
                 aggregated_batch_metrics["batch_std_reward"] = (
                     rewards_tensor.std().item()
                 )
            else:
                aggregated_batch_metrics["batch_std_reward"] = 0.0


        if return_dict:
            # Prepare the dictionary to be returned
            # with one key-value pair per line for readability
            # and to avoid line length issues.
            returned_data = {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "acc_tensor": data.batch["acc"],
                "all_scores_list": scores,
                "aggregated_batch_metrics": aggregated_batch_metrics,
                "wandb_tables_payload": wandb_tables_payload,
            }
            return returned_data
        else:
            return reward_tensor
