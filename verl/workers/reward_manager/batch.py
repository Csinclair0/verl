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

import torch
import wandb

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

        scores = self.verify(data)
        rewards = []
        already_printed = {}
        min_score = min(scores) + 0.001
        max_score = max(scores) - 0.001
        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                
                # Log to wandb table
                table_data = [[
                    prompt_str,
                    response_str,
                    ground_truth,
                    scores[i], 
                    "General"
                ]]
                table = wandb.Table(
                    columns=["Prompt", "Response", "Ground Truth", "Score", "Type"],
                    data=table_data
                )
                wandb.log({"prompt_response_data": table})
                already_printed[data_source] = already_printed.get(data_source, 0) + 1
            if scores[i] < min_score:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                
                # Log to wandb table
                table_data = [[
                    prompt_str,
                    response_str,
                    ground_truth,
                    scores[i], 
                    "Low"
                ]]
                table = wandb.Table(
                    columns=["Prompt", "Response", "Ground Truth", "Score", "Type"],
                    data=table_data
                )
                wandb.log({"prompt_response_data": table})

                print(f"Prompt: {prompt_str}\nResponse: {response_str}\nGround Truth: {ground_truth}\nScore: {scores[i]}")
                
            if scores[i] > max_score:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                
                # Log to wandb table
                table_data = [[
                    prompt_str,
                    response_str,
                    ground_truth,
                    scores[i], 
                    "High"
                ]]  
                table = wandb.Table(
                    columns=["Prompt", "Response", "Ground Truth", "Score", "Type"],
                    data=table_data
                )
                wandb.log({"prompt_response_data": table}) 

                print(f"Prompt: {prompt_str}\nResponse: {response_str}\nGround Truth: {ground_truth}\nScore: {scores[i]}")
                
            

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
