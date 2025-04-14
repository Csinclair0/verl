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

import torch.nn as nn
import torch

from verl.models.mcore.gpt_model import gptmodel_forward
from transformers import Gemma3Config
from megatron.core.transformer import TransformerConfig
from verl.utils.model import get_parallel_gptmodel_from_config

# Import any necessary functionality from the original model
from .modeling_gemma_original import (
    Gemma3RMSNorm,
    Gemma3MLP,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
    apply_rotary_pos_emb,
)


class ParallelGemma3ForCausalLMRmPadPP(nn.Module):
    """Parallel Gemma3 model for causal language modeling with Pipeline Parallelism."""

    def __init__(self, config: Gemma3Config, pre_process=True, post_process=True, pack_seqs=False):
        super().__init__()
        self.config = config
        self.pack_seqs = pack_seqs

        tf_config = TransformerConfig(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            ffn_hidden_size=config.intermediate_size,
            kv_channels=None,
            hidden_dropout=config.hidden_dropout,
            attention_dropout=config.attention_dropout,
            normalization="RMSNorm",
            layernorm_epsilon=config.rms_norm_eps,
            init_method_std=config.initializer_range,
            # Gemma-specific parameters
            use_scaled_init=True,
            use_positional_embedding=True,
            positional_embedding_type="rope",
            rotary_base=config.rope_theta if hasattr(config, "rope_theta") else 10000.0,
        )

        self.model = get_parallel_gptmodel_from_config(
            tf_config,
            config,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=config.tie_word_embeddings,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        packed_seq_params=None,
        cu_seqlens=None,
        max_seqlen_in_batch=None,
    ):
        # Handle packed inputs if provided
        if self.pack_seqs and cu_seqlens is not None and max_seqlen_in_batch is not None:
            packed_params = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen_in_batch": max_seqlen_in_batch,
            }
        else:
            packed_params = packed_seq_params

        return gptmodel_forward(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sequence_parallel=self.config.sequence_parallel_enabled
            if hasattr(self.config, "sequence_parallel_enabled") else False,
            pack_seqs=self.pack_seqs,
            packed_seq_params=packed_params,
        )


class ParallelGemma3ForValueRmPadPP(nn.Module):
    """Parallel Gemma3 model for value modeling with Pipeline Parallelism."""

    def __init__(self, config: Gemma3Config, pre_process=True, post_process=True, pack_seqs=False):
        super().__init__()
        self.config = config
        self.pack_seqs = pack_seqs

        tf_config = TransformerConfig(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            ffn_hidden_size=config.intermediate_size,
            kv_channels=None,
            hidden_dropout=config.hidden_dropout,
            attention_dropout=config.attention_dropout,
            normalization="RMSNorm",
            layernorm_epsilon=config.rms_norm_eps,
            init_method_std=config.initializer_range,
            # Gemma-specific parameters
            use_scaled_init=True,
            use_positional_embedding=True,
            positional_embedding_type="rope",
            rotary_base=config.rope_theta if hasattr(config, "rope_theta") else 10000.0,
        )

        self.model = get_parallel_gptmodel_from_config(
            tf_config,
            config,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=False,
            value=True,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        packed_seq_params=None,
        cu_seqlens=None,
        max_seqlen_in_batch=None,
    ):
        # Handle packed inputs if provided
        if self.pack_seqs and cu_seqlens is not None and max_seqlen_in_batch is not None:
            packed_params = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen_in_batch": max_seqlen_in_batch,
            }
        else:
            packed_params = packed_seq_params

        return gptmodel_forward(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sequence_parallel=self.config.sequence_parallel_enabled
            if hasattr(self.config, "sequence_parallel_enabled") else False,
            value_model=True,
            pack_seqs=self.pack_seqs,
            packed_seq_params=packed_params,
        )


class ParallelGemma3ForCausalLMRmPad(nn.Module):
    """Parallel Gemma3 model for causal language modeling without Pipeline Parallelism."""

    def __init__(self, config: Gemma3Config, pre_process=True, post_process=True, pack_seqs=False):
        super().__init__()
        self.config = config
        self.pack_seqs = pack_seqs

        tf_config = TransformerConfig(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            ffn_hidden_size=config.intermediate_size,
            kv_channels=None,
            hidden_dropout=config.hidden_dropout,
            attention_dropout=config.attention_dropout,
            normalization="RMSNorm",
            layernorm_epsilon=config.rms_norm_eps,
            init_method_std=config.initializer_range,
            # Gemma-specific parameters
            use_scaled_init=True,
            use_positional_embedding=True,
            positional_embedding_type="rope",
            rotary_base=config.rope_theta if hasattr(config, "rope_theta") else 10000.0,
        )

        self.model = get_parallel_gptmodel_from_config(
            tf_config,
            config,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=config.tie_word_embeddings,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        packed_seq_params=None,
        cu_seqlens=None,
        max_seqlen_in_batch=None,
    ):
        # Handle packed inputs if provided
        if self.pack_seqs and cu_seqlens is not None and max_seqlen_in_batch is not None:
            packed_params = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen_in_batch": max_seqlen_in_batch,
            }
        else:
            packed_params = packed_seq_params

        return gptmodel_forward(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sequence_parallel=self.config.sequence_parallel_enabled
            if hasattr(self.config, "sequence_parallel_enabled") else False,
            pack_seqs=self.pack_seqs,
            packed_seq_params=packed_params,
        )


class ParallelGemma3ForConditionalGenerationRmPadPP(nn.Module):
    """Parallel Gemma3 model for conditional generation with Pipeline Parallelism."""

    def __init__(self, config: Gemma3Config, pre_process=True, post_process=True, pack_seqs=False):
        super().__init__()
        self.config = config
        self.pack_seqs = pack_seqs
        
        tf_config = TransformerConfig(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            ffn_hidden_size=config.intermediate_size,
            kv_channels=None,
            hidden_dropout=config.hidden_dropout,
            attention_dropout=config.attention_dropout,
            normalization="RMSNorm",
            layernorm_epsilon=config.rms_norm_eps,
            init_method_std=config.initializer_range,
            # Gemma-specific parameters
            use_scaled_init=True,
            use_positional_embedding=True,
            positional_embedding_type="rope",
            rotary_base=config.rope_theta if hasattr(config, "rope_theta") else 10000.0,
        )
        
        self.model = get_parallel_gptmodel_from_config(
            tf_config,
            config,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=config.tie_word_embeddings,
        )
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        packed_seq_params=None,
        cu_seqlens=None,
        max_seqlen_in_batch=None,
        pixel_values=None,
    ):
        # Handle packed inputs if provided
        if self.pack_seqs and cu_seqlens is not None and max_seqlen_in_batch is not None:
            packed_params = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen_in_batch": max_seqlen_in_batch,
            }
        else:
            packed_params = packed_seq_params
            
        # In conditional generation model, additional handling for pixel_values would be needed
        # This would typically be done by a vision encoder followed by projection
        # For verl implementation, this should be handled within gptmodel_forward or related functions
        
        return gptmodel_forward(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sequence_parallel=self.config.sequence_parallel_enabled
            if hasattr(self.config, "sequence_parallel_enabled") else False,
            pack_seqs=self.pack_seqs,
            packed_seq_params=packed_params,
        ) 