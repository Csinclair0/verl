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
import torch.nn as nn
from typing import Optional, Tuple, Dict

from transformers import Gemma3Config

from .modeling_gemma_original import (
    Gemma3ForCausalLM,
    Gemma3MLP,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3Attention,
    Gemma3TextScaledWordEmbedding,
)


class Gemma3PackedSelfAttention(nn.Module):
    """Modified Gemma3 self-attention to work with packed inputs."""
    
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention = Gemma3Attention(config, layer_idx)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_in_batch: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        # For packed sequences, we don't need attention_mask as it's handled by cu_seqlens
        # and flash attention will use this information
        
        # Process positions for rotary embeddings if needed
        if position_embeddings is None:
            # Generate position IDs based on cu_seqlens
            # This requires knowing sequence lengths which can be derived from cu_seqlens
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            position_ids = torch.cat([torch.arange(seq_len) for seq_len in seq_lens])
            
            # Create rotary embeddings
            rope = Gemma3RotaryEmbedding(self.config)
            position_embeddings = rope(hidden_states, position_ids)
        
        # Flash attention with packed sequences doesn't need traditional attention mask
        attention_mask = None
        
        # Call the attention module
        attn_output, _ = self.attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return attn_output


class Gemma3PackedDecoderLayer(nn.Module):
    """Modified Gemma3 decoder layer to work with packed inputs."""
    
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Use original MLP and normalization layers
        self.self_attn = Gemma3PackedSelfAttention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_in_batch: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        # Save residual
        residual = hidden_states
        
        # Input layernorm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention with packed inputs
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
            position_embeddings=position_embeddings,
            **kwargs
        )
        
        # Add residual and post-attention layernorm
        hidden_states = residual + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Pre-feedforward layernorm
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        
        # MLP
        mlp_output = self.mlp(hidden_states)
        
        # Add residual and post-feedforward layernorm
        hidden_states = residual + mlp_output
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        
        return hidden_states


class Gemma3PackedModel(nn.Module):
    """Gemma3 model adapted for packed input sequences."""
    
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=config.pad_token_id,
            embed_scale=config.embed_scale if hasattr(config, "embed_scale") else 1.0
        )
        
        # Layers
        self.layers = nn.ModuleList(
            [Gemma3PackedDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        
        # Final layer norm
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def post_init(self):
        """Initialize weights using the same initialization as the original model."""
        # This method would be called to initialize weights
        # In a full implementation, this would use the same initialization as Gemma3PreTrainedModel
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_in_batch: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with packed inputs.
        
        Args:
            input_ids: Flattened input token IDs of shape (total_nnz,)
            cu_seqlens: Cumulative sequence lengths of shape (batch_size + 1,)
            max_seqlen_in_batch: Maximum sequence length in this batch
            position_embeddings: Optional rotary position embeddings
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
                position_embeddings=position_embeddings,
                **kwargs
            )
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class Gemma3PackedForCausalLM(nn.Module):
    """Gemma3 for causal language modeling with packed inputs."""
    
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        
        # Model
        self.model = Gemma3PackedModel(config)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def post_init(self):
        """Initialize weights and tie embeddings if needed."""
        # This method would be called to initialize weights and tie embeddings if needed
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_in_batch: int,
        labels: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with packed inputs.
        
        Args:
            input_ids: Flattened input token IDs of shape (total_nnz,)
            cu_seqlens: Cumulative sequence lengths of shape (batch_size + 1,)
            max_seqlen_in_batch: Maximum sequence length in this batch
            labels: Optional labels for computing the loss
            position_embeddings: Optional rotary position embeddings
        """
        # Get hidden states from model
        hidden_states = self.model(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
            position_embeddings=position_embeddings,
            **kwargs
        )
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Reshape logits to (batch_size, sequence_length, vocab_size)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
        }
        
        
def convert_gemma3_to_packed(
    model: Gemma3ForCausalLM, 
    config: Gemma3Config
) -> Gemma3PackedForCausalLM:
    """
    Convert a standard Gemma3ForCausalLM model to a packed version.
    
    Args:
        model: The standard Gemma3ForCausalLM model
        config: The configuration
        
    Returns:
        A Gemma3PackedForCausalLM model with copied weights
    """
    packed_model = Gemma3PackedForCausalLM(config)
    
    # Copy weights from embeddings
    packed_model.model.embed_tokens.weight.data.copy_(model.model.embed_tokens.weight.data)
    
    # Copy weights from layers
    for i, layer in enumerate(model.model.layers):
        # Attention weights
        packed_model.model.layers[i].self_attn.attention.q_proj.weight.data.copy_(
            layer.self_attn.q_proj.weight.data
        )
        packed_model.model.layers[i].self_attn.attention.k_proj.weight.data.copy_(
            layer.self_attn.k_proj.weight.data
        )
        packed_model.model.layers[i].self_attn.attention.v_proj.weight.data.copy_(
            layer.self_attn.v_proj.weight.data
        )
        packed_model.model.layers[i].self_attn.attention.o_proj.weight.data.copy_(
            layer.self_attn.o_proj.weight.data
        )
        
        # Normalization weights
        packed_model.model.layers[i].self_attn.attention.q_norm.weight.data.copy_(
            layer.self_attn.q_norm.weight.data
        )
        packed_model.model.layers[i].self_attn.attention.k_norm.weight.data.copy_(
            layer.self_attn.k_norm.weight.data
        )
        
        # MLP weights
        packed_model.model.layers[i].mlp.gate_proj.weight.data.copy_(
            layer.mlp.gate_proj.weight.data
        )
        packed_model.model.layers[i].mlp.up_proj.weight.data.copy_(
            layer.mlp.up_proj.weight.data
        )
        packed_model.model.layers[i].mlp.down_proj.weight.data.copy_(
            layer.mlp.down_proj.weight.data
        )
        
        # Layer norms
        packed_model.model.layers[i].input_layernorm.weight.data.copy_(
            layer.input_layernorm.weight.data
        )
        packed_model.model.layers[i].post_attention_layernorm.weight.data.copy_(
            layer.post_attention_layernorm.weight.data
        )
        packed_model.model.layers[i].pre_feedforward_layernorm.weight.data.copy_(
            layer.pre_feedforward_layernorm.weight.data
        )
        packed_model.model.layers[i].post_feedforward_layernorm.weight.data.copy_(
            layer.post_feedforward_layernorm.weight.data
        )
    
    # Final layer norm
    packed_model.model.norm.weight.data.copy_(model.model.norm.weight.data)
    
    # LM head
    packed_model.lm_head.weight.data.copy_(model.lm_head.weight.data)
    
    return packed_model 