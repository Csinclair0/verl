#!/usr/bin/env python
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
import unittest
from transformers import Gemma3Config

from verl.models.gemma3.megatron.modeling_gemma_original import Gemma3ForCausalLM
from verl.models.gemma3.megatron.modeling_gemma_packed import (
    Gemma3PackedForCausalLM, 
    convert_gemma3_to_packed
)


class TestGemma3Packed(unittest.TestCase):
    
    def setUp(self):
        """Set up test case with minimal configuration."""
        # Use a very small config for faster testing
        self.config = Gemma3Config(
            vocab_size=1000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            pad_token_id=0,
            max_position_embeddings=128,
            tie_word_embeddings=True,
        )
        
        # Create original and packed models
        self.original_model = Gemma3ForCausalLM(self.config)
        self.packed_model = convert_gemma3_to_packed(self.original_model, self.config)
        
        # Put models in eval mode for inference
        self.original_model.eval()
        self.packed_model.eval()
    
    def test_embedding_weight_copy(self):
        """Test that embedding weights are correctly copied from original to packed model."""
        original_weight = self.original_model.model.embed_tokens.weight
        packed_weight = self.packed_model.model.embed_tokens.weight
        
        # Check shapes match
        self.assertEqual(original_weight.shape, packed_weight.shape)
        
        # Check weights are equal
        self.assertTrue(torch.allclose(original_weight, packed_weight))
    
    def test_attention_weight_copy(self):
        """Test that attention weights are correctly copied from original to packed model."""
        # Check first layer attention weights
        original_q_proj = self.original_model.model.layers[0].self_attn.q_proj.weight
        packed_q_proj = self.packed_model.model.layers[0].self_attn.attention.q_proj.weight
        
        # Check shapes match
        self.assertEqual(original_q_proj.shape, packed_q_proj.shape)
        
        # Check weights are equal
        self.assertTrue(torch.allclose(original_q_proj, packed_q_proj))
    
    def test_forward_equivalence(self):
        """Test that the packed model produces equivalent outputs to the original model."""
        batch_size = 2
        seq_len = 10
        
        # Create random input IDs
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, seq_len),
            dtype=torch.long
        )
        
        # Create attention mask (all 1s for simplicity)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        # Run original model
        with torch.no_grad():
            original_outputs = self.original_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Convert to packed format
        # Flatten input_ids and create cu_seqlens
        flat_input_ids = input_ids.reshape(-1)
        cu_seqlens = torch.tensor(
            [0] + [seq_len * i for i in range(1, batch_size + 1)],
            dtype=torch.long
        )
        
        # Run packed model
        with torch.no_grad():
            packed_outputs = self.packed_model(
                input_ids=flat_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=seq_len
            )
        
        # Reshape packed outputs to compare with original outputs
        packed_logits_reshaped = packed_outputs['logits'].view(batch_size, seq_len, -1)
        
        # Check shapes match
        self.assertEqual(
            original_outputs.logits.shape, 
            packed_logits_reshaped.shape
        )
        
        # Outputs won't be exactly equal due to different attention implementations,
        # but they should be close for valid inputs
        self.assertTrue(
            torch.allclose(
                original_outputs.logits, 
                packed_logits_reshaped, 
                rtol=1e-3, 
                atol=1e-3
            )
        )


if __name__ == "__main__":
    unittest.main() 