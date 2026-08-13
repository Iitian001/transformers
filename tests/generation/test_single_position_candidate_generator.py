# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from transformers import GenerationConfig
from transformers.generation.candidate_generator import SinglePositionMultiTokenCandidateGenerator
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.testing_utils import require_torch


class SuppressFirstToken(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.clone()
        scores[:, 0] = -float("inf")
        return scores


class Gemma4AssistantForCausalLM(nn.Module):
    """Minimal assistant stub for exercising the single-position candidate generator."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.generation_config = GenerationConfig(
            num_assistant_tokens=1,
            eos_token_id=3,
            pad_token_id=0,
        )
        self.register_buffer("fixed_logits", torch.tensor([[[4.0, 3.0, 2.0, 1.0]]]))

    @property
    def device(self):
        return self.fixed_logits.device

    def forward(self, inputs_embeds, **kwargs):
        batch_size = inputs_embeds.shape[0]
        return SimpleNamespace(
            logits=self.fixed_logits.expand(batch_size, -1, -1),
            last_hidden_state=torch.zeros(batch_size, 1, 2, device=inputs_embeds.device),
        )


@require_torch
class SinglePositionMultiTokenCandidateGeneratorTest(unittest.TestCase):
    def _make_generator(self, do_sample: bool):
        input_ids = torch.tensor([[1, 2]])
        assistant_model = Gemma4AssistantForCausalLM()
        generation_config = GenerationConfig(
            do_sample=do_sample,
            max_length=4,
            min_length=0,
            eos_token_id=3,
            pad_token_id=0,
        )
        generator = SinglePositionMultiTokenCandidateGenerator(
            input_ids=input_ids,
            assistant_model=assistant_model,
            target_model_input_embeddings=nn.Embedding(4, 2),
            generation_config=generation_config,
            model_kwargs={"attention_mask": torch.ones_like(input_ids)},
            logits_processor=LogitsProcessorList([SuppressFirstToken()]),
        )
        model_outputs = SimpleNamespace(
            hidden_states=(torch.zeros(1, 1, 2),),
            shared_kv_states={},
        )
        model_kwargs = {"attention_mask": torch.ones_like(input_ids)}
        return generator, input_ids, model_kwargs, model_outputs

    def test_greedy_drafting_applies_logits_processors(self):
        generator, input_ids, model_kwargs, model_outputs = self._make_generator(do_sample=False)

        candidate_ids, candidate_logits = generator.get_candidates(
            input_ids=input_ids,
            model_kwargs=model_kwargs,
            model_outputs=model_outputs,
            is_first_iteration=False,
            n_last_matches=0,
        )

        self.assertEqual(candidate_ids[0, -1].item(), 1)
        self.assertTrue(torch.isneginf(candidate_logits[0, 0, 0]))
        torch.testing.assert_close(candidate_logits[0, 0, 1:], torch.tensor([3.0, 2.0, 1.0]))

    def test_sampling_returns_the_distribution_used_to_sample(self):
        generator, input_ids, model_kwargs, model_outputs = self._make_generator(do_sample=True)

        with patch("torch.multinomial", return_value=torch.tensor([[1]])) as multinomial:
            candidate_ids, candidate_logits = generator.get_candidates(
                input_ids=input_ids,
                model_kwargs=model_kwargs,
                model_outputs=model_outputs,
                is_first_iteration=False,
                n_last_matches=0,
            )

        self.assertEqual(candidate_ids[0, -1].item(), 1)
        self.assertTrue(torch.isneginf(candidate_logits[0, 0, 0]))
        expected_scores = torch.tensor([[-float("inf"), 3.0, 2.0, 1.0]])
        expected_probs = torch.softmax(expected_scores, dim=-1)
        torch.testing.assert_close(multinomial.call_args.args[0], expected_probs)
        torch.testing.assert_close(candidate_logits[:, 0, :], expected_scores)
