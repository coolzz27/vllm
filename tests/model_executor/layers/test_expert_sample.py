# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.expert_sample import (
    enable_expert_sample,
    maybe_apply_expert_sample,
)


def test_expert_sample_disabled_is_identity() -> None:
    logits = torch.randn(2, 8)

    assert maybe_apply_expert_sample(logits, top_k=4) is logits


def test_expert_sample_zero_temperature_matches_topk() -> None:
    logits = torch.randn(2, 8)
    with enable_expert_sample(keep=2, sample_range=8, temperature=0):
        sampled = maybe_apply_expert_sample(logits, top_k=4)

    selected = sampled != torch.finfo(sampled.dtype).min
    expected_ids = logits.topk(4, dim=-1).indices
    expected = torch.zeros_like(selected).scatter(-1, expected_ids, True)
    assert torch.equal(selected, expected)
    assert torch.equal(sampled[selected], logits[selected])


def test_expert_sample_preserves_head_and_samples_only_from_range() -> None:
    logits = torch.arange(9, -1, -1, dtype=torch.float32).unsqueeze(0)
    torch.manual_seed(7)
    with enable_expert_sample(keep=2, sample_range=6, temperature=1):
        sampled = maybe_apply_expert_sample(logits, top_k=4)

    selected = sampled != torch.finfo(sampled.dtype).min
    selected_ids = selected.nonzero(as_tuple=False)[:, 1].tolist()
    assert selected.sum().item() == 4
    assert {0, 1}.issubset(selected_ids)
    assert set(selected_ids).issubset(set(range(6)))
    assert torch.equal(sampled[selected], logits[selected])


def test_expert_sample_context_is_restored() -> None:
    logits = torch.randn(1, 8)
    with enable_expert_sample(temperature=0):
        sampled = maybe_apply_expert_sample(logits, top_k=4)
        assert sampled is not logits

    assert maybe_apply_expert_sample(logits, top_k=4) is logits
