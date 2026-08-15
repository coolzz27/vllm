# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertSampleConfig:
    keep: int | None
    sample_range: int | None
    temperature: float


_expert_sample_config: ContextVar[ExpertSampleConfig | None] = ContextVar(
    "expert_sample_config", default=None
)


@contextmanager
def enable_expert_sample(
    keep: int | None = None,
    sample_range: int | None = None,
    temperature: float = 1.0,
) -> Iterator[None]:
    """Enable request-scoped Expert-Sample routing for all fused MoE layers."""
    if temperature < 0:
        raise ValueError("Expert-Sample temperature must be nonnegative.")
    token = _expert_sample_config.set(
        ExpertSampleConfig(keep, sample_range, float(temperature))
    )
    try:
        yield
    finally:
        _expert_sample_config.reset(token)


def maybe_apply_expert_sample(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    config = _expert_sample_config.get()
    if config is None:
        return router_logits

    num_experts = router_logits.shape[-1]
    keep = top_k // 2 + 1 if config.keep is None else int(config.keep)
    sample_range = min(
        num_experts,
        4 * top_k if config.sample_range is None else int(config.sample_range),
    )
    if not 0 <= keep <= top_k:
        raise ValueError(f"Expert-Sample keep must be in [0, {top_k}], got {keep}.")
    if not top_k <= sample_range <= num_experts:
        raise ValueError(
            "Expert-Sample range must be between top_k "
            f"({top_k}) and num_experts ({num_experts}), got {sample_range}."
        )

    ranked_logits, ranked_indices = torch.topk(
        router_logits.float(), k=sample_range, dim=-1
    )
    selected = ranked_indices[..., :keep]
    tail_slots = top_k - keep
    if tail_slots:
        tail_logits = ranked_logits[..., keep:]
        if config.temperature == 0:
            tail_positions = torch.topk(tail_logits, k=tail_slots, dim=-1).indices
        else:
            uniform = torch.rand_like(tail_logits).clamp_(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            tail_positions = torch.topk(
                tail_logits / config.temperature + gumbel,
                k=tail_slots,
                dim=-1,
            ).indices
        sampled_tail = ranked_indices[..., keep:].gather(-1, tail_positions)
        selected = torch.cat((selected, sampled_tail), dim=-1)

    sampled_logits = torch.full_like(
        router_logits, torch.finfo(router_logits.dtype).min
    )
    return sampled_logits.scatter(-1, selected, router_logits.gather(-1, selected))
