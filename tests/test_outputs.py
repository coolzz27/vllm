# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.outputs import RequestOutput, _merge_kv_transfer_params

pytestmark = pytest.mark.cpu_test


def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None


def test_merge_conf_es_stats():
    merged = _merge_kv_transfer_params(
        {"conf_es_stats": {"low": 1, "total": 3, "stats_version": 2}},
        {"conf_es_stats": {"low": 2, "total": 4, "stats_version": 3}},
    )
    assert merged == {"conf_es_stats": {"low": 3, "total": 7, "stats_version": 3}}
