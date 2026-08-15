# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

from vllm import SamplingParams
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.parallel_sampling import ParentRequest


def test_parent_request_to_output_stream() -> None:
    parent_request = ParentRequest(make_request(SamplingParams(n=2)))
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished
    assert ([output_0], False, None) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ([output_1], False, None) == parent_request.get_outputs(
        "child_id_1", output_1
    )
    assert ([output_0], False, None) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ([output_1], False, None) == parent_request.get_outputs(
        "child_id_1", output_1
    )

    # output_1 finished
    output_1.finish_reason = "ended"
    assert ([output_0], False, None) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ([output_1], False, None) == parent_request.get_outputs(
        "child_id_1", output_1
    )
    # Finished output_1 had already returned, DO NOT returned again
    assert ([output_0], False, None) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False, None)

    # output_0 finished
    output_0.finish_reason = "ended"
    assert ([output_0], True, None) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert parent_request.get_outputs("child_id_1", output_1) == ([], True, None)
    # Finished output_0 had already returned, DO NOT returned again
    assert parent_request.get_outputs("child_id_0", output_0) == ([], True, None)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], True, None)


def test_parent_request_to_output_final_only() -> None:
    parent_request = ParentRequest(
        make_request(SamplingParams(n=2, output_kind=RequestOutputKind.FINAL_ONLY))
    )
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished, return nothing
    assert parent_request.get_outputs("child_id_0", output_0) == ([], False, None)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False, None)
    # output_1 finished, but outputs won't be returned until all child requests finished
    output_1.finish_reason = "ended"
    assert parent_request.get_outputs("child_id_0", output_0) == ([], False, None)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False, None)
    # output_0 finished, as all child requests finished, the output would be returned
    output_0.finish_reason = "ended"
    assert ([output_0, output_1], True, None) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ([output_0, output_1], True, None) == parent_request.get_outputs(
        "child_id_1", output_1
    )


def test_parent_request_aggregates_conf_es_stats() -> None:
    parent = ParentRequest(
        make_request(SamplingParams(n=2, output_kind=RequestOutputKind.FINAL_ONLY))
    )
    parent.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0,
        text="child 0",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )
    output_1 = CompletionOutput(
        index=1,
        text="child 1",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )
    child_0_params = {
        "conf_es_stats": {
            "low": 2,
            "total": 5,
            "eligible": 3,
            "replay_total": 2,
            "replay_changed": 1,
            "stats_version": 3,
        }
    }
    child_1_params = {
        "conf_es_stats": {
            "low": 4,
            "total": 7,
            "eligible": 6,
            "replay_total": 4,
            "replay_changed": 2,
            "stats_version": 2,
        }
    }

    assert parent.get_outputs("child_id_0", output_0, child_0_params) == (
        [],
        False,
        child_0_params,
    )
    outputs, finished, params = parent.get_outputs(
        "child_id_1", output_1, child_1_params
    )
    assert outputs == [output_0, output_1]
    assert finished
    assert params == {
        "conf_es_stats": {
            "low": 6,
            "total": 12,
            "eligible": 9,
            "replay_total": 6,
            "replay_changed": 3,
            "stats_version": 3,
        }
    }


def test_parallel_sampling_child_requests_preserve_session_id() -> None:
    request = make_request(SamplingParams(n=2))
    request.session_id = "session-1"
    parent_request = ParentRequest(request)

    for idx in range(parent_request.n):
        request_id, child_params = parent_request.get_child_info(idx)
        child_request = request if idx == parent_request.n - 1 else copy(request)
        child_request.request_id = request_id
        child_request.sampling_params = child_params

        assert child_request.session_id == "session-1"


def make_request(sampling_params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="parent_id",
        external_req_id="ext_parent_id",
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
