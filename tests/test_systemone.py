import asyncio
import json
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from gravier.testing import FakeProto, FakeScope

from padwan_proxy.proxy import build_router
from padwan_proxy.systemone import Laya, load

QUESTIONS = {
    "approval": {
        "type": "choice",
        "instructions": "Is the requested local read authorized?",
        "criteria": {"allow": "Authorized local read", "deny": "Forbidden read"},
    }
}
ANSWER = {
    "type": "choice",
    "choice": "allow",
    "confidence": 0.999,
    "probabilities": {"allow": 0.999, "deny": 0.001},
}


class FakeAgent:
    """Stands in for laya.Agent: ~4 characters per token, no weights."""

    cfg = {"max_len": 512, "head_max_len": 192}
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def tok(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text) // 4))}

    def system_one(self, state: Any, questions: Any) -> dict[str, Any]:
        self.calls.append((state, questions))
        return {
            "model": "laya-rl-agent",
            "answers": {name: ANSWER for name in questions},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }


@pytest.fixture
def agent():
    return FakeAgent()


@pytest.fixture
def router(agent):
    return build_router(client=Mock(), model="glm-4.6", laya=Laya(agent))


async def post(router, body: bytes) -> FakeProto:
    proto = FakeProto(body)
    await router.dispatch(FakeScope(method="POST", path="/systemone"), proto)  # type: ignore[arg-type]
    return proto


async def test_no_route_without_a_model():
    router = build_router(client=Mock(), model="glm-4.6")
    proto = await post(router, b'{"state": {}, "questions": {}}')
    assert proto.status == 404


@pytest.mark.parametrize(
    "state, status, called",
    [
        pytest.param({"tool": "Bash"}, 200, True, id="fits"),
        # A cut tail would hide the tool call being judged: refuse, never guess.
        pytest.param({"tool": "x" * 2000}, 413, False, id="over_the_window"),
    ],
)
async def test_state_must_survive_the_window(router, agent, state, status, called):
    proto = await post(router, json.dumps({"state": state, "questions": QUESTIONS}))
    assert proto.status == status
    assert bool(agent.calls) is called
    if called:
        assert json.loads(proto.body)["answers"]["approval"] == ANSWER
        assert agent.calls == [(state, QUESTIONS)]


@pytest.mark.parametrize(
    "body, status",
    [
        pytest.param(b'{"questions": {}}', 400, id="missing_state"),
        pytest.param(b'{"state": "text", "questions": {}}', 400, id="state_not_object"),
        # laya crashes on an empty batch rather than answering nothing.
        pytest.param(b'{"state": {}, "questions": {}}', 400, id="no_questions"),
        pytest.param(b"{", 400, id="invalid_json"),
        pytest.param(b'{"state": {"a": "' + b"x" * 70_000 + b'"}}', 413, id="oversize"),
    ],
)
async def test_malformed_requests_never_reach_the_model(router, agent, body, status):
    proto = await post(router, body)
    assert proto.status == status
    assert agent.calls == []


@pytest.mark.parametrize(
    "question",
    [
        pytest.param({"type": "choice", "criteria": ["a", "b"]}, id="no_instructions"),
        pytest.param(
            {
                "type": "choice",
                "instructions": "Pick",
                "criteria": list(map(str, range(100))),
            },
            id="oversized_head",
        ),
        pytest.param(
            {
                "type": "score",
                "instructions": "Rate",
                "criteria": {"a": "low", "b": "high"},
            },
            id="unordered_score",
        ),
    ],
)
async def test_invalid_questions_never_reach_the_model(router, agent, question):
    response = await post(
        router, json.dumps({"state": {}, "questions": {"q": question}})
    )
    assert response.status == 400
    assert agent.calls == []


async def test_inference_failure_is_not_leaked(router, agent, monkeypatch):
    def fail(state, questions):
        raise RuntimeError("cuda device-side assert at 0x7f")

    monkeypatch.setattr(agent, "system_one", fail)
    proto = await post(router, json.dumps({"state": {}, "questions": QUESTIONS}))
    assert proto.status == 500
    assert "cuda" not in proto.body.decode()


@pytest.mark.parametrize(
    "cfg, budget",
    [
        pytest.param({"max_len": 512, "head_max_len": 192}, 316, id="default"),
        pytest.param({}, 316, id="missing_config"),
        pytest.param({"max_len": 8, "head_max_len": 192}, 0, id="never_negative"),
    ],
)
def test_state_budget_leaves_room_for_the_question(agent, cfg, budget):
    agent.cfg = cfg
    assert Laya(agent).state_budget == budget


async def test_calls_are_serialized(agent):
    """The model is not reentrant: a CUDA OOM moves it to the CPU mid-call."""
    depth = 0
    peak = 0

    def track(state, questions):
        nonlocal depth, peak
        depth += 1
        peak = max(peak, depth)
        try:
            time.sleep(0.01)
            return FakeAgent.system_one(agent, state, questions)
        finally:
            depth -= 1

    agent.system_one = track
    laya = Laya(agent)
    await asyncio.gather(*(laya.system_one({}, QUESTIONS) for _ in range(8)))
    assert peak == 1


async def test_cancelled_caller_does_not_unlock_running_inference(agent):
    started, release, overlapped = (threading.Event() for _ in range(3))
    depth = peak = 0

    def predict(state, questions):
        nonlocal depth, peak
        depth += 1
        peak = max(peak, depth)
        if depth > 1:
            overlapped.set()
        started.set()
        try:
            if not release.wait(3):
                raise TimeoutError("test did not release inference")
            return FakeAgent.system_one(agent, state, questions)
        finally:
            depth -= 1

    agent.system_one = predict
    laya = Laya(agent)
    first = asyncio.create_task(laya.system_one({}, QUESTIONS))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(laya.system_one({}, QUESTIONS))
        assert not await asyncio.to_thread(overlapped.wait, 0.1)
    finally:
        release.set()
    await second
    assert peak == 1


@pytest.mark.parametrize(
    "subfolder",
    [pytest.param(None, id="english"), pytest.param("multilingual", id="multilingual")],
)
def test_load_warms_the_selected_checkpoint(agent, monkeypatch, subfolder):
    loader = Mock(return_value=agent)
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=loader))
    loaded = load("convaiinnovations/laya", subfolder=subfolder)
    loader.assert_called_once_with("convaiinnovations/laya", subfolder=subfolder)
    assert loaded.device == "cpu"
    assert len(agent.calls) == 1
    assert agent.calls[0][1]["approval"]["criteria"].keys() == {"allow", "deny", "ask"}


def test_state_budget_counts_the_normalized_mask_token(agent):
    class Tokenizer:
        mask_token = "[MASK]"

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.replace(self.mask_token, ""))))}

    agent.tok = Tokenizer()
    agent.cfg = {"max_len": 198, "head_max_len": 192}
    assert not Laya(agent).fits("[MASK][MASK][MASK]")
