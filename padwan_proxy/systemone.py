import asyncio
import json
import sys
import threading
from collections.abc import Mapping
from typing import Any, Protocol, TypedDict, cast

from padwan_ai.typesafe.models import Question, SystemOneResponse

__all__ = ("Laya", "SystemOneRequest", "load")


class SystemOneRequest(TypedDict):
    """The subset of TypeSafe's `systemone` body the local endpoint answers."""

    state: dict[str, Any]
    questions: dict[str, dict[str, Any]]


class _Agent(Protocol):
    """The slice of `laya.Agent` the proxy uses; laya itself is untyped."""

    cfg: Mapping[str, Any]
    tok: Any
    device: Any

    def system_one(self, state: Any, questions: Any) -> Any: ...


class Laya:
    """A loaded Laya checkpoint answering System One questions one call at a time."""

    def __init__(self, agent: _Agent) -> None:
        self._agent = agent
        # Not reentrant: a CUDA OOM moves the model to the CPU mid-call.
        self._lock = threading.Lock()
        # build_sequence fills head_max_len with the question, then cuts the state tail.
        self.state_budget = max(
            0, agent.cfg.get("max_len", 512) - agent.cfg.get("head_max_len", 192) - 4
        )

    @property
    def device(self) -> str:
        return str(self._agent.device)

    def fits(self, state: object) -> bool:
        """Whether the state survives the window; a cut tail hides the judged call."""
        # Mirrors laya.common.serialize_state.
        text = (
            state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        )
        if mask_token := getattr(self._agent.tok, "mask_token", None):
            text = text.replace(mask_token, " ")
        tokens = self._agent.tok(text, add_special_tokens=False)
        return len(tokens["input_ids"]) <= self.state_budget

    async def system_one(
        self, state: Any, questions: Mapping[str, Question]
    ) -> SystemOneResponse:
        return await asyncio.to_thread(self._predict, state, questions)

    def _predict(
        self, state: Any, questions: Mapping[str, Question]
    ) -> SystemOneResponse:
        # Cancellation of the awaiter must not release a running inference's lock.
        with self._lock:
            return cast("SystemOneResponse", self._agent.system_one(state, questions))


def load(model: str, *, subfolder: str | None = None) -> Laya:
    """Load a Laya checkpoint, downloading weights from Hugging Face on first use."""
    import laya  # pyright: ignore[reportMissingImports]  # the optional `laya` extra

    from .approvals import _QUESTION

    loaded = Laya(laya.load(model, subfolder=subfolder))
    loaded._predict(
        {
            "user_requests": ["Run the unit tests."],
            "cwd": "/workspace",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
        },
        {"approval": _QUESTION},
    )
    # Warmup can move inference to the CPU after a GPU failure.
    print(
        f"laya: {model} ({subfolder or 'root'}) warmed on {loaded.device}",
        file=sys.stderr,
    )
    return loaded
