from contextlib import nullcontext

import pytest

import padwan_proxy.trace as trace


class TestSessionContext:
    @pytest.mark.parametrize(
        ("langfuse_active", "session_id", "propagates"),
        [
            pytest.param(True, "sess-1", True, id="langfuse+session"),
            pytest.param(True, None, False, id="langfuse-no-session"),
            pytest.param(False, "sess-1", False, id="otlp"),
        ],
    )
    def test_propagates_only_with_langfuse_and_session(
        self, monkeypatch, langfuse_active: bool, session_id: str | None, propagates
    ):
        monkeypatch.setattr(trace, "_langfuse_active", langfuse_active)

        ctx = trace.session_context(session_id)

        assert isinstance(ctx, nullcontext) is not propagates
        with ctx:
            pass


class TestEnableTracing:
    @pytest.mark.parametrize(
        "env, expected",
        [
            pytest.param({"LANGFUSE_PUBLIC_KEY": "pk"}, "langfuse", id="langfuse"),
            pytest.param({}, "otlp", id="otlp"),
        ],
    )
    @pytest.mark.parametrize("capture_content", [False, True])
    def test_backend_picked_by_env(
        self, monkeypatch, env: dict[str, str], expected, capture_content
    ):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        calls: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            trace,
            "_enable_langfuse",
            lambda *, capture_content: calls.append(("langfuse", capture_content)),
        )
        monkeypatch.setattr(
            trace,
            "_enable_otlp",
            lambda *, capture_content: calls.append(("otlp", capture_content)),
        )

        trace.enable_tracing(capture_content=capture_content)

        assert calls == [(expected, capture_content)]

    def test_missing_extra_exits_with_hint(self, monkeypatch, capsys):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

        def _raise(*, capture_content: bool) -> None:
            raise ImportError("no opentelemetry")

        monkeypatch.setattr(trace, "_enable_otlp", _raise)

        with pytest.raises(SystemExit):
            trace.enable_tracing()

        assert "trace extra" in capsys.readouterr().out
