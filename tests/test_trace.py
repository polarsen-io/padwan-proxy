import pytest

import padwan_proxy.trace as trace


class TestEnableTracing:
    @pytest.mark.parametrize(
        "env, expected",
        [
            pytest.param({"LANGFUSE_PUBLIC_KEY": "pk"}, "langfuse", id="langfuse"),
            pytest.param({}, "otlp", id="otlp"),
        ],
    )
    def test_backend_picked_by_env(self, monkeypatch, env: dict[str, str], expected):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        calls: list[str] = []
        monkeypatch.setattr(trace, "_enable_langfuse", lambda: calls.append("langfuse"))
        monkeypatch.setattr(trace, "_enable_otlp", lambda: calls.append("otlp"))

        trace.enable_tracing()

        assert calls == [expected]

    def test_missing_extra_exits_with_hint(self, monkeypatch, capsys):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

        def _raise() -> None:
            raise ImportError("no opentelemetry")

        monkeypatch.setattr(trace, "_enable_otlp", _raise)

        with pytest.raises(SystemExit):
            trace.enable_tracing()

        assert "trace extra" in capsys.readouterr().out
