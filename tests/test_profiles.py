from contextlib import nullcontext
from pathlib import Path

import pytest
from piou import CommandError

from padwan_proxy.profiles import PROFILES_ENV, expand_profile
from padwan_proxy.proxy import proxy_command

_PROFILES = """
[scw]
backend_url = "https://api.scaleway.ai/v1/"
model = "glm-5.3"
claude_config = "~/.claude-pglm"
context_window = 256000
rich = true
verbose = false

[typo]
modle = "glm-5.3"

[bad_type]
context_window = "big"
verbose = "yes"
"""

_SCW = [
    "--backend-url",
    "https://api.scaleway.ai/v1/",
    "--model",
    "glm-5.3",
    "--claude-config",
    str(Path("~/.claude-pglm").expanduser()),
    "--context-window",
    "256000",
    "--rich",
]
# Profile flags left once -m and --context-window are given
_SCW_UNSET = [
    "--backend-url",
    "https://api.scaleway.ai/v1/",
    "--claude-config",
    str(Path("~/.claude-pglm").expanduser()),
    "--rich",
]


@pytest.mark.parametrize(
    "args, profiles, expected, error",
    [
        pytest.param(
            ["-m", "x", "-v"], _PROFILES, ["-m", "x", "-v"], None, id="no_profile"
        ),
        pytest.param(
            ["--profile", "scw"],
            _PROFILES,
            [*_SCW, "--profile", "scw"],
            None,
            id="fills",
        ),
        pytest.param(
            ["--profile", "scw", "-m", "glm-4.6", "--context-window", "1"],
            _PROFILES,
            [*_SCW_UNSET, "--profile", "scw", "-m", "glm-4.6", "--context-window", "1"],
            None,
            id="given_flags_win_including_aliases",
        ),
        pytest.param(
            ["--profile", "scw"], None, None, "does not exist", id="missing_file"
        ),
        pytest.param(
            ["--profile", "nope"],
            _PROFILES,
            None,
            "available: scw, typo",
            id="unknown_profile",
        ),
        pytest.param(
            ["--profile", "typo"],
            _PROFILES,
            None,
            "unknown option 'modle'",
            id="unknown_option",
        ),
        pytest.param(
            ["--profile", "bad_type"],
            _PROFILES,
            None,
            "invalid value for 'verbose'",
            id="non_bool_for_flag",
        ),
        pytest.param(
            ["--profile"], _PROFILES, None, "needs a profile name", id="missing_name"
        ),
        pytest.param(
            ["--profile", "scw", "--profile", "typo"],
            _PROFILES,
            None,
            "only once",
            id="repeated",
        ),
    ],
)
def test_expand_profile(tmp_path, monkeypatch, args, profiles, expected, error):
    path = tmp_path / "profiles.toml"
    if profiles is not None:
        path.write_text(profiles)
    monkeypatch.setenv(PROFILES_ENV, str(path))
    with nullcontext() if error is None else pytest.raises(CommandError) as exc:
        result = expand_profile(args, proxy_command)
    if error is not None:
        assert error in exc.value.message
        return
    assert result == expected
