from __future__ import annotations

import inspect
import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args, get_type_hints

from piou import CommandError

from .defaults import ENV_PREFIX

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ("PROFILE_FLAG", "PROFILES_ENV", "expand_profile", "profiles_path")

PROFILE_FLAG = "--profile"
PROFILES_ENV = ENV_PREFIX + "PROFILES"


def profiles_path() -> Path:
    """`$PADWAN_PROXY_PROFILES`, else `$XDG_CONFIG_HOME/padwan-proxy/profiles.toml`."""
    if path := os.environ.get(PROFILES_ENV):
        return Path(path).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(config_home) / "padwan-proxy" / "profiles.toml"


def _load_profile(name: str, path: Path) -> dict[str, Any]:
    try:
        profiles = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise CommandError(f"{PROFILE_FLAG} {name}: {path} does not exist") from None
    except tomllib.TOMLDecodeError as e:
        raise CommandError(f"{path}: {e}") from e
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        available = ", ".join(k for k, v in profiles.items() if isinstance(v, dict))
        raise CommandError(
            f"no profile {name!r} in {path} (available: {available or 'none'})"
        )
    return cast("dict[str, Any]", profile)


def _profile_name(args: Sequence[str]) -> str | None:
    positions = [i for i, arg in enumerate(args) if arg == PROFILE_FLAG]
    if not positions:
        return None
    if len(positions) > 1:
        raise CommandError(f"pass {PROFILE_FLAG} only once")
    i = positions[0]
    if i + 1 >= len(args) or args[i + 1].startswith("-"):
        raise CommandError(f"{PROFILE_FLAG} needs a profile name")
    return args[i + 1]


def expand_profile(args: Sequence[str], command: Callable[..., Any]) -> list[str]:
    """Prepend the `--profile` table as flags, skipping options `args` already sets."""
    name = _profile_name(args)
    if name is None:
        return list(args)
    path = profiles_path()
    profile = _load_profile(name, path)

    hints = get_type_hints(command)
    options = {
        p.name: p.default
        for p in inspect.signature(command).parameters.values()
        if p.name != "profile"
    }
    given = {
        p_name
        for p_name, opt in options.items()
        for flag in (*opt.keyword_args, opt.negative_flag)
        if flag and flag in args
    }

    flags: list[str] = []
    for key, value in profile.items():
        opt = options.get(key)
        if opt is None:
            raise CommandError(f"profile {name!r} in {path}: unknown option {key!r}")
        if key in given:
            continue
        flag = max(opt.keyword_args, key=len)
        hint = hints[key]
        if (hint is bool) != isinstance(value, bool) or isinstance(value, (dict, list)):
            raise CommandError(
                f"profile {name!r} in {path}: invalid value for {key!r}: {value!r}"
            )
        if hint is bool:
            if value:
                flags.append(flag)
            elif opt.negative_flag:
                flags.append(opt.negative_flag)
            continue
        if Path in (hint, *get_args(hint)):
            value = Path(value).expanduser()
        flags += [flag, str(value)]
    return [*flags, *args]
