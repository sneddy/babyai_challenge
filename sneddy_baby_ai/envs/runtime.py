"""Runtime helpers for environment registration."""

from __future__ import annotations

import contextlib
import io
import warnings


def suppress_babyai_sampling_rejection_logs() -> None:
    """Hide noisy BabyAI mission-generation rejection prints."""

    try:
        from minigrid.envs.babyai.core import roomgrid_level
    except Exception:
        return

    original_print = getattr(roomgrid_level, "print", print)
    if getattr(roomgrid_level, "_sneddy_sampling_log_patch_installed", False):
        return

    def filtered_print(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        if text.startswith("Sampling rejected:") or text.startswith("Timeout during mission generation:"):
            return
        return original_print(*args, **kwargs)

    roomgrid_level.print = filtered_print
    roomgrid_level._sneddy_sampling_log_patch_installed = True

def suppress_known_runtime_warnings() -> None:
    """Hide noisy third-party warnings that are not actionable here."""

    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
        module=r"pygame\.pkgdata",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The package name gym_minigrid has been deprecated in favor of minigrid\..*",
        category=DeprecationWarning,
        module=r"gym\.envs\.registration",
    )


suppress_known_runtime_warnings()

import minigrid

suppress_babyai_sampling_rejection_logs()


@contextlib.contextmanager
def suppress_noisy_env_output():
    """Suppress noisy BabyAI/minigrid generation prints during env creation/reset."""

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def ensure_babyai_envs_registered() -> None:
    """Import minigrid so BabyAI gym ids are registered.

    Gymnasium setups require importing `minigrid` before calling
    `gym.make("BabyAI-...")`.
    """

    with suppress_noisy_env_output():
        _ = minigrid
