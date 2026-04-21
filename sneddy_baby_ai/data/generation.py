"""Generate lightweight BabyAI bot demonstrations."""

from __future__ import annotations

import contextlib
from pathlib import Path
import signal

import gymnasium as gym
from minigrid.utils.baby_ai_bot import BabyAIBot
import numpy as np

from ..auxiliary.labels import build_aux_targets
from ..auxiliary.specs import get_aux_specs
from ..common.constants import DEMO_SEED_BASE
from ..config.loader import get_config
from ..envs.catalog import ALL_LEADERBOARD_ENVS, LEADERBOARD_TIERS
from ..envs.runtime import ensure_babyai_envs_registered, suppress_noisy_env_output
from ..envs.wrappers import preprocess_observation
from ..training.evaluation import create_progress_bar
from .demos import DemoBatch, save_demo_file
from .vocabulary import MissionVocabulary, TokenizationConfig, build_global_vocab


@contextlib.contextmanager
def _time_limit(seconds: float | None):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(f"demo attempt exceeded {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_bot(env):
    instr_module = type(env.unwrapped.instrs).__module__

    if not instr_module.startswith("minigrid."):
        raise RuntimeError(f"Unsupported BabyAI instruction module: {instr_module}")

    return BabyAIBot(env)


def _env_stem(env_name: str) -> str:
    return env_name.replace("BabyAI-", "").replace("-v0", "").lower()


def _parse_envs(raw_envs: str | None) -> list[str]:
    if not raw_envs:
        return []
    alias_map = {
        "easy": LEADERBOARD_TIERS["easy"],
        "medium": LEADERBOARD_TIERS["moderate"],
        "moderate": LEADERBOARD_TIERS["moderate"],
        "hard": LEADERBOARD_TIERS["hard"],
        "all": ALL_LEADERBOARD_ENVS,
    }
    env_names: list[str] = []
    for item in raw_envs.split(","):
        token = item.strip()
        if not token:
            continue
        expanded = alias_map.get(token.lower())
        if expanded is not None:
            env_names.extend(expanded)
        else:
            env_names.append(token)
    return list(dict.fromkeys(env_names))


def generate_demos(
    env_name: str,
    output_path: str,
    episodes: int,
    vocab: MissionVocabulary,
    seed: int = 0,
    time_limit_sec: float = 5.0,
    aux_preset: str | None = None,
) -> None:
    ensure_babyai_envs_registered()
    aux_specs = get_aux_specs(aux_preset)
    images = []
    mission_tokens = []
    mission_masks = []
    actions = []
    dones = []
    aux_targets_by_name = {spec.name: [] for spec in aux_specs}
    aux_masks_by_name = {spec.name: [] for spec in aux_specs}

    with suppress_noisy_env_output():
        env = gym.make(env_name)
    progress = create_progress_bar(total=episodes, desc=f"demos {_env_stem(env_name)}", unit="ep")
    try:
        successful_episodes = 0
        failed_episodes = 0
        attempt_index = 0
        progress.set_postfix(fails=failed_episodes, attempts=attempt_index)
        while successful_episodes < episodes:
            with suppress_noisy_env_output():
                obs, _ = env.reset(seed=DEMO_SEED_BASE + seed + attempt_index)
            attempt_index += 1
            bot = _build_bot(env)
            done = False
            episode_images = []
            episode_mission_tokens = []
            episode_mission_masks = []
            episode_actions = []
            episode_dones = []
            episode_aux_targets = {spec.name: [] for spec in aux_specs}
            episode_aux_masks = {spec.name: [] for spec in aux_specs}
            episode_success = False

            try:
                with _time_limit(time_limit_sec):
                    while not done:
                        processed = preprocess_observation(obs, vocab)
                        if aux_specs:
                            aux_targets, aux_masks = build_aux_targets(env.unwrapped, aux_specs)
                        action = bot.replan()
                        next_obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        episode_images.append(processed["image"])
                        episode_mission_tokens.append(processed["mission_tokens"])
                        episode_mission_masks.append(processed["mission_mask"])
                        episode_actions.append(action)
                        episode_dones.append(int(done))
                        if aux_specs:
                            for spec in aux_specs:
                                episode_aux_targets[spec.name].append(aux_targets[spec.name])
                                episode_aux_masks[spec.name].append(aux_masks[spec.name])

                        obs = next_obs
                        if done:
                            episode_success = reward > 0
            except TimeoutError:
                episode_success = False

            if not episode_success:
                failed_episodes += 1
                progress.set_postfix(fails=failed_episodes, attempts=attempt_index)
                continue

            images.extend(episode_images)
            mission_tokens.extend(episode_mission_tokens)
            mission_masks.extend(episode_mission_masks)
            actions.extend(episode_actions)
            dones.extend(episode_dones)
            if aux_specs:
                for spec in aux_specs:
                    aux_targets_by_name[spec.name].extend(episode_aux_targets[spec.name])
                    aux_masks_by_name[spec.name].extend(episode_aux_masks[spec.name])
            successful_episodes += 1
            progress.n = successful_episodes
            progress.set_postfix(fails=failed_episodes, attempts=attempt_index)
            progress.refresh()
    finally:
        progress.close()
        env.close()

    batch = DemoBatch(
        image=np.asarray(images, dtype=np.int64),
        mission_tokens=np.asarray(mission_tokens, dtype=np.int64),
        mission_mask=np.asarray(mission_masks, dtype=np.int64),
        action=np.asarray(actions, dtype=np.int64),
        done=np.asarray(dones, dtype=np.int64),
        aux_targets={
            name: np.asarray(values, dtype=np.int64)
            for name, values in aux_targets_by_name.items()
        },
        aux_masks={
            name: np.asarray(values, dtype=np.int64)
            for name, values in aux_masks_by_name.items()
        },
    )
    save_demo_file(output_path, batch)


def generate_demo_suite(
    *,
    env: str | None,
    envs: str | None,
    output: str | None,
    output_dir: str | None,
    episodes: int,
    seed: int,
    vocab_path: str | None,
    config_name: str,
    time_limit_sec: float,
    aux_preset: str | None,
) -> None:
    if bool(env) == bool(envs):
        raise RuntimeError("Use exactly one of --env or --envs.")

    config = get_config(config_name)
    resolved_vocab_path = vocab_path or config["artifacts"]["vocab_path"]
    tokenization = TokenizationConfig(**config["tokenizer"])
    if Path(resolved_vocab_path).exists():
        vocab = MissionVocabulary.load(resolved_vocab_path)
    else:
        vocab = build_global_vocab(
            output_path=resolved_vocab_path,
            episodes_per_env=64,
            seed=seed,
            tokenization=tokenization,
        )

    if env is not None:
        if output is None:
            raise RuntimeError("--output is required when using --env.")
        generate_demos(
            env,
            output,
            episodes,
            vocab=vocab,
            seed=seed,
            time_limit_sec=time_limit_sec,
            aux_preset=aux_preset,
        )
        return

    env_names = _parse_envs(envs)
    if not env_names:
        raise RuntimeError("--envs was provided but no environment names were parsed.")
    target_output_dir = Path(output_dir or config["artifacts"]["demos"])
    target_output_dir.mkdir(parents=True, exist_ok=True)

    for env_index, env_name in enumerate(env_names):
        output_path = target_output_dir / f"{_env_stem(env_name)}.npz"
        generate_demos(
            env_name,
            str(output_path),
            episodes,
            vocab=vocab,
            seed=seed + env_index * 100_000,
            time_limit_sec=time_limit_sec,
            aux_preset=aux_preset,
        )
        print(output_path)
