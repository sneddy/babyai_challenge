"""Local evaluation helpers for exported submissions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from ..envs.runtime import ensure_babyai_envs_registered


def load_submission_agent(submission_dir: str):
    submission_dir = str(Path(submission_dir).resolve())
    inference_path = Path(submission_dir) / "inference.py"
    spec = importlib.util.spec_from_file_location("submission_inference", inference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load submission inference from {inference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BabyAIAgent(model_dir=str(Path(submission_dir) / "model"))


def evaluate_submission(submission_dir: str, seed_file: str, max_steps: int = 1000) -> dict:
    ensure_babyai_envs_registered()
    agent = load_submission_agent(submission_dir)
    seed_data = json.loads(Path(seed_file).read_text(encoding="utf-8"))
    results = {}

    total_successes = 0
    total_episodes = 0

    for env_name, seeds in seed_data["environments"].items():
        env = gym.make(env_name)
        try:
            successes = []
            lengths = []
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                agent.reset()
                done = False
                steps = 0
                reward = 0.0
                while not done and steps < max_steps:
                    action = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    steps += 1
                successes.append(int(reward > 0))
                lengths.append(steps)
            env_result = {
                "success_rate": float(np.mean(successes) if successes else 0.0),
                "num_successes": int(sum(successes)),
                "num_episodes": len(successes),
                "mean_episode_length": float(np.mean(lengths) if lengths else 0.0),
            }
            results[env_name] = env_result
            total_successes += env_result["num_successes"]
            total_episodes += env_result["num_episodes"]
        finally:
            env.close()

    results["overall_success_rate"] = total_successes / total_episodes if total_episodes else 0.0
    return results
