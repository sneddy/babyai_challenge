"""Episode rollout helpers for exported checkpoints and expert bots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from minigrid.utils.baby_ai_bot import BabyAIBot

from ..data.vocabulary import MissionVocabulary
from ..envs.runtime import ensure_babyai_envs_registered, suppress_noisy_env_output
from ..envs.wrappers import preprocess_observation
from ..models.core import build_policy_from_checkpoint


ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


@dataclass
class EpisodeRecord:
    mission: str
    reward: float
    success: bool
    steps: int
    actions: list[int]
    frames: list[Image.Image]
    status: str


class ExportedCheckpointAgent:
    """Inference wrapper for exported .pt checkpoints."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model, checkpoint = build_policy_from_checkpoint(checkpoint_path, map_location=self.device)
        self.recurrent = bool(checkpoint.get("recurrent", False))
        self.vocab = MissionVocabulary.from_dict(checkpoint["vocab"])
        self._state = None

    def reset(self) -> None:
        if self.recurrent and hasattr(self.model, "initial_state"):
            self._state = self.model.initial_state(batch_size=1, device=self.device)
        else:
            self._state = None

    def predict(self, obs: dict, deterministic: bool = True) -> int:
        processed = preprocess_observation(obs, self.vocab)
        tensor_obs = {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for key, value in processed.items()
        }
        with torch.no_grad():
            if self.recurrent:
                action, _value, self._state = self.model.act(
                    tensor_obs,
                    state=self._state,
                    deterministic=deterministic,
                )
            else:
                action, _value = self.model.act(tensor_obs, deterministic=deterministic)
        return int(action.item())


class ExpertBotAgent:
    """Wrapper around BabyAIBot for rollout parity with the policy agent."""

    def __init__(self, env) -> None:
        self.env = env
        self.bot = None

    def reset(self) -> None:
        self.bot = BabyAIBot(self.env)

    def predict(self, obs: dict, deterministic: bool = True) -> int:
        del obs, deterministic
        if self.bot is None:
            raise RuntimeError("Expert bot was used before reset().")
        return int(self.bot.replan())


def _make_env(env_name: str):
    ensure_babyai_envs_registered()
    with suppress_noisy_env_output():
        return gym.make(env_name, render_mode="rgb_array")


def _action_name(action: int) -> str:
    return ACTION_NAMES.get(int(action), str(action))


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_frame(frame: np.ndarray, header: str, status: str, mission: str, footer: str) -> Image.Image:
    image = Image.fromarray(frame)
    width, height = image.size
    measure = Image.new("RGB", (width, 1), color=(255, 255, 255))
    measure_draw = ImageDraw.Draw(measure)
    mission_lines = _wrap_text(measure_draw, f"mission: {mission}", max_width=width - 16)
    top_padding = 8
    line_height = 14
    top_band_height = top_padding * 2 + line_height * (2 + len(mission_lines))
    bottom_band_height = 24

    canvas = Image.new("RGB", (width, height + top_band_height + bottom_band_height), color=(255, 255, 255))
    canvas.paste(image, (0, top_band_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, top_padding), header, fill=(0, 0, 0))
    draw.text((8, top_padding + line_height), f"status: {status}", fill=(0, 0, 0))
    for line_index, line in enumerate(mission_lines):
        y = top_padding + line_height * (line_index + 2)
        draw.text((8, y), line, fill=(0, 0, 0))
    draw.text((8, top_band_height + height + 6), footer, fill=(0, 0, 0))
    return canvas


def rollout_agent(
    *,
    env_name: str,
    seed: int,
    agent,
    label: str,
    max_steps: int = 256,
    env=None,
) -> EpisodeRecord:
    owns_env = env is None
    if env is None:
        env = _make_env(env_name)
    try:
        with suppress_noisy_env_output():
            obs, _info = env.reset(seed=seed)
        agent.reset()
        mission = str(obs["mission"])
        actions: list[int] = []
        frames: list[Image.Image] = []
        reward = 0.0
        done = False
        step_index = 0
        status = "running"

        while not done and step_index < max_steps:
            frame = env.render()
            action = agent.predict(obs, deterministic=True)
            header = f"{label} | {env_name} | seed={seed}"
            footer = f"step={step_index} action={_action_name(action)}"
            frames.append(_draw_frame(frame, header, status, mission, footer))

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            actions.append(int(action))
            step_index += 1

        final_frame = env.render()
        if reward > 0:
            status = "success"
        elif done:
            status = "fail"
        else:
            status = "max_steps"
        frames.append(
            _draw_frame(
                final_frame,
                f"{label} | {env_name} | seed={seed}",
                status,
                mission,
                f"final steps={step_index} reward={reward:.3f} status={status}",
            )
        )
        return EpisodeRecord(
            mission=mission,
            reward=float(reward),
            success=bool(reward > 0),
            steps=step_index,
            actions=actions,
            frames=frames,
            status=status,
        )
    finally:
        if owns_env:
            env.close()


def _combine_frames(left: Image.Image, right: Image.Image) -> Image.Image:
    width = left.width + right.width
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def save_gif(frames: list[Image.Image], output_path: str, duration_ms: int = 180) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("frames must be non-empty")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def save_mp4(frames: list[Image.Image], output_path: str, duration_ms: int = 180) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("frames must be non-empty")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to save mp4 output.")

    fps = max(1.0, 1000.0 / max(duration_ms, 1))
    with tempfile.TemporaryDirectory(prefix="babyai_rollout_frames_") as temp_dir:
        temp_path = Path(temp_dir)
        for frame_index, frame in enumerate(frames):
            frame.save(temp_path / f"frame_{frame_index:05d}.png")
        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.4f}",
            "-i",
            str(temp_path / "frame_%05d.png"),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path


def save_media(frames: list[Image.Image], output_path: str, duration_ms: int = 180, output_format: str = "mp4") -> Path:
    if output_format == "gif":
        return save_gif(frames, output_path, duration_ms=duration_ms)
    if output_format == "mp4":
        return save_mp4(frames, output_path, duration_ms=duration_ms)
    raise ValueError(f"Unsupported output format: {output_format}")


def compare_policy_vs_expert(
    *,
    env_name: str,
    seed: int,
    checkpoint_path: str,
    output_path: str,
    max_steps: int = 256,
    device: str = "cpu",
    duration_ms: int = 180,
    output_format: str = "mp4",
) -> tuple[EpisodeRecord, EpisodeRecord, Path]:
    policy_agent = ExportedCheckpointAgent(checkpoint_path, device=device)
    policy_episode = rollout_agent(
        env_name=env_name,
        seed=seed,
        agent=policy_agent,
        label="policy",
        max_steps=max_steps,
    )

    expert_env = _make_env(env_name)
    try:
        expert_agent = ExpertBotAgent(expert_env)
        expert_episode = rollout_agent(
            env_name=env_name,
            seed=seed,
            agent=expert_agent,
            label="expert",
            max_steps=max_steps,
            env=expert_env,
        )
    finally:
        expert_env.close()

    frame_count = max(len(policy_episode.frames), len(expert_episode.frames))
    policy_frames = policy_episode.frames + [policy_episode.frames[-1]] * (frame_count - len(policy_episode.frames))
    expert_frames = expert_episode.frames + [expert_episode.frames[-1]] * (frame_count - len(expert_episode.frames))
    combined_frames = [
        _combine_frames(policy_frame, expert_frame)
        for policy_frame, expert_frame in zip(policy_frames, expert_frames, strict=True)
    ]
    media_path = save_media(
        combined_frames,
        output_path,
        duration_ms=duration_ms,
        output_format=output_format,
    )
    return policy_episode, expert_episode, media_path
