"""RL training service for single-task and multi-task BabyAI runs."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from ..bc.service import train_bc
from ..evaluation import (
    ProgressEvalCallback,
    compute_adaptive_env_sampling_weights,
    evaluate_env_suite,
    format_sampling_summary,
)
from ...config.loader import get_config
from ...data.vocabulary import MissionVocabulary, TokenizationConfig, build_global_vocab
from ...envs.catalog import ALL_LEADERBOARD_ENVS, LEADERBOARD_TIERS
from ...envs.curriculum import CURRICULUM_STAGES, get_replay_envs, get_stage
from ...envs.wrappers import make_vec_env
from ...models.core import ModelConfig, load_exported_checkpoint, save_exported_checkpoint
from ...models.sb3 import BabyAIFeaturesExtractor, export_feedforward_policy_from_sb3
from ...models.transfer import (
    checkpoint_looks_like_torch_export,
    initialize_feedforward_sb3_from_exported_checkpoint,
)

@dataclass(frozen=True)
class RunSpec:
    run_name: str
    train_env_names: list[str]
    eval_env_names: list[str]
    sampling_weights: dict[str, float] | None
    stage_name: str | None = None
    all_envs: bool = False


@dataclass(frozen=True)
class ArtifactPaths:
    latest_state_path: Path
    best_state_path: Path
    interrupted_state_path: Path
    latest_export_path: Path
    best_export_path: Path
    metadata_path: Path


def _env_stem(env_name: str) -> str:
    return env_name.replace("BabyAI-", "").replace("-v0", "")


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


def _build_sampling_weights(current_envs: list[str], replay_envs: list[str], config: dict) -> dict[str, float] | None:
    if not replay_envs:
        return None

    current_fraction = float(config["curriculum"]["current_stage_fraction"])
    replay_fraction = float(config["curriculum"]["replay_fraction"])
    total = current_fraction + replay_fraction
    if total <= 0:
        return None

    weights: dict[str, float] = {}
    if current_envs:
        normalized_current = current_fraction / total
        for env_name in current_envs:
            weights[env_name] = normalized_current / len(current_envs)
    if replay_envs:
        normalized_replay = replay_fraction / total
        for env_name in replay_envs:
            weights[env_name] = normalized_replay / len(replay_envs)
    return weights


def resolve_run_spec(args, config: dict) -> RunSpec:
    tier_flags = [args.easy, args.moderate, args.hard]
    selector_count = sum(
        bool(value)
        for value in [args.env, args.envs, args.stage, args.all_envs, any(tier_flags)]
    )
    if selector_count > 1:
        raise RuntimeError("Use only one selector source: --env, --envs, --stage, --all-envs, or tier flags.")

    if args.env:
        env_names = [args.env]
        run_name = args.run_name or _env_stem(args.env)
        return RunSpec(
            run_name=run_name,
            train_env_names=env_names,
            eval_env_names=env_names,
            sampling_weights=None,
        )

    if any(tier_flags):
        env_names: list[str] = []
        if args.easy:
            env_names.extend(LEADERBOARD_TIERS["easy"])
        if args.moderate:
            env_names.extend(LEADERBOARD_TIERS["moderate"])
        if args.hard:
            env_names.extend(LEADERBOARD_TIERS["hard"])
        env_names = list(dict.fromkeys(env_names))
        if not env_names:
            raise RuntimeError("Tier flags were provided but no environments were selected.")
        if args.run_name:
            run_name = args.run_name
        else:
            selected_tiers = [
                name
                for name, enabled in [
                    ("easy", args.easy),
                    ("moderate", args.moderate),
                    ("hard", args.hard),
                ]
                if enabled
            ]
            run_name = "_".join(selected_tiers)
        return RunSpec(
            run_name=run_name,
            train_env_names=env_names,
            eval_env_names=list(env_names),
            sampling_weights=None,
        )

    if args.envs:
        env_names = _parse_envs(args.envs)
        if not env_names:
            raise RuntimeError("--envs was provided but no environment names were parsed.")
        run_name = args.run_name or ("multitask" if len(env_names) > 1 else _env_stem(env_names[0]))
        return RunSpec(
            run_name=run_name,
            train_env_names=env_names,
            eval_env_names=list(env_names),
            sampling_weights=None,
        )

    if args.stage:
        current_envs = list(get_stage(args.stage).env_names)
        replay_envs = get_replay_envs(args.stage)
        train_envs = list(dict.fromkeys(current_envs + replay_envs))
        sampling_weights = _build_sampling_weights(current_envs, replay_envs, config)
        return RunSpec(
            run_name=args.run_name or args.stage,
            train_env_names=train_envs,
            eval_env_names=train_envs,
            sampling_weights=sampling_weights,
            stage_name=args.stage,
        )

    if args.all_envs:
        return RunSpec(
            run_name=args.run_name or "leaderboard_multitask",
            train_env_names=list(ALL_LEADERBOARD_ENVS),
            eval_env_names=list(ALL_LEADERBOARD_ENVS),
            sampling_weights=None,
            all_envs=True,
        )

    default_env = CURRICULUM_STAGES[0].env_names[0]
    return RunSpec(
        run_name=args.run_name or _env_stem(default_env),
        train_env_names=[default_env],
        eval_env_names=[default_env],
        sampling_weights=None,
    )


def _tokenization_config_from_config(config: dict) -> TokenizationConfig:
    return TokenizationConfig(**config["tokenizer"])


def _load_or_build_global_vocab(config: dict, seed: int, rebuild: bool = False) -> MissionVocabulary:
    vocab_path = config["artifacts"]["vocab_path"]
    tokenization = _tokenization_config_from_config(config)
    if not rebuild and os.path.exists(vocab_path):
        vocab = MissionVocabulary.load(vocab_path)
        if vocab.tokenization == tokenization:
            return vocab
    return build_global_vocab(
        output_path=vocab_path,
        episodes_per_env=64,
        seed=seed,
        tokenization=tokenization,
    )


def _ensure_bc_checkpoint(
    *,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    config: dict,
    seed: int,
    bc_checkpoint_path: str | None,
    bc_demo_paths: list[str] | None,
    warm_start_path: str | None,
    eval_env_names: list[str] | None,
    bc_min_sampling_proba: float | None,
) -> str | None:
    if bc_checkpoint_path and bc_demo_paths:
        raise RuntimeError("Use either --bc-checkpoint or --bc-demos, not both.")
    if bc_checkpoint_path:
        return bc_checkpoint_path
    if not bc_demo_paths:
        return None

    _load_or_build_global_vocab(config, seed, rebuild=False)
    export_dir = Path(config["artifacts"]["exports"])
    export_dir.mkdir(parents=True, exist_ok=True)
    target_path = export_dir / f"{run_spec.run_name}_bc.pt"
    train_bc(
        demo_path=bc_demo_paths,
        checkpoint_path=str(target_path),
        vocab_path=config["artifacts"]["vocab_path"],
        config_name=config_name,
        model_preset=model_preset,
        seed=seed,
        warm_start_path=warm_start_path,
        eval_env_names=eval_env_names,
        min_sampling_proba=bc_min_sampling_proba,
    )
    return str(target_path)


def _artifact_paths(run_name: str, algorithm: str, artifacts: dict) -> ArtifactPaths:
    checkpoint_ext = ".zip"
    latest_state_path = Path(artifacts["checkpoints"]) / f"{run_name}_latest{checkpoint_ext}"
    best_state_path = Path(artifacts["checkpoints"]) / f"{run_name}_best{checkpoint_ext}"
    interrupted_state_path = Path(artifacts["checkpoints"]) / f"{run_name}_interrupted{checkpoint_ext}"
    latest_export_path = Path(artifacts["exports"]) / f"{run_name}_policy.pt"
    best_export_path = Path(artifacts["exports"]) / f"{run_name}_best.pt"
    metadata_path = Path(artifacts["exports"]) / f"{run_name}_policy.json"
    return ArtifactPaths(
        latest_state_path=latest_state_path,
        best_state_path=best_state_path,
        interrupted_state_path=interrupted_state_path,
        latest_export_path=latest_export_path,
        best_export_path=best_export_path,
        metadata_path=metadata_path,
    )


def _create_ppo_model(config: dict, train_env, vocab, model_config: ModelConfig, seed: int):
    policy_kwargs = {
        "features_extractor_class": BabyAIFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": model_config.features_dim,
            "vocab_size": vocab.size,
            "model_config": model_config.to_dict(),
        },
        "net_arch": config["ppo"]["net_arch"],
    }

    learning_rate = config["ppo"]["learning_rate"]
    end_learning_rate = config["ppo"].get("end_learning_rate")
    learning_rate_period = config["ppo"].get("learning_rate_period")
    if end_learning_rate is not None or learning_rate_period is not None:
        start_lr = float(learning_rate)
        end_lr = float(0.0 if end_learning_rate is None else end_learning_rate)
        total_timesteps = max(int(config["training"]["total_timesteps"]), 1)
        period_timesteps = total_timesteps if learning_rate_period is None else max(int(learning_rate_period), 1)

        def learning_rate(progress_remaining: float) -> float:
            elapsed_timesteps = (1.0 - float(progress_remaining)) * total_timesteps
            cosine_progress = min(max(elapsed_timesteps / period_timesteps, 0.0), 1.0)
            cosine_value = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            return end_lr + (start_lr - end_lr) * cosine_value

    return PPO(
        config["ppo"]["policy"],
        train_env,
        learning_rate=learning_rate,
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        verbose=config["ppo"]["verbose"],
        seed=seed,
        policy_kwargs=policy_kwargs,
        device=config["training"]["device"],
    )


def _resolve_feedforward_model_config(
    *,
    config: dict,
    warm_start_path: str | None,
    train_env,
) -> ModelConfig:
    model_config = ModelConfig.from_dict(config["model"])
    if not warm_start_path:
        return model_config

    if checkpoint_looks_like_torch_export(warm_start_path):
        try:
            payload = load_exported_checkpoint(warm_start_path, map_location="cpu")
            checkpoint_model_config = payload.get("model_config")
            if checkpoint_model_config:
                return ModelConfig.from_dict(checkpoint_model_config)
        except Exception:
            return model_config
        return model_config

    try:
        source_model = PPO.load(warm_start_path, env=train_env, device=config["training"]["device"])
        source_config = getattr(source_model.policy.features_extractor.core, "config", None)
        if source_config is not None:
            return ModelConfig.from_dict(source_config.to_dict())
    except Exception:
        return model_config
    return model_config


def _initialize_feedforward_policy(model, checkpoint_path: str, train_env, device: str) -> None:
    if checkpoint_looks_like_torch_export(checkpoint_path):
        initialize_feedforward_sb3_from_exported_checkpoint(model, checkpoint_path, map_location=device)
        return

    source_model = PPO.load(checkpoint_path, env=train_env, device=device)
    model.policy.load_state_dict(source_model.policy.state_dict(), strict=True)


def _run_initial_warm_start_eval(
    *,
    model,
    run_spec: RunSpec,
    vocab,
    config: dict,
    seed: int,
):
    if len(run_spec.train_env_names) <= 1:
        return None
    env_metrics, train_success_rate, val_success_rate, _train_len, _val_len = evaluate_env_suite(
        predictor=lambda env_name, seeds, max_steps: model_predictor(
            model,
            env_name,
            vocab,
            seeds,
            max_steps,
        ),
        env_names=run_spec.eval_env_names,
        vocab=vocab,
        seed=seed,
        n_eval_episodes=config["training"]["n_eval_episodes"],
    )
    updated_weights = compute_adaptive_env_sampling_weights(
        env_metrics,
        uniform_alpha=float(config["curriculum"]["sampling_uniform_alpha"]),
    )
    print(
        f"[warm-start eval] "
        f"train_sr={train_success_rate:.2%} "
        f"val_sr={val_success_rate:.2%}"
    )
    env_summary = " ".join(
        f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={metrics['val_success_rate']:.0%}"
        for env_name, metrics in env_metrics.items()
    )
    if env_summary:
        print(env_summary)
    sampling_summary = format_sampling_summary(updated_weights)
    if sampling_summary:
        print(f"[warm-start sampling] {sampling_summary}")
    return updated_weights


def model_predictor(model, env_name: str, vocab, seeds: list[int], max_steps: int):
    from ..evaluation import _evaluate_on_seeds

    return _evaluate_on_seeds(model, env_name, vocab, seeds, max_steps=max_steps)


def _save_feedforward_latest(
    *,
    model,
    vocab,
    model_config: ModelConfig,
    artifact_paths: ArtifactPaths,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    total_timesteps_requested: int,
    save_tag: str,
    vocab_path: str,
) -> None:
    model.save(str(artifact_paths.latest_state_path))
    export_policy = export_feedforward_policy_from_sb3(
        model,
        vocab_payload=vocab.to_dict(),
        model_config=model_config.to_dict(),
    )
    save_exported_checkpoint(
        str(artifact_paths.latest_export_path),
        policy_state=export_policy.state_dict(),
        model_config=model_config,
        vocab_payload=vocab.to_dict(),
        recurrent=False,
    )
    metadata = {
        "algorithm": "ppo",
        "run_name": run_spec.run_name,
        "train_envs": run_spec.train_env_names,
        "eval_envs": run_spec.eval_env_names,
        "config_name": config_name,
        "model_preset": model_preset,
        "seed": seed,
        "num_timesteps": model.num_timesteps,
        "total_timesteps_requested": total_timesteps_requested,
        "latest_state_path": str(artifact_paths.latest_state_path),
        "latest_export_path": str(artifact_paths.latest_export_path),
        "vocab_path": vocab_path,
        "save_tag": save_tag,
    }
    artifact_paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _save_feedforward_best(
    *,
    model,
    vocab,
    model_config: ModelConfig,
    artifact_paths: ArtifactPaths,
) -> None:
    model.save(str(artifact_paths.best_state_path))
    export_policy = export_feedforward_policy_from_sb3(
        model,
        vocab_payload=vocab.to_dict(),
        model_config=model_config.to_dict(),
    )
    save_exported_checkpoint(
        str(artifact_paths.best_export_path),
        policy_state=export_policy.state_dict(),
        model_config=model_config,
        vocab_payload=vocab.to_dict(),
        recurrent=False,
    )


def train_feedforward(
    *,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    total_timesteps: int | None,
    resume_path: str | None,
    warm_start_path: str | None,
    bc_checkpoint_path: str | None,
    rebuild_vocab: bool = False,
) -> Path:

    config = get_config(config_name, model_preset=model_preset)
    if total_timesteps is not None:
        config["training"]["total_timesteps"] = total_timesteps
    config["training"]["seed"] = seed

    artifacts = config["artifacts"]
    os.makedirs(artifacts["checkpoints"], exist_ok=True)
    os.makedirs(artifacts["logs"], exist_ok=True)
    os.makedirs(artifacts["exports"], exist_ok=True)

    vocab = _load_or_build_global_vocab(config, seed, rebuild=rebuild_vocab)
    artifact_paths = _artifact_paths(run_spec.run_name, "ppo", artifacts)
    train_env = make_vec_env(
        env_name=run_spec.train_env_names if len(run_spec.train_env_names) > 1 else run_spec.train_env_names[0],
        vocab=vocab,
        n_envs=config["training"]["n_envs"],
        seed=seed,
        monitor_dir=os.path.join(artifacts["checkpoints"], "monitors", run_spec.run_name),
        sampling_weights=run_spec.sampling_weights,
    )
    model_config = _resolve_feedforward_model_config(
        config=config,
        warm_start_path=warm_start_path,
        train_env=train_env,
    )

    if resume_path:
        if checkpoint_looks_like_torch_export(resume_path):
            raise RuntimeError("--resume for PPO expects an SB3 .zip checkpoint, not a torch export.")
        model = PPO.load(resume_path, env=train_env, device=config["training"]["device"])
        model.set_env(train_env)
    else:
        model = _create_ppo_model(config, train_env, vocab, model_config, seed)
        if warm_start_path:
            _initialize_feedforward_policy(model, warm_start_path, train_env, config["training"]["device"])
        if bc_checkpoint_path:
            _initialize_feedforward_policy(model, bc_checkpoint_path, train_env, config["training"]["device"])
        if warm_start_path and len(run_spec.train_env_names) > 1:
            initial_weights = _run_initial_warm_start_eval(
                model=model,
                run_spec=run_spec,
                vocab=vocab,
                config=config,
                seed=seed,
            )
            if initial_weights is not None:
                train_env.env_method("set_sampling_weights", initial_weights)

    def save_on_eval(**eval_metrics):
        if len(run_spec.train_env_names) > 1 and eval_metrics.get("env_metrics"):
            updated_weights = compute_adaptive_env_sampling_weights(
                eval_metrics["env_metrics"],
                uniform_alpha=float(config["curriculum"]["sampling_uniform_alpha"]),
            )
            train_env.env_method("set_sampling_weights", updated_weights)
        if eval_metrics["timesteps"] % config["training"]["save_freq"] == 0:
            _save_feedforward_latest(
                model=model,
                vocab=vocab,
                model_config=model_config,
                artifact_paths=artifact_paths,
                run_spec=run_spec,
                config_name=config_name,
                model_preset=model_preset,
                seed=seed,
                total_timesteps_requested=config["training"]["total_timesteps"],
                save_tag=f"latest_eval_{eval_metrics['timesteps']}",
                vocab_path=artifacts["vocab_path"],
            )
        if eval_metrics.get("is_new_best"):
            _save_feedforward_best(
                model=model,
                vocab=vocab,
                model_config=model_config,
                artifact_paths=artifact_paths,
            )

    progress_callback = ProgressEvalCallback(
        run_name=run_spec.run_name,
        env_names=run_spec.eval_env_names,
        vocab=vocab,
        seed=seed,
        total_timesteps=config["training"]["total_timesteps"],
        eval_freq=config["training"]["eval_freq"],
        n_eval_episodes=config["training"]["n_eval_episodes"],
        sampling_uniform_alpha=float(config["curriculum"]["sampling_uniform_alpha"]),
        log_dir=artifacts["logs"],
        save_on_eval=save_on_eval,
        verbose=1,
    )

    interrupted = False
    try:
        learn_kwargs = {
            "total_timesteps": config["training"]["total_timesteps"],
            "callback": CallbackList([progress_callback]),
        }
        if resume_path:
            remaining_timesteps = max(config["training"]["total_timesteps"] - model.num_timesteps, 0)
            learn_kwargs["total_timesteps"] = remaining_timesteps
            learn_kwargs["reset_num_timesteps"] = False
        model.learn(**learn_kwargs)
    except KeyboardInterrupt:
        interrupted = True
        model.save(str(artifact_paths.interrupted_state_path))
        _save_feedforward_latest(
            model=model,
            vocab=vocab,
            model_config=model_config,
            artifact_paths=artifact_paths,
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps_requested=config["training"]["total_timesteps"],
            save_tag="interrupt",
            vocab_path=artifacts["vocab_path"],
        )

    if not interrupted:
        _save_feedforward_latest(
            model=model,
            vocab=vocab,
            model_config=model_config,
            artifact_paths=artifact_paths,
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps_requested=config["training"]["total_timesteps"],
            save_tag="final",
            vocab_path=artifacts["vocab_path"],
        )

    train_env.close()
    return artifact_paths.latest_export_path


def _selector_args(run_spec: RunSpec) -> str:
    if run_spec.stage_name:
        selector = f"--stage {run_spec.stage_name}"
        if run_spec.run_name != run_spec.stage_name:
            selector += f" --run-name {run_spec.run_name}"
        return selector
    if run_spec.all_envs:
        selector = "--all-envs"
        if run_spec.run_name != "leaderboard_multitask":
            selector += f" --run-name {run_spec.run_name}"
        return selector
    if len(run_spec.train_env_names) == 1:
        selector = f"--env {run_spec.train_env_names[0]}"
        if run_spec.run_name != _env_stem(run_spec.train_env_names[0]):
            selector += f" --run-name {run_spec.run_name}"
        return selector
    return f"--envs {','.join(run_spec.train_env_names)} --run-name {run_spec.run_name}"


def _print_next_steps(run_spec: RunSpec, config_name: str, seed: int, total_timesteps: int, artifacts: ArtifactPaths) -> None:
    print(f"\n[artifacts] latest checkpoint: {artifacts.latest_state_path}")
    print(f"[artifacts] exported submission policy: {artifacts.latest_export_path}")
    print(f"[artifacts] best checkpoint: {artifacts.best_state_path}")
    print(f"[artifacts] best exported policy: {artifacts.best_export_path}")
    print("[next] export submission:")
    print(f"python3 -m sneddy_baby_ai.cli.export_submission --checkpoint {artifacts.latest_export_path}")
    print("[next] export best submission:")
    print(f"python3 -m sneddy_baby_ai.cli.export_submission --checkpoint {artifacts.best_export_path}")
    print("[next] validate submission:")
    print("python3 babyai-ml8103-leaderboard-2026/evaluation/validate_submission.py demo_submission.zip")
    print("[next] local leaderboard evaluation:")
    print(
        "python3 babyai-ml8103-leaderboard-2026/evaluation/evaluate_submission.py "
        "--submission demo_submission.zip --seed-dir babyai-ml8103-leaderboard-2026/eval_seeds"
    )
    print("[next] resume training:")
    print(
        f"python3 -m sneddy_baby_ai.cli.train_rl {_selector_args(run_spec)} --config {config_name} --seed {seed} "
        f"--timesteps {total_timesteps} --resume {artifacts.latest_state_path}"
    )


def run_train_rl(
    *,
    env: str | None = None,
    envs: str | None = None,
    stage: str | None = None,
    all_envs: bool = False,
    easy: bool = False,
    moderate: bool = False,
    hard: bool = False,
    run_name: str | None = None,
    config_name: str = "default",
    model_preset: str | None = None,
    seed: int = 42,
    total_timesteps: int | None = None,
    resume_path: str | None = None,
    warm_start_path: str | None = None,
    bc_checkpoint_path: str | None = None,
    bc_demo_paths: list[str] | None = None,
    bc_min_sampling_proba: float | None = None,
    rebuild_vocab: bool = False,
) -> Path:
    if resume_path and warm_start_path:
        raise RuntimeError("Use either --resume or --warm-start, not both.")

    config = get_config(config_name, model_preset=model_preset)
    selector_args = SimpleNamespace(
        env=env,
        envs=envs,
        stage=stage,
        all_envs=all_envs,
        easy=easy,
        moderate=moderate,
        hard=hard,
        run_name=run_name,
    )
    run_spec = resolve_run_spec(selector_args, config)

    resolved_bc_checkpoint_path = _ensure_bc_checkpoint(
        run_spec=run_spec,
        config_name=config_name,
        model_preset=model_preset,
        config=config,
        seed=seed,
        bc_checkpoint_path=bc_checkpoint_path,
        bc_demo_paths=bc_demo_paths,
        warm_start_path=warm_start_path,
        eval_env_names=run_spec.eval_env_names,
        bc_min_sampling_proba=bc_min_sampling_proba,
    )
    effective_warm_start_path = None if resolved_bc_checkpoint_path is not None else warm_start_path

    export_path = train_feedforward(
        run_spec=run_spec,
        config_name=config_name,
        model_preset=model_preset,
        seed=seed,
        total_timesteps=total_timesteps,
        resume_path=resume_path,
        warm_start_path=effective_warm_start_path,
        bc_checkpoint_path=resolved_bc_checkpoint_path,
        rebuild_vocab=rebuild_vocab,
    )

    resolved_total_timesteps = int(total_timesteps or config["training"]["total_timesteps"])
    artifact_paths = _artifact_paths(run_spec.run_name, "ppo", config["artifacts"])
    _print_next_steps(run_spec, config_name, seed, resolved_total_timesteps, artifact_paths)
    return export_path
