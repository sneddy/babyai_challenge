"""Export training artifacts into demo_submission/."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from ..data.vocabulary import MissionVocabulary
from ..models.core import checkpoint_to_json_summary, load_exported_checkpoint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBMISSION_DIR = ROOT / "demo_submission"


def _copy_template(template_name: str, destination: Path) -> None:
    source = ROOT / "demo_submission" / template_name
    if source.resolve() == destination.resolve():
        return
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def export_submission(checkpoint_path: str, output_dir: str = str(DEFAULT_SUBMISSION_DIR), zip_output: str | None = None) -> Path:
    checkpoint = load_exported_checkpoint(checkpoint_path, map_location="cpu")
    output_dir_path = Path(output_dir)
    model_dir = output_dir_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, model_dir / "agent.zip")

    vocab = MissionVocabulary.from_dict(checkpoint["vocab"])
    vocab.save(str(model_dir / "vocab.json"))
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_config": checkpoint["model_config"],
                "recurrent": checkpoint.get("recurrent", False),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (model_dir / "summary.json").write_text(checkpoint_to_json_summary(checkpoint_path), encoding="utf-8")

    _copy_template("inference.py", output_dir_path / "inference.py")
    _copy_template("models.py", output_dir_path / "models.py")
    _copy_template("requirements.txt", output_dir_path / "requirements.txt")

    if zip_output is None:
        zip_output = str(output_dir_path.with_suffix(".zip"))
    zip_path = Path(zip_output)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(output_dir_path.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(output_dir_path))
    return zip_path
