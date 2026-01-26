#!/usr/bin/env python3
"""
CLI runner untuk seluruh pipeline SSEG-NASE.

Usage:
    # PowerShell / CMD
    python .\scripts\cli_run.py .\configs\experiments\exp_full_pipeline.json

Konsep:
- config JSON menentukan tahap yang dijalankan dan argumen spesifik.
- setiap tahap dipanggil sebagai subprocess terhadap script yang ada di folder scripts/.
"""
from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DEFAULT_TIMEOUT: int = 60 * 60 * 24  # 24 hours for long runs (can be overridden)

logger = logging.getLogger("cli_run")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
)
logger.addHandler(handler)


@dataclass
class StageCommand:
    name: str
    command: List[str]
    cwd: Optional[Path] = None
    timeout: Optional[int] = None


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_subprocess(cmd: StageCommand) -> int:
    logger.info("Running stage: %s", cmd.name)
    logger.info("Command: %s", " ".join(shlex.quote(a) for a in cmd.command))
    try:
        result = subprocess.run(
            cmd.command,
            cwd=str(cmd.cwd) if cmd.cwd else None,
            check=False,
            stdout=None,  # Inherit parent, show live output
            stderr=None,  # Inherit parent, show live output
            text=True,
            timeout=cmd.timeout or DEFAULT_TIMEOUT,
        )
        logger.info("Exit code: %d", result.returncode)
        if result.returncode != 0:
            logger.error("Stage '%s' failed (exit %d).", cmd.name, result.returncode)
        return result.returncode
    except subprocess.TimeoutExpired as e:
        logger.error("Stage '%s' timed out after %s seconds", cmd.name, e.timeout)
        return 124  # conventional timeout exit code


def ensure_executable_python() -> str:
    return sys.executable


def build_stage_commands(config: Dict[str, Any], base_dir: Path) -> List[StageCommand]:
    py = ensure_executable_python()
    scripts_dir = base_dir / "scripts"
    out_dir = Path(config.get("output_dir", "outputs"))
    data_dir = Path(config.get("data_dir", "datasets"))

    stages: List[StageCommand] = []

    # 1. Generate curriculum (optional)
    if config.get("stages", {}).get("generate_curriculum", True):
        gen_cfg = config.get("generate_curriculum", {})
        args = [
            "--output-dir",
            str(data_dir / "synthetic_curriculum"),
            "--image-size",
            str(gen_cfg.get("image_size", 84)),
            "--seed",
            str(gen_cfg.get("seed", 42)),
            "--samples-multiplier",
            str(gen_cfg.get("samples_multiplier", 1.0)),
        ]
        levels = gen_cfg.get("levels")
        if levels:
            args += [str(x) for x in ["--levels"] + [str(l) for l in levels]]
        cmd = [py, str(scripts_dir / "generate_curriculum.py")] + args
        stages.append(StageCommand(name="generate_curriculum", command=cmd, cwd=base_dir))

    # 2. Train main model
    if config.get("stages", {}).get("train", True):
        train_cfg = config.get("train", {})
        args = [
            "--experiment-name",
            train_cfg.get("experiment_name", "sseg_nase_exp"),
            "--output-dir",
            str(out_dir),
            "--data-dir",
            str(data_dir),
            "--hardware",
            train_cfg.get("hardware", "rtx3060"),
            "--max-epochs",
            str(train_cfg.get("max_epochs", 100)),
            "--seed",
            str(train_cfg.get("seed", 42)),
        ]
        if train_cfg.get("config"):
            args += ["--config", str(train_cfg["config"])]
        if train_cfg.get("resume"):
            args += ["--resume", str(train_cfg["resume"])]
        if train_cfg.get("debug", False):
            args += ["--debug"]
        cmd = [py, str(scripts_dir / "train_sseg.py")] + args
        stages.append(StageCommand(name="train", command=cmd, cwd=base_dir))

    # 3. Evaluate
    if config.get("stages", {}).get("evaluate", True):
        eval_cfg = config.get("evaluate", {})
        checkpoint = eval_cfg.get(
            "checkpoint", str(out_dir / train_cfg.get("experiment_name", "sseg_nase_exp") / "checkpoints" / "final_model.pt")
        )
        args = [
            "--checkpoint",
            checkpoint,
            "--data-dir",
            str(eval_cfg.get("data_dir", data_dir / "minimagenet")),
            "--output-dir",
            str(out_dir / "evaluation"),
            "--num-ways",
            str(eval_cfg.get("num_ways", 5)),
            "--num-shots",
        ]
        # allow list for num_shots
        shots = eval_cfg.get("num_shots", [1, 5])
        if isinstance(shots, list):
            args += [str(s) for s in shots]
        else:
            args.append(str(shots))
        args += [
            "--num-episodes",
            str(eval_cfg.get("num_episodes", 600)),
            "--seed",
            str(eval_cfg.get("seed", 42)),
            "--device",
            eval_cfg.get("device", "cuda"),
        ]
        cmd = [py, str(scripts_dir / "evaluate_fewshot.py")] + args
        stages.append(StageCommand(name="evaluate", command=cmd, cwd=base_dir))

    # 4. Ablation (optional)
    if config.get("stages", {}).get("ablation", False):
        ab_cfg = config.get("ablation", {})
        args = [
            "--output-dir",
            str(out_dir / "ablations"),
            "--data-dir",
            str(data_dir),
            "--hardware",
            ab_cfg.get("hardware", "rtx3060"),
            "--max-epochs",
            str(ab_cfg.get("max_epochs", 100)),
            "--seed",
            str(ab_cfg.get("seed", 42)),
        ]
        if ab_cfg.get("configs"):
            args += ["--configs"] + [str(c) for c in ab_cfg["configs"]]
        cmd = [py, str(scripts_dir / "run_ablation.py")] + args
        stages.append(StageCommand(name="ablation", command=cmd, cwd=base_dir))

    # 5. Analyze evolution
    if config.get("stages", {}).get("analyze", False):
        an_cfg = config.get("analyze", {})
        args = [
            "--checkpoint-dir",
            str(out_dir / train_cfg.get("experiment_name", "sseg_nase_exp") / "checkpoints"),
            "--output-dir",
            str(out_dir / "analysis"),
            "--format",
            an_cfg.get("format", "pdf"),
        ]
        cmd = [py, str(scripts_dir / "analyze_evolution.py")] + args
        stages.append(StageCommand(name="analyze", command=cmd, cwd=base_dir))

    # 6. Export results
    if config.get("stages", {}).get("export", False):
        ex_cfg = config.get("export", {})
        args = [
            "--results-dir",
            str(out_dir / "results"),
            "--output-dir",
            str(out_dir / "export"),
            "--format",
        ]
        formats = ex_cfg.get("format", ["latex", "markdown", "json", "csv"])
        args += formats
        if ex_cfg.get("include_baselines", False):
            args += ["--include-baselines"]
        cmd = [py, str(scripts_dir / "export_results.py")] + args
        stages.append(StageCommand(name="export", command=cmd, cwd=base_dir))

    return stages


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python scripts/cli_run.py <config.json>")
        return 2

    config_path = Path(argv[0])
    base_dir = Path.cwd()
    try:
        config = load_json(config_path)
    except Exception as exc:  # noqa: BLE001 (catch for top-level handling)
        logger.error("Failed to load config: %s", exc)
        return 1

    logger.info("Loaded config: %s", config_path)
    # Optional: validate minimal keys
    try:
        stages = build_stage_commands(config, base_dir)
    except Exception as exc:
        logger.exception("Failed to build stage commands: %s", exc)
        return 1

    # Create outputs directory if provided
    out_dir = Path(config.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Execute stages sequentially, stop on failure unless config.override_fail_continue True
    continue_on_error = config.get("continue_on_error", False)
    for stage in stages:
        code = run_subprocess(stage)
        if code != 0 and not continue_on_error:
            logger.error("Stopping pipeline due to stage failure: %s", stage.name)
            return code

    logger.info("Pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())