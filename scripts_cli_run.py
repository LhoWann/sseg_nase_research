
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
DEFAULT_TIMEOUT: int = 60 * 60 * 24  
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
            stdout=None,  
            stderr=None,  
            text=True,
            timeout=cmd.timeout or DEFAULT_TIMEOUT,
        )
        logger.info("Exit code: %d", result.returncode)
        if result.returncode != 0:
            logger.error("Stage '%s' failed (exit %d).", cmd.name, result.returncode)
        return result.returncode
    except subprocess.TimeoutExpired as e:
        logger.error("Stage '%s' timed out after %s seconds", cmd.name, e.timeout)
        return 124  
def ensure_executable_python() -> str:
    return sys.executable

def build_stage_commands(config: Dict[str, Any], base_dir: Path) -> List[StageCommand]:
    py = ensure_executable_python()
    scripts_dir = base_dir / "scripts"
    
    train_exp = config.get("train", {}).get("experiment_name", "sseg_nase_exp")
    
    # Logic: config > default local path
    default_out_dir = base_dir / "outputs" / train_exp
    default_data_dir = base_dir / "datasets"
    
    out_dir = Path(config.get("output_dir", default_out_dir.parent)) # Parent because train_sseg appends exp_name
    data_dir = Path(config.get("data_dir", default_data_dir))
    
    stages: List[StageCommand] = []
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
    train_cfg = config.get("train", {})
    if config.get("stages", {}).get("train", True):
        args = [
            "--experiment-name",
            train_cfg.get("experiment_name", "sseg_nase_exp"),
            "--output-dir",
            str(out_dir),
            "--data-dir",
            str(data_dir),
            "--hardware",
            config.get("hardware_profile", train_cfg.get("hardware", "default_gpu")),
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
            str(out_dir / "results"),
            "--num-ways",
            str(eval_cfg.get("num_ways", 5)),
            "--num-shots",
        ]
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
        if train_cfg.get("config"):
            args += ["--config", str(train_cfg["config"])]
        cmd = [py, str(scripts_dir / "evaluate_fewshot.py")] + args
        stages.append(StageCommand(name="evaluate", command=cmd, cwd=base_dir))
    if config.get("stages", {}).get("ablation", False):
        ab_cfg = config.get("ablation", {})
        args = [
            "--output-dir",
            str(out_dir / "ablations"),
            "--data-dir",
            str(data_dir),
            "--hardware",
            config.get("hardware_profile", ab_cfg.get("hardware", "default_gpu")),
            "--max-epochs",
            str(ab_cfg.get("max_epochs", 100)),
            "--seed",
            str(ab_cfg.get("seed", 42)),
        ]
        if ab_cfg.get("ablations"):
            args += ["--ablations"] + [str(a) for a in ab_cfg["ablations"]]
        cmd = [py, str(scripts_dir / "run_ablation.py")] + args
        stages.append(StageCommand(name="ablation", command=cmd, cwd=base_dir))
    if config.get("stages", {}).get("analyze", False):
        an_cfg = config.get("analyze", {})
        args = [
            "--checkpoints-dir",
            str(out_dir / train_cfg.get("experiment_name", "sseg_nase_exp") / "checkpoints"),
            "--output-dir",
            str(out_dir / "analysis"),
            "--format",
            an_cfg.get("format", "pdf"),
        ]
        cmd = [py, str(scripts_dir / "analyze_evolution.py")] + args
        stages.append(StageCommand(name="analyze", command=cmd, cwd=base_dir))
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
    except Exception as exc:  
        logger.error("Failed to load config: %s", exc)
        return 1
    logger.info("Loaded config: %s", config_path)
    try:
        stages = build_stage_commands(config, base_dir)
    except Exception as exc:
        logger.exception("Failed to build stage commands: %s", exc)
        return 1
    out_dir_path = Path(config.get("output_dir", "outputs"))
    
    train_exp = config.get("train", {}).get("experiment_name", "sseg_nase_exp")

    default_out_parent = base_dir / "outputs"
    
    out_dir = Path(config.get("output_dir", default_out_parent))
    
    out_dir.mkdir(parents=True, exist_ok=True)
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