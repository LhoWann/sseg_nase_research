import argparse
from pathlib import Path
import sys
import json
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.evaluation_config import EvaluationConfig
from configs.evaluation_config import FewShotConfig
from configs.hardware_config import get_hardware_config
from data.benchmarks.minimagenet_dataset import MiniImageNetDataset
from data.benchmarks.episode_sampler import EpisodeSampler
from evaluation.evaluators.efficiency_evaluator import EfficiencyEvaluator
from evaluation.evaluators.fewshot_evaluator import FewShotEvaluator
from evaluation.protocols.benchmark_protocol import BenchmarkProtocol
from models.backbones.evolvable_cnn import EvolvableCNN
from utils.io.checkpoint_manager import CheckpointManager
from utils.logging.custom_logger import get_logger
from utils.logging.custom_logger import LogLevel
from utils.reproducibility.seed_everything import seed_everything
from visualization.reporters.result_formatter import ResultFormatter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on few-shot classification"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment config YAML file (must match training)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        default=None,
        help="Path to model checkpoint. If not provided, will auto-detect the latest or best checkpoint in the experiment's checkpoints directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./datasets/minimagenet"),
        help="Path to MiniImageNet dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/evaluation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-ways",
        type=int,
        default=5,
        help="Number of classes per episode",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Number of shots to evaluate",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=600,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for evaluation",
    )
    parser.add_argument(
        "--adaptation-steps",
        type=int,
        default=0,
        help="Number of test-time adaptation steps (0 to disable)",
    )
    parser.add_argument(
        "--adaptation-lr",
        type=float,
        default=0.01,
        help="Learning rate for test-time adaptation",
    )
    return parser.parse_args()
def load_model(checkpoint_path: Path, device: str, config_path: Path = None) -> EvolvableCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print("DEBUG: Starting load_model")
    print("DEBUG: checkpoint['architecture_summary'] =", checkpoint.get("architecture_summary", None))
    from configs.evolution_config import EvolutionConfig, SeedNetworkConfig
    import yaml
    arch = checkpoint.get("architecture_summary", {})
    # Fix: Architecture summary uses 'num_channels' or 'channels', not 'channel_progression'
    channel_progression = arch.get("num_channels") or arch.get("channels") or arch.get("channel_progression")
    num_blocks = arch.get("num_blocks")
    initial_channels = None
    if channel_progression and isinstance(channel_progression, list) and len(channel_progression) > 0:
        initial_channels = channel_progression[0]
        num_blocks = len(channel_progression)
    if (initial_channels is None or num_blocks is None) and config_path is not None and config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        evolution_cfg = config.get("evolution", {})
        seed_cfg = evolution_cfg.get("seed_network", {})
        if initial_channels is None:
            initial_channels = seed_cfg.get("initial_channels", 16)
        if num_blocks is None:
            num_blocks = seed_cfg.get("initial_blocks", 3)
    if initial_channels is None:
        initial_channels = 16
    if num_blocks is None:
        num_blocks = 3
    seed_config = SeedNetworkConfig(
        initial_channels=initial_channels,
        initial_blocks=num_blocks,
    )
    evolution_config = EvolutionConfig(seed_network=seed_config)
    if channel_progression and len(channel_progression) == num_blocks:
        model = EvolvableCNN(
            seed_config=seed_config,
            evolution_config=evolution_config,
            channel_progression=channel_progression
        )
    else:
        model = EvolvableCNN(
            seed_config=seed_config,
            evolution_config=evolution_config
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    def move_to_device(module, device):
        module.to(device)
        for child in module.children():
            move_to_device(child, device)
    move_to_device(model, device)
    num_params = sum(p.numel() for p in model.parameters())
    if num_params == 0:
        raise RuntimeError(
        )
    return model
def evaluate(args: argparse. Namespace) -> dict:
    logger = get_logger(
        name="evaluate_fewshot",
        level=LogLevel.INFO,
        log_file=args.output_dir / "evaluation.log",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None or not checkpoint_path.exists():
        experiment_name = args.output_dir.parent.name if args.output_dir.name == "evaluation" else args.output_dir.name
        checkpoints_dir = args.output_dir.parent / "checkpoints"
        found_checkpoint = None
        if checkpoints_dir.exists():
            from utils.io.checkpoint_manager import CheckpointManager
            manager = CheckpointManager(checkpoint_dir=checkpoints_dir, experiment_name=experiment_name)
            found_checkpoint = manager.get_best_checkpoint() or manager.get_latest_checkpoint()
        if not found_checkpoint or not found_checkpoint.exists():
            parent_dir = args.output_dir.parent
            for subdir in parent_dir.iterdir():
                if subdir.is_dir():
                    candidate = subdir / "checkpoints"
                    if candidate.exists():
                        from utils.io.checkpoint_manager import CheckpointManager
                        manager = CheckpointManager(checkpoint_dir=candidate, experiment_name=experiment_name)
                        cp = manager.get_best_checkpoint() or manager.get_latest_checkpoint()
                        if cp and cp.exists():
                            found_checkpoint = cp
                            checkpoints_dir = candidate
                            break
        if not found_checkpoint or not found_checkpoint.exists():
            logger.error(f"Could not find any valid checkpoints for experiment: {experiment_name} in {args.output_dir.parent}")
            raise FileNotFoundError(f"No checkpoints found for experiment: {experiment_name} in {args.output_dir.parent}")
        checkpoint_path = found_checkpoint
        logger.info(f"Auto-detected checkpoint: {checkpoint_path}")
    else:
        logger.info(f"Loading model from: {checkpoint_path}")
    seed_everything(args.seed, deterministic=True)
    config_path = getattr(args, "config", None)
    if config_path is None or not (config_path and config_path.exists()):
        import glob
        search_dirs = [args.output_dir, args.output_dir.parent]
        found_yaml = None
        for d in search_dirs:
            if d.exists():
                yamls = list(d.glob("*.yaml"))
                if yamls:
                    found_yaml = yamls[0]
                    break
        if found_yaml:
            config_path = found_yaml
            logger.warning(f"Config YAML not provided, auto-detected: {config_path}")
        else:
            logger.warning("Config YAML not provided and could not be auto-detected. Model architecture may not match checkpoint!")
    model = load_model(checkpoint_path, args.device, config_path)
    architecture = model.get_architecture_summary()
    logger.log_architecture(
        num_blocks=architecture["num_blocks"],
        num_params=architecture["total_params"],
        feature_dim=architecture["feature_dim"],
    )
    few_shot_config = FewShotConfig(
        num_ways=args.num_ways,
        num_shots=tuple(args.num_shots),
        num_queries_per_class=15,
        num_episodes=args.num_episodes,
    )
    import yaml
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
    else:
        full_config = {}
    eval_config = full_config.get("evaluation", {})
    adaptation_config = {}
    adaptation_config["steps"] = eval_config.get("adaptation_steps", 0)
    adaptation_config["lr"] = eval_config.get("adaptation_lr", 0.01)
    adaptation_config["use_ttt"] = eval_config.get("use_ttt", False)
    adaptation_config["ttt_steps"] = eval_config.get("ttt_steps", 5)
    adaptation_config["ttt_lr"] = eval_config.get("ttt_lr", 0.001)
    adaptation_config["classifier_type"] = eval_config.get("classifier_type", "linear")
    if args.adaptation_steps > 0:
        adaptation_config["steps"] = args.adaptation_steps
    if args.adaptation_lr != 0.01: 
        adaptation_config["lr"] = args.adaptation_lr
    evaluator = FewShotEvaluator(
        model=model,
        config=few_shot_config,
        device=args.device,
        adaptation_config=adaptation_config,
    )
    model = model.to(args.device)
    efficiency_evaluator = EfficiencyEvaluator(model=model, device=args.device)
    efficiency_metrics = efficiency_evaluator.evaluate()
    logger.info(f"Parameters: {efficiency_metrics.params_millions:.2f}M")
    logger.info(f"FLOPs: {efficiency_metrics.flops_giga:.2f}G")
    logger.info(f"Inference Time: {efficiency_metrics.inference_time_ms:.2f}ms")
    results = {
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "efficiency":  {
            "num_parameters": efficiency_metrics.num_parameters,
            "params_millions": efficiency_metrics.params_millions,
            "flops":  efficiency_metrics.flops,
            "flops_giga": efficiency_metrics.flops_giga,
            "inference_time_ms": efficiency_metrics.inference_time_ms,
        },
        "few_shot_results": [],
    }
    test_dataset = MiniImageNetDataset(
        root_dir=args.data_dir,
        split="test",
        image_size=84,
        augment=False,
    )
    for num_shots in args.num_shots:
        logger.info(f"Evaluating {args.num_ways}-way {num_shots}-shot.")
        episode_sampler = EpisodeSampler(
            dataset=test_dataset,
            num_ways=args.num_ways,
            num_shots=num_shots,
            num_queries=15,
            num_episodes=args.num_episodes,
        )
        ci, accuracies = evaluator.run_evaluation(episode_sampler)
        logger.log_evaluation_result(
            num_shots=num_shots,
            mean_accuracy=ci.mean,
            margin=ci.margin,
        )
        results["few_shot_results"].append({
            "num_ways": args.num_ways,
            "num_shots": num_shots,
            "mean_accuracy": ci.mean,
            "std_accuracy":  ci.std,
            "margin": ci.margin,
            "ci_lower": ci.lower,
            "ci_upper": ci.upper,
            "num_episodes": args.num_episodes,
        })
    results_path = args.output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")
    generate_summary_report(results, args.output_dir)
    return results
def generate_summary_report(results:  dict, output_dir: Path) -> None:
    formatter = ResultFormatter()
    lines = []
    lines.append("Few-Shot Evaluation Results")
    lines.append("")
    lines.append("Model Architecture")
    lines.append("")
    lines.append(f"Number of Blocks: {results['architecture']['num_blocks']}")
    lines.append(f"Feature Dimension: {results['architecture']['feature_dim']}")
    lines.append(f"Total Parameters: {results['architecture']['total_params']: ,}")
    lines.append("")
    lines.append("Efficiency Metrics")
    lines.append("")
    lines.append(f"Parameters:  {results['efficiency']['params_millions']:.2f}M")
    lines.append(f"FLOPs: {results['efficiency']['flops_giga']:.2f}G")
    lines.append(f"Inference Time: {results['efficiency']['inference_time_ms']:.2f}ms")
    lines.append("")
    lines.append("Few-Shot Performance")
    lines.append("")
    lines.append("N-way, K-shot, Accuracy, 95% CI")
    for fs_result in results["few_shot_results"]:
        accuracy_str = formatter.format_accuracy(
            fs_result["mean_accuracy"],
            fs_result["margin"],
        )
        ci_str = f"[{fs_result['ci_lower']:.2f}, {fs_result['ci_upper']:.2f}]"
        lines.append(f"{fs_result['num_ways']}, {fs_result['num_shots']}, {accuracy_str}, {ci_str}")
    report_path = output_dir / "evaluation_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
def main() -> None:
    args = parse_arguments()
    results = evaluate(args)
    print("\nEvaluation Results:")
    print("-" * 50)
    for fs_result in results["few_shot_results"]:
        print(
            f"{fs_result['num_ways']}-way {fs_result['num_shots']}-shot: {fs_result['mean_accuracy']:.2f}% +/- {fs_result['margin']:.2f}%"
        )
    print("-" * 50)
    print(f"Results saved to: {args.output_dir}")
if __name__ == "__main__": 
    main()
