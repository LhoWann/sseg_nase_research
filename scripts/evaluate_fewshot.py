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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on few-shot classification"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
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
    
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> EvolvableCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    from configs.evolution_config import EvolutionConfig
    from configs.evolution_config import SeedNetworkConfig
    
    architecture = checkpoint.get("architecture_summary", {})
    
    seed_config = SeedNetworkConfig(
        initial_channels=16,
        initial_blocks=architecture.get("num_blocks", 3),
    )
    evolution_config = EvolutionConfig(seed_network=seed_config)
    
    model = EvolvableCNN(
        seed_config=seed_config,
        evolution_config=evolution_config,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate(args: argparse. Namespace) -> dict:
    logger = get_logger(
        name="evaluate_fewshot",
        level=LogLevel.INFO,
        log_file=args.output_dir / "evaluation.log",
    )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    seed_everything(args.seed, deterministic=True)
    
    logger.info(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)
    
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
    
    evaluator = FewShotEvaluator(
        model=model,
        config=few_shot_config,
        device=args.device,
    )
    
    efficiency_evaluator = EfficiencyEvaluator(model=model, device=args.device)
    efficiency_metrics = efficiency_evaluator.evaluate()
    
    logger.info(f"Parameters: {efficiency_metrics.params_millions:.2f}M")
    logger.info(f"FLOPs: {efficiency_metrics.flops_giga:.2f}G")
    logger.info(f"Inference Time: {efficiency_metrics.inference_time_ms:.2f}ms")
    
    results = {
        "checkpoint": str(args.checkpoint),
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
        logger.info(f"Evaluating {args.num_ways}-way {num_shots}-shot...")
        
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
            "std":  ci.std,
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
    
    lines = [
        "# Few-Shot Evaluation Results",
        "",
        "## Model Architecture",
        "",
        f"- Number of Blocks: {results['architecture']['num_blocks']}",
        f"- Feature Dimension: {results['architecture']['feature_dim']}",
        f"- Total Parameters: {results['architecture']['total_params']: ,}",
        "",
        "## Efficiency Metrics",
        "",
        f"- Parameters:  {results['efficiency']['params_millions']:.2f}M",
        f"- FLOPs: {results['efficiency']['flops_giga']:.2f}G",
        f"- Inference Time: {results['efficiency']['inference_time_ms']:.2f}ms",
        "",
        "## Few-Shot Performance",
        "",
        "| N-way | K-shot | Accuracy | 95% CI |",
        "|-------|--------|----------|--------|",
    ]
    
    for fs_result in results["few_shot_results"]:
        accuracy_str = formatter.format_accuracy(
            fs_result["mean_accuracy"],
            fs_result["margin"],
        )
        ci_str = f"[{fs_result['ci_lower']:.2f}, {fs_result['ci_upper']:.2f}]"
        
        lines.append(
            f"| {fs_result['num_ways']} | {fs_result['num_shots']} | "
            f"{accuracy_str} | {ci_str} |"
        )
    
    report_path = output_dir / "evaluation_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_arguments()
    
    results = evaluate(args)
    
    print("\nEvaluation Results:")
    print("-" * 50)
    
    for fs_result in results["few_shot_results"]:
        print(
            f"{fs_result['num_ways']}-way {fs_result['num_shots']}-shot:  "
            f"{fs_result['mean_accuracy']:.2f}% ± {fs_result['margin']:.2f}%"
        )
    
    print("-" * 50)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__": 
    main()