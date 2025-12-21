import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.ablation_studies.ablation_runner import AblationRunner
from experiments.ablation_studies.ablation_configs import AblationType
from configs.hardware_config import get_hardware_config
from utils.logging.custom_logger import get_logger
from utils.logging.custom_logger import LogLevel
from utils.reproducibility.seed_everything import seed_everything


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation studies for SSEG-NASE"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/ablation"),
        help="Output directory for ablation results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./datasets"),
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="rtx3060",
        choices=["rtx3060", "rtx3090", "a100"],
        help="Hardware profile for optimization",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum training epochs per ablation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=None,
        choices=[t.name for t in AblationType],
        help="Specific ablations to run (default: all)",
    )
    
    return parser.parse_args()


def run_ablations(args: argparse.Namespace) -> None:
    logger = get_logger(
        name="run_ablation",
        level=LogLevel.INFO,
        log_file=args.output_dir / "ablation.log",
    )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    seed_everything(args.seed, deterministic=True)
    
    hardware_config = get_hardware_config(args.hardware)
    
    runner = AblationRunner(
        output_dir=args.output_dir,
        data_root=args.data_dir,
        hardware_config=hardware_config,
        seed=args.seed,
    )
    
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Hardware: {args.hardware}")
    logger.info(f"Max epochs: {args.max_epochs}")
    
    if args.ablations:
        ablation_types = [AblationType[name] for name in args.ablations]
        logger.info(f"Running selected ablations: {args.ablations}")
        results = runner.run_selected_ablations(ablation_types, args.max_epochs)
    else:
        logger.info("Running all ablations")
        results = runner.run_all_ablations(args.max_epochs)
    
    logger.info(f"Completed {len(results)} ablation studies")
    
    for result in results:
        logger.info(
            f"  {result.config_name}: "
            f"blocks={result.final_num_blocks}, "
            f"params={result.final_num_params:,}, "
            f"mutations={result.num_mutations}"
        )


def main() -> None:
    args = parse_arguments()
    run_ablations(args)


if __name__ == "__main__":
    main()
