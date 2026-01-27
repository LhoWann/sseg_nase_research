import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.evolution.architecture_tracker import ArchitectureTracker
from visualization.plotters.evolution_plotter import EvolutionPlotter
from visualization.plotters.evolution_plotter import PlotStyle
from visualization.plotters.loss_plotter import LossPlotter
from utils.logging.custom_logger import get_logger
from utils.logging.custom_logger import LogLevel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_arguments() -> argparse. Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze network evolution from training logs"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing training checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis results (default: checkpoint_dir/analysis)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for figures",
    )
    
    return parser.parse_args()


def load_evolution_data(checkpoint_dir: Path) -> dict:
    metadata_files = list(checkpoint_dir.glob("*_metadata.json"))
    
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {checkpoint_dir}")
    
    metadata_path = metadata_files[0]
    
    with open(metadata_path, "r") as f:
        all_metadata = json.load(f)
    
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    
    evolution_data = {
        "epochs": [],
        "num_params": [],
        "num_blocks": [],
        "feature_dims": [],
        "ssl_losses": [],
        "channel_history": [],
        "mutation_epochs": [],
        "mutation_types": [],
        "level_boundaries": [],
    }
    
    import torch
    
    for ckpt_path in checkpoints: 
        # PyTorch 2.6+ sets weights_only=True by default, which can cause UnpicklingError for custom classes.
        # Setting weights_only=False restores previous behavior. Only do this if you trust the checkpoint source.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        metadata = checkpoint.get("metadata", {})
        architecture = checkpoint.get("architecture_summary", {})
        
        evolution_data["epochs"].append(metadata.get("epoch", 0))
        evolution_data["num_params"].append(
            architecture.get("total_params", 0) / 1e6
        )
        evolution_data["num_blocks"].append(
            architecture.get("num_blocks", 0)
        )
        evolution_data["feature_dims"].append(
            architecture.get("feature_dim", 0)
        )
        evolution_data["ssl_losses"].append(
            metadata.get("ssl_loss", 0.0)
        )
        evolution_data["channel_history"].append(
            architecture.get("channel_progression", [])
        )
        
        mutation_history = checkpoint.get("mutation_history", [])
        for mutation in mutation_history:
            if mutation["epoch"] not in evolution_data["mutation_epochs"]: 
                evolution_data["mutation_epochs"].append(mutation["epoch"])
                evolution_data["mutation_types"].append(mutation["mutation_type"])
    
    return evolution_data


def analyze_mutations(evolution_data:  dict) -> dict:
    mutation_counts = {}
    
    for mutation_type in evolution_data["mutation_types"]:
        if mutation_type not in mutation_counts: 
            mutation_counts[mutation_type] = 0
        mutation_counts[mutation_type] += 1
    
    param_deltas = []
    
    if len(evolution_data["num_params"]) > 1:
        for i in range(1, len(evolution_data["num_params"])):
            delta = evolution_data["num_params"][i] - evolution_data["num_params"][i-1]
            if abs(delta) > 0.001: 
                param_deltas.append(delta)
    
    analysis = {
        "total_mutations": len(evolution_data["mutation_epochs"]),
        "mutation_counts": mutation_counts,
        "initial_params_m": evolution_data["num_params"][0] if evolution_data["num_params"] else 0,
        "final_params_m":  evolution_data["num_params"][-1] if evolution_data["num_params"] else 0,
        "initial_blocks":  evolution_data["num_blocks"][0] if evolution_data["num_blocks"] else 0,
        "final_blocks":  evolution_data["num_blocks"][-1] if evolution_data["num_blocks"] else 0,
        "param_growth_factor": (
            evolution_data["num_params"][-1] / evolution_data["num_params"][0]
            if evolution_data["num_params"] and evolution_data["num_params"][0] > 0
            else 1.0
        ),
        "avg_param_delta_per_mutation": (
            sum(param_deltas) / len(param_deltas)
            if param_deltas
            else 0.0
        ),
    }
    
    return analysis


def generate_visualizations(
    evolution_data:  dict,
    output_dir: Path,
    plot_format: str,
) -> list[Path]:
    style = PlotStyle(save_format=plot_format.strip() if isinstance(plot_format, str) else plot_format)
    evolution_plotter = EvolutionPlotter(output_dir, style)
    loss_plotter = LossPlotter(output_dir, style)
    generated_files = []
    if evolution_data["epochs"] and evolution_data["num_params"]:
        trajectory_path = evolution_plotter.plot_architecture_trajectory(
            epochs=evolution_data["epochs"],
            num_params=evolution_data["num_params"],
            num_blocks=evolution_data["num_blocks"],
            mutation_epochs=evolution_data["mutation_epochs"],
            level_boundaries=evolution_data.get("level_boundaries"),
        )
        generated_files.append(trajectory_path)
    if evolution_data["channel_history"]:
        valid_channels = [ch for ch in evolution_data["channel_history"] if ch]
        if valid_channels:
            channel_path = evolution_plotter.plot_channel_progression(
                epochs=evolution_data["epochs"][: len(valid_channels)],
                channel_history=valid_channels,
            )
            generated_files.append(channel_path)
    mutation_counts = {}
    for mt in evolution_data["mutation_types"]:
        mutation_counts[mt] = mutation_counts.get(mt, 0) + 1
    if mutation_counts:
        mutation_path = evolution_plotter.plot_mutation_distribution(
            mutation_types=list(mutation_counts.keys()),
            mutation_counts=list(mutation_counts.values()),
        )
        generated_files.append(mutation_path)
    if evolution_data["ssl_losses"]:
        loss_path = loss_plotter.plot_loss_with_mutations(
            epochs=evolution_data["epochs"],
            total_loss=evolution_data["ssl_losses"],
            mutation_epochs=evolution_data["mutation_epochs"],
        )
        generated_files.append(loss_path)
    return generated_files


def generate_report(
    evolution_data:  dict,
    analysis: dict,
    output_dir: Path,
) -> Path:
    lines = [
        "# Network Evolution Analysis Report",
        "",
        "## Evolution Summary",
        "",
        f"- Total Epochs: {len(evolution_data['epochs'])}",
        f"- Total Mutations: {analysis['total_mutations']}",
        f"- Initial Parameters: {analysis['initial_params_m']:.3f}M",
        f"- Final Parameters: {analysis['final_params_m']:.3f}M",
        f"- Parameter Growth Factor: {analysis['param_growth_factor']:.2f}x",
        f"- Initial Blocks: {analysis['initial_blocks']}",
        f"- Final Blocks: {analysis['final_blocks']}",
        "",
        "## Mutation Distribution",
        "",
        "| Type | Count |",
        "|------|-------|",
    ]
    
    for mutation_type, count in analysis["mutation_counts"].items():
        lines.append(f"| {mutation_type} | {count} |")
    
    lines.extend([
        "",
        "## Evolution Timeline",
        "",
        "| Epoch | Mutation | Params After |",
        "|-------|----------|--------------|",
    ])
    
    for i, epoch in enumerate(evolution_data["mutation_epochs"]):
        mutation_type = evolution_data["mutation_types"][i]
        
        epoch_idx = evolution_data["epochs"].index(epoch) if epoch in evolution_data["epochs"] else -1
        params_after = evolution_data["num_params"][epoch_idx] if epoch_idx >= 0 else "-"
        
        if isinstance(params_after, float):
            params_after = f"{params_after:.3f}M"
        
        lines.append(f"| {epoch} | {mutation_type} | {params_after} |")
    
    report_path = output_dir / "evolution_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    return report_path


def analyze_evolution(args: argparse. Namespace) -> None:
    logger = get_logger(
        name="analyze_evolution",
        level=LogLevel.INFO,
    )
    
    output_dir = args.output_dir or (args.checkpoint_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading evolution data from:  {args.checkpoint_dir}")
    evolution_data = load_evolution_data(args.checkpoint_dir)
    
    logger.info("Analyzing mutation patterns")
    analysis = analyze_mutations(evolution_data)
    
    logger.info("Generating visualizations")
    figure_paths = generate_visualizations(
        evolution_data=evolution_data,
        output_dir=output_dir,
        plot_format=args.format,
    )
    
    logger.info("Generating analysis report")
    report_path = generate_report(
        evolution_data=evolution_data,
        analysis=analysis,
        output_dir=output_dir,
    )
    
    print("\nEvolution Analysis Summary")
    print("=" * 50)
    print(f"Total Mutations: {analysis['total_mutations']}")
    print(f"Parameter Growth:  {analysis['initial_params_m']:.3f}M → {analysis['final_params_m']:.3f}M")
    print(f"Block Growth: {analysis['initial_blocks']} → {analysis['final_blocks']}")
    print("")
    print("Mutation Counts:")
    
    for mutation_type, count in analysis["mutation_counts"].items():
        print(f"  {mutation_type}: {count}")
    
    print("=" * 50)
    print(f"\nGenerated {len(figure_paths)} figures")
    print(f"Report saved to: {report_path}")
    print(f"Results saved to: {output_dir}")


def main() -> None:
    args = parse_arguments()
    analyze_evolution(args)


if __name__ == "__main__":
    main()