import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.reporters.latex_table_generator import LatexTableGenerator
from visualization.reporters.latex_table_generator import TableRow
from visualization.reporters.markdown_reporter import MarkdownReporter
from visualization.reporters.result_formatter import ResultFormatter
from visualization.reporters.result_formatter import ResultExporter
from experiments.comparisons.baseline_comparisons import BaselineComparison
from experiments.comparisons.baseline_comparisons import BASELINE_SPECS
from utils.logging.custom_logger import get_logger
from utils.logging.custom_logger import LogLevel


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export experiment results to publication-ready formats"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for exported files (default: results_dir/export)",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["latex", "markdown", "json", "csv"],
        choices=["latex", "markdown", "json", "csv"],
        help="Output formats to generate",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include baseline comparisons in tables",
    )
    
    return parser.parse_args()


def load_experiment_results(results_dir: Path) -> dict:
    results = {
        "evaluation": None,
        "ablation": None,
        "evolution": None,
    }
    
    eval_path = results_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            results["evaluation"] = json.load(f)
    
    ablation_path = results_dir / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path, "r") as f:
            results["ablation"] = json.load(f)
    
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            analysis_path = subdir / "analysis" / "evolution_analysis_report.md"
            if analysis_path.exists():
                results["evolution"] = str(subdir)
                break
    
    return results


def generate_main_results_table(
    results: dict,
    include_baselines: bool,
    latex_gen: LatexTableGenerator,
    md_reporter: MarkdownReporter,
    output_dir: Path,
) -> None:
    rows = []
    
    if include_baselines:
        for method, spec in BASELINE_SPECS.items():
            rows.append(TableRow(
                method=spec.name,
                backbone=spec.backbone,
                params=f"{spec.num_params / 1e6:.2f}M",
                flops=f"{spec.flops / 1e9:.2f}G",
                one_shot=f"{spec.one_shot_reported:.2f}±0.80",
                five_shot=f"{spec.five_shot_reported:.2f}±0.65",
                is_ours=False,
            ))
    
    if results["evaluation"]:
        eval_data = results["evaluation"]
        
        one_shot = "-"
        five_shot = "-"
        
        for fs in eval_data.get("few_shot_results", []):
            if fs["num_shots"] == 1:
                one_shot = f"{fs['mean_accuracy']:.2f}±{fs['margin']:.2f}"
            elif fs["num_shots"] == 5:
                five_shot = f"{fs['mean_accuracy']:.2f}±{fs['margin']:.2f}"
        
        efficiency = eval_data.get("efficiency", {})
        
        rows.append(TableRow(
            method="SSEG-NASE",
            backbone="Evolved-CNN",
            params=f"{efficiency.get('params_millions', 0):.2f}M",
            flops=f"{efficiency.get('flops_giga', 0):.2f}G",
            one_shot=one_shot,
            five_shot=five_shot,
            is_ours=True,
        ))
    
    if rows:
        latex_table = latex_gen.generate_main_results_table(
            rows=rows,
            caption="Comparison with state-of-the-art methods on MiniImageNet",
            label="tab:main_results",
        )
        latex_gen.save_table(latex_table, "main_results.tex")
        
        methods = [r.method for r in rows]
        backbones = [r.backbone for r in rows]
        params = [r.params for r in rows]
        flops = [r.flops for r in rows]
        one_shots = [r.one_shot for r in rows]
        five_shots = [r.five_shot for r in rows]
        
        highlight_idx = None
        for i, r in enumerate(rows):
            if r.is_ours:
                highlight_idx = i
                break
        
        md_table = md_reporter.generate_comparison_table(
            methods=methods,
            backbones=backbones,
            params=params,
            flops=flops,
            one_shot=one_shots,
            five_shot=five_shots,
            highlight_idx=highlight_idx,
        )
        md_reporter.save_report(md_table, "main_results.md")


def generate_ablation_table(
    results: dict,
    latex_gen: LatexTableGenerator,
    md_reporter: MarkdownReporter,
    output_dir: Path,
) -> None:
    if not results["ablation"]:
        return
    
    ablation_data = results["ablation"]
    
    configs = []
    sseg = []
    nase = []
    curriculum = []
    distillation = []
    one_shot = []
    five_shot = []
    
    component_mapping = {
        "SEED_ONLY": (False, False, False, False),
        "SSL_ONLY": (False, False, False, False),
        "SSEG_ONLY": (True, False, False, False),
        "SSEG_CURRICULUM": (True, False, True, False),
        "SSEG_DISTILLATION": (True, False, True, True),
        "SSEG_NASE": (True, True, False, True),
        "FULL_PIPELINE": (True, True, True, True),
    }
    
    for ablation in ablation_data:
        configs.append(ablation["config_name"])
        
        components = component_mapping.get(
            ablation["ablation_type"],
            (False, False, False, False)
        )
        sseg.append(components[0])
        nase.append(components[1])
        curriculum.append(components[2])
        distillation.append(components[3])
        
        one_shot_acc = "-"
        five_shot_acc = "-"
        
        for fs in ablation.get("few_shot_results", []):
            if fs["num_shots"] == 1:
                one_shot_acc = f"{fs['mean_accuracy']:.2f}±{fs['margin']:.2f}"
            elif fs["num_shots"] == 5:
                five_shot_acc = f"{fs['mean_accuracy']:.2f}±{fs['margin']:.2f}"
        
        one_shot.append(one_shot_acc)
        five_shot.append(five_shot_acc)
    
    latex_table = latex_gen.generate_ablation_table(
        configs=configs,
        components={
            "SSEG": sseg,
            "NASE": nase,
            "Curriculum": curriculum,
            "Distill": distillation,
        },
        one_shot_results=one_shot,
        five_shot_results=five_shot,
        caption="Ablation study on MiniImageNet",
        label="tab:ablation",
    )
    latex_gen.save_table(latex_table, "ablation_results.tex")
    
    md_table = md_reporter.generate_ablation_table(
        configs=configs,
        sseg=sseg,
        nase=nase,
        curriculum=curriculum,
        distillation=distillation,
        one_shot=one_shot,
        five_shot=five_shot,
    )
    md_reporter.save_report(md_table, "ablation_results.md")


def export_to_json(results: dict, output_dir: Path) -> None:
    exporter = ResultExporter(output_dir)
    
    if results["evaluation"]:
        exporter.export_to_json(
            results["evaluation"],
            "evaluation_results.json",
        )
    
    if results["ablation"]:
        exporter.export_to_json(
            results["ablation"],
            "ablation_results.json",
        )


def export_to_csv(results: dict, output_dir: Path) -> None:
    exporter = ResultExporter(output_dir)
    
    if results["evaluation"]:
        eval_data = results["evaluation"]
        
        headers = ["num_ways", "num_shots", "mean_accuracy", "std", "margin"]
        rows = []
        
        for fs in eval_data.get("few_shot_results", []):
            rows.append([
                fs["num_ways"],
                fs["num_shots"],
                f"{fs['mean_accuracy']:.4f}",
                f"{fs['std']:.4f}",
                f"{fs['margin']:.4f}",
            ])
        
        exporter.export_to_csv(headers, rows, "evaluation_results.csv")
    
    if results["ablation"]:
        headers = [
            "config_name",
            "ablation_type",
            "1shot_accuracy",
            "5shot_accuracy",
            "params_millions",
            "flops_giga",
        ]
        rows = []
        
        for ablation in results["ablation"]:
            one_shot = "-"
            five_shot = "-"
            
            for fs in ablation.get("few_shot_results", []):
                if fs["num_shots"] == 1:
                    one_shot = f"{fs['mean_accuracy']:.4f}"
                elif fs["num_shots"] == 5:
                    five_shot = f"{fs['mean_accuracy']:.4f}"
            
            efficiency = ablation.get("efficiency", {})
            
            rows.append([
                ablation["config_name"],
                ablation["ablation_type"],
                one_shot,
                five_shot,
                f"{efficiency.get('params_millions', 0):.4f}",
                f"{efficiency.get('flops_giga', 0):.4f}",
            ])
        
        exporter.export_to_csv(headers, rows, "ablation_results.csv")


def export_results(args: argparse.Namespace) -> None:
    logger = get_logger(
        name="export_results",
        level=LogLevel.INFO,
    )
    
    output_dir = args.output_dir or (args.results_dir / "export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading results from: {args.results_dir}")
    results = load_experiment_results(args.results_dir)
    
    latex_gen = LatexTableGenerator(output_dir)
    md_reporter = MarkdownReporter(output_dir)
    
    exported_files = []
    
    if "latex" in args.format or "markdown" in args.format:
        logger.info("Generating main results table")
        generate_main_results_table(
            results=results,
            include_baselines=args.include_baselines,
            latex_gen=latex_gen,
            md_reporter=md_reporter,
            output_dir=output_dir,
        )
        
        if "latex" in args.format:
            exported_files.append("main_results.tex")
        if "markdown" in args.format:
            exported_files.append("main_results.md")
        
        if results["ablation"]:
            logger.info("Generating ablation table")
            generate_ablation_table(
                results=results,
                latex_gen=latex_gen,
                md_reporter=md_reporter,
                output_dir=output_dir,
            )
            
            if "latex" in args.format:
                exported_files.append("ablation_results.tex")
            if "markdown" in args.format:
                exported_files.append("ablation_results.md")
    
    if "json" in args.format:
        logger.info("Exporting to JSON")
        export_to_json(results, output_dir)
        exported_files.extend(["evaluation_results.json", "ablation_results.json"])
    
    if "csv" in args.format:
        logger.info("Exporting to CSV")
        export_to_csv(results, output_dir)
        exported_files.extend(["evaluation_results.csv", "ablation_results.csv"])
    
    logger.info(f"Export complete. Files saved to: {output_dir}")
    for f in exported_files:
        logger.info(f"  - {f}")


def main() -> None:
    args = parse_arguments()
    export_results(args)


if __name__ == "__main__":
    main()
