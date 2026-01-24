import argparse
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.curriculum_config import CurriculumConfig
from configs.curriculum_config import CurriculumLevel
from data.curriculum.synthetic_generator import GenerationConfig
from data.curriculum.synthetic_generator import SyntheticGenerator
from data.curriculum.difficulty_scorer import DifficultyScorer
from utils.logging.custom_logger import get_logger
from utils.logging.custom_logger import LogLevel
from utils.reproducibility.seed_everything import seed_everything
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic curriculum data for SSEG pretraining"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets/synthetic_curriculum"),
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=84,
        help="Size of generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Curriculum levels to generate",
    )
    parser.add_argument(
        "--samples-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for number of samples per level",
    )
    
    return parser.parse_args()


def generate_level_data(
    level: CurriculumLevel,
    num_samples: int,
    generator:  SyntheticGenerator,
    scorer: DifficultyScorer,
    output_dir: Path,
    logger,
) -> dict:
    level_dir = output_dir / f"level_{level.value}_{level.name.lower()}"
    level_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = level_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    metadata = {
        "level": level.value,
        "level_name": level.name,
        "num_samples": num_samples,
        "samples":  [],
    }
    
    logger.info(f"Generating {num_samples} samples for {level.name}")
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    for idx in tqdm(
        range(num_samples),
        desc=f"Level {level.value}: {level.name.title()}",
        dynamic_ncols=True,
        bar_format=bar_format,
        leave=False
    ):
        torch.manual_seed(idx)
        
        if level == CurriculumLevel.BASIC:
            image = generator.generate_basic_shape()
        elif level == CurriculumLevel.TEXTURE:
            image = generator.generate_texture()
        elif level == CurriculumLevel.OBJECT:
            image = generator.generate_complex_object()
        else:
            image = generator.generate_adversarial()
        
        difficulty_components = scorer.score(image)
        difficulty_score = difficulty_components.aggregate((0.4, 0.3, 0.3))
        
        image_filename = f"sample_{idx:06d}.pt"
        image_path = images_dir / image_filename
        torch.save(image, image_path)
        
        metadata["samples"].append({
            "index": idx,
            "filename": image_filename,
            "difficulty_score": difficulty_score,
            "edge_density": difficulty_components.edge_density,
            "color_variance": difficulty_components.color_variance,
            "spatial_frequency": difficulty_components.spatial_frequency,
        })
    
    metadata["samples"].sort(key=lambda x: x["difficulty_score"])
    
    import json
    metadata_path = level_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def generate_curriculum(args: argparse. Namespace) -> None:
    logger = get_logger(
        name="generate_curriculum",
        level=LogLevel.INFO,
        log_file=args.output_dir / "generation.log",
    )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    seed_everything(args.seed, deterministic=True)
    
    curriculum_config = CurriculumConfig(image_size=args.image_size)
    
    gen_config = GenerationConfig(
        image_size=args.image_size,
        seed=args.seed,
    )
    generator = SyntheticGenerator(gen_config)
    scorer = DifficultyScorer()
    
    logger.info(f"Generating curriculum data to:  {args.output_dir}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Levels: {args.levels}")
    
    all_metadata = {}
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    for level_int in tqdm(
        args.levels,
        desc="Curriculum Levels",
        dynamic_ncols=True,
        bar_format=bar_format,
        leave=True
    ):
        level = CurriculumLevel(level_int)
        level_spec = curriculum_config.get_level_spec(level)
        
        num_samples = int(level_spec.num_samples * args.samples_multiplier)
        
        metadata = generate_level_data(
            level=level,
            num_samples=num_samples,
            generator=generator,
            scorer=scorer,
            output_dir=args.output_dir,
            logger=logger,
        )
        
        all_metadata[level.name] = metadata
        
        logger.info(f"Completed {level.name}:  {num_samples} samples")
    
    import json
    summary_path = args.output_dir / "curriculum_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    logger.info("Curriculum generation completed successfully")
    
    print("\nCurriculum Generation Summary")
    print("=" * 50)
    
    for level_name, info in all_metadata.items():
        print(f"{level_name}:  {info['num_samples']} samples")
    
    print("=" * 50)
    print(f"Data saved to: {args.output_dir}")


def main() -> None:
    args = parse_arguments()
    generate_curriculum(args)


if __name__ == "__main__":
    main()