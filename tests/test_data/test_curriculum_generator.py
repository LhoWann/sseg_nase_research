import pytest
import torch
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
class CurriculumLevel(IntEnum):
    BASIC = 1
    TEXTURE = 2
    OBJECT = 3
    ADVERSARIAL = 4
@dataclass(frozen=True)
class LevelSpec:
    name: str
    num_samples: int
    complexity_min: float
    complexity_max: float
@dataclass
class CurriculumConfig:
    image_size: int = 84
    difficulty_weights: tuple = (0.4, 0.3, 0.3)
    def __post_init__(self):
        self._level_specs = {
            CurriculumLevel.BASIC: LevelSpec(
                name="basic",
                num_samples=100,
                complexity_min=0.0,
                complexity_max=0.25,
            ),
            CurriculumLevel.TEXTURE: LevelSpec(
                name="texture",
                num_samples=200,
                complexity_min=0.25,
                complexity_max=0.50,
            ),
            CurriculumLevel.OBJECT:  LevelSpec(
                name="object",
                num_samples=300,
                complexity_min=0.50,
                complexity_max=0.75,
            ),
            CurriculumLevel.ADVERSARIAL: LevelSpec(
                name="adversarial",
                num_samples=100,
                complexity_min=0.75,
                complexity_max=1.0,
            ),
        }
    def get_level_spec(self, level:  CurriculumLevel) -> LevelSpec:
        return self._level_specs[level]
@dataclass(frozen=True)
class GenerationConfig:
    image_size: int
    seed: Optional[int] = None
class SyntheticGenerator:
    def __init__(self, config: GenerationConfig):
        self._config = config
        self._generator = torch.Generator()
        if config.seed is not None:
            self._generator.manual_seed(config.seed)
    def generate_basic_shape(self) -> torch.Tensor:
        image = torch.zeros(3, self._config.image_size, self._config.image_size)
        color = torch.rand(3, generator=self._generator)
        center_y = self._config.image_size // 2
        center_x = self._config.image_size // 2
        radius = torch.randint(
            self._config.image_size // 8,
            self._config.image_size // 3,
            (1,),
            generator=self._generator,
        ).item()
        y_coords = torch.arange(self._config.image_size).view(-1, 1)
        x_coords = torch.arange(self._config.image_size).view(1, -1)
        dist_sq = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
        mask = dist_sq <= radius ** 2
        for c in range(3):
            image[c][mask] = color[c]
        return image
    def generate_texture(self) -> torch.Tensor:
        image = torch.zeros(3, self._config.image_size, self._config.image_size)
        base_freq = torch.rand(1, generator=self._generator).item() * 10 + 5
        y_coords = torch.arange(
            self._config.image_size, dtype=torch.float32
        ).view(-1, 1)
        x_coords = torch.arange(
            self._config.image_size, dtype=torch.float32
        ).view(1, -1)
        for c in range(3):
            phase = torch.rand(1, generator=self._generator).item() * 6.28
            normalized_coords = (y_coords + x_coords) / self._config.image_size
            pattern = torch.sin(normalized_coords * base_freq * 6.28 + phase)
            noise = torch.rand(
                self._config.image_size,
                self._config.image_size,
                generator=self._generator,
            ) * 0.2
            image[c] = (pattern + 1) / 2 + noise
        return torch.clamp(image, 0, 1)
    def generate_complex_object(self) -> torch.Tensor:
        background_color = torch.rand(3, generator=self._generator) * 0.3
        image = background_color.view(3, 1, 1).expand(
            3, self._config.image_size, self._config.image_size
        ).clone()
        num_shapes = torch.randint(2, 5, (1,), generator=self._generator).item()
        for _ in range(num_shapes):
            shape = self.generate_basic_shape()
            offset_y = torch.randint(
                -20, 21, (1,), generator=self._generator
            ).item()
            offset_x = torch.randint(
                -20, 21, (1,), generator=self._generator
            ).item()
            if offset_y != 0 or offset_x != 0:
                shape = torch.roll(shape, shifts=(offset_y, offset_x), dims=(1, 2))
            mask = shape.sum(dim=0) > 0
            image[: , mask] = shape[: , mask]
        return torch.clamp(image, 0, 1)
    def generate_adversarial(self) -> torch.Tensor:
        base_image = self.generate_complex_object()
        perturbation = torch.randn(
            3,
            self._config.image_size,
            self._config.image_size,
            generator=self._generator,
        ) * 0.1
        return torch.clamp(base_image + perturbation, 0, 1)
@dataclass(frozen=True)
class DifficultyComponents:
    edge_density: float
    color_variance: float
    spatial_frequency: float
    def aggregate(self, weights: tuple) -> float:
        return (
            weights[0] * self.edge_density
            + weights[1] * self.color_variance
            + weights[2] * self.spatial_frequency
        )
class DifficultyScorer:
    def __init__(self, device: str = "cpu"):
        self._device = device
    def compute_edge_density(self, image: torch. Tensor) -> float:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        grayscale = image.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        edges_x = torch.nn.functional.conv2d(grayscale, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(grayscale, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return edge_magnitude.mean().item()
    def compute_color_variance(self, image: torch. Tensor) -> float:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        per_channel_var = image.var(dim=(2, 3))
        return per_channel_var.mean().item()
    def compute_spatial_frequency(self, image: torch. Tensor) -> float:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        grayscale = image.mean(dim=1)
        fft_result = torch.fft.fft2(grayscale)
        magnitude = torch.abs(fft_result)
        return magnitude.mean().item()
    def score(self, image: torch. Tensor) -> DifficultyComponents:
        edge = self.compute_edge_density(image)
        color = self.compute_color_variance(image)
        freq = self.compute_spatial_frequency(image)
        max_edge = 10.0
        max_color = 1.0
        max_freq = 50.0
        return DifficultyComponents(
            edge_density=min(edge / max_edge, 1.0),
            color_variance=min(color / max_color, 1.0),
            spatial_frequency=min(freq / max_freq, 1.0),
        )
class CurriculumDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: CurriculumConfig,
        level:  CurriculumLevel,
        seed: Optional[int] = None,
    ):
        self._config = config
        self._level = level
        self._seed = seed
        level_spec = config.get_level_spec(level)
        self._num_samples = level_spec.num_samples
        gen_config = GenerationConfig(
            image_size=config.image_size,
            seed=seed,
        )
        self._generator = SyntheticGenerator(gen_config)
        self._scorer = DifficultyScorer()
        self._cache:  dict = {}
    def __len__(self) -> int:
        return self._num_samples
    def __getitem__(self, index: int) -> torch.Tensor:
        if index >= self._num_samples:
            raise IndexError(
            )
        if index in self._cache:
            return self._cache[index]
        torch.manual_seed(index + (self._seed or 0))
        if self._level == CurriculumLevel.BASIC: 
            image = self._generator.generate_basic_shape()
        elif self._level == CurriculumLevel.TEXTURE: 
            image = self._generator.generate_texture()
        elif self._level == CurriculumLevel. OBJECT:
            image = self._generator.generate_complex_object()
        else: 
            image = self._generator.generate_adversarial()
        if len(self._cache) < 100: 
            self._cache[index] = image
        return image
    def get_difficulty_score(self, index: int) -> float:
        image = self[index]
        components = self._scorer.score(image)
        return components.aggregate(self._config.difficulty_weights)
class CurriculumScheduler:
    def __init__(self, config: CurriculumConfig, seed: Optional[int] = None):
        self._config = config
        self._seed = seed
        self._current_level = CurriculumLevel.BASIC
        self._datasets: dict = {}
    def get_current_dataset(self) -> CurriculumDataset:
        if self._current_level not in self._datasets:
            self._datasets[self._current_level] = CurriculumDataset(
                config=self._config,
                level=self._current_level,
                seed=self._seed,
            )
        return self._datasets[self._current_level]
    def advance_level(self) -> bool:
        if self._current_level < CurriculumLevel.ADVERSARIAL:
            self._current_level = CurriculumLevel(self._current_level + 1)
            return True
        return False
    def reset(self) -> None:
        self._current_level = CurriculumLevel. BASIC
        self._datasets.clear()
    @property
    def current_level(self) -> CurriculumLevel:
        return self._current_level
    @property
    def is_final_level(self) -> bool:
        return self._current_level == CurriculumLevel.ADVERSARIAL
    @property
    def progress(self) -> float:
        return self._current_level / len(CurriculumLevel)
class TestSyntheticGenerator: 
    @pytest.fixture
    def generator(self) -> SyntheticGenerator:
        config = GenerationConfig(image_size=84, seed=42)
        return SyntheticGenerator(config)
    def test_generate_basic_shape_returns_correct_shape(
        self, generator:  SyntheticGenerator
    ) -> None:
        image = generator.generate_basic_shape()
        assert image.shape == (3, 84, 84)
        assert image.dtype == torch.float32
    def test_generate_basic_shape_values_in_valid_range(
        self, generator: SyntheticGenerator
    ) -> None:
        image = generator.generate_basic_shape()
        assert image.min() >= 0.0
        assert image.max() <= 1.0
    def test_generate_texture_returns_correct_shape(
        self, generator: SyntheticGenerator
    ) -> None:
        image = generator.generate_texture()
        assert image.shape == (3, 84, 84)
        assert image.dtype == torch.float32
    def test_generate_texture_values_in_valid_range(
        self, generator: SyntheticGenerator
    ) -> None:
        image = generator.generate_texture()
        assert image.min() >= 0.0
        assert image.max() <= 1.0
    def test_generate_complex_object_returns_correct_shape(
        self, generator: SyntheticGenerator
    ) -> None:
        image = generator.generate_complex_object()
        assert image.shape == (3, 84, 84)
        assert image.dtype == torch.float32
    def test_generate_adversarial_returns_correct_shape(
        self, generator:  SyntheticGenerator
    ) -> None:
        image = generator.generate_adversarial()
        assert image.shape == (3, 84, 84)
        assert image.dtype == torch.float32
    def test_generator_reproducibility_with_same_seed(self) -> None:
        config1 = GenerationConfig(image_size=84, seed=123)
        config2 = GenerationConfig(image_size=84, seed=123)
        gen1 = SyntheticGenerator(config1)
        gen2 = SyntheticGenerator(config2)
        image1 = gen1.generate_basic_shape()
        image2 = gen2.generate_basic_shape()
        assert torch.allclose(image1, image2)
    def test_generator_different_results_with_different_seeds(self) -> None:
        config1 = GenerationConfig(image_size=84, seed=123)
        config2 = GenerationConfig(image_size=84, seed=456)
        gen1 = SyntheticGenerator(config1)
        gen2 = SyntheticGenerator(config2)
        image1 = gen1.generate_basic_shape()
        image2 = gen2.generate_basic_shape()
        assert not torch.allclose(image1, image2)
class TestDifficultyScorer:
    @pytest.fixture
    def scorer(self) -> DifficultyScorer:
        return DifficultyScorer(device="cpu")
    def test_score_returns_difficulty_components(
        self, scorer: DifficultyScorer
    ) -> None:
        image = torch.rand(3, 84, 84)
        components = scorer.score(image)
        assert isinstance(components, DifficultyComponents)
        assert 0.0 <= components.edge_density <= 1.0
        assert 0.0 <= components.color_variance <= 1.0
        assert 0.0 <= components.spatial_frequency <= 1.0
    def test_compute_edge_density_returns_float(
        self, scorer: DifficultyScorer
    ) -> None:
        image = torch.rand(3, 84, 84)
        edge_density = scorer.compute_edge_density(image)
        assert isinstance(edge_density, float)
        assert edge_density >= 0.0
    def test_compute_color_variance_returns_float(
        self, scorer:  DifficultyScorer
    ) -> None:
        image = torch.rand(3, 84, 84)
        color_variance = scorer.compute_color_variance(image)
        assert isinstance(color_variance, float)
        assert color_variance >= 0.0
    def test_compute_spatial_frequency_returns_float(
        self, scorer: DifficultyScorer
    ) -> None:
        image = torch.rand(3, 84, 84)
        spatial_frequency = scorer.compute_spatial_frequency(image)
        assert isinstance(spatial_frequency, float)
        assert spatial_frequency >= 0.0
    def test_aggregate_with_valid_weights(self) -> None:
        components = DifficultyComponents(
            edge_density=0.5,
            color_variance=0.3,
            spatial_frequency=0.7,
        )
        weights = (0.4, 0.3, 0.3)
        aggregated = components.aggregate(weights)
        expected = 0.4 * 0.5 + 0.3 * 0.3 + 0.3 * 0.7
        assert abs(aggregated - expected) < 1e-6
class TestCurriculumDataset:
    @pytest.fixture
    def config(self) -> CurriculumConfig: 
        return CurriculumConfig(image_size=84)
    def test_dataset_length_matches_level_spec(
        self, config: CurriculumConfig
    ) -> None:
        dataset = CurriculumDataset(
            config=config,
            level=CurriculumLevel. BASIC,
            seed=42,
        )
        expected_length = config.get_level_spec(CurriculumLevel.BASIC).num_samples
        assert len(dataset) == expected_length
    def test_getitem_returns_tensor(self, config:  CurriculumConfig) -> None:
        dataset = CurriculumDataset(
            config=config,
            level=CurriculumLevel.BASIC,
            seed=42,
        )
        image = dataset[0]
        assert isinstance(image, torch. Tensor)
        assert image.shape == (3, 84, 84)
    def test_getitem_out_of_range_raises_error(
        self, config:  CurriculumConfig
    ) -> None:
        dataset = CurriculumDataset(
            config=config,
            level=CurriculumLevel.BASIC,
            seed=42,
        )
        with pytest.raises(IndexError):
            _ = dataset[len(dataset) + 100]
    def test_get_difficulty_score_returns_float(
        self, config:  CurriculumConfig
    ) -> None:
        dataset = CurriculumDataset(
            config=config,
            level=CurriculumLevel.BASIC,
            seed=42,
        )
        score = dataset.get_difficulty_score(0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
class TestCurriculumScheduler: 
    @pytest.fixture
    def scheduler(self) -> CurriculumScheduler:
        config = CurriculumConfig(image_size=84)
        return CurriculumScheduler(config, seed=42)
    def test_initial_level_is_basic(
        self, scheduler:  CurriculumScheduler
    ) -> None:
        assert scheduler.current_level == CurriculumLevel.BASIC
    def test_advance_level_increments_level(
        self, scheduler: CurriculumScheduler
    ) -> None:
        initial_level = scheduler.current_level
        advanced = scheduler.advance_level()
        assert advanced is True
        assert scheduler.current_level == CurriculumLevel(initial_level + 1)
    def test_advance_level_returns_false_at_final_level(
        self, scheduler: CurriculumScheduler
    ) -> None:
        while scheduler.current_level < CurriculumLevel.ADVERSARIAL:
            scheduler.advance_level()
        advanced = scheduler.advance_level()
        assert advanced is False
        assert scheduler.current_level == CurriculumLevel. ADVERSARIAL
    def test_is_final_level_true_at_adversarial(
        self, scheduler: CurriculumScheduler
    ) -> None:
        while not scheduler.is_final_level:
            scheduler.advance_level()
        assert scheduler.is_final_level is True
        assert scheduler.current_level == CurriculumLevel.ADVERSARIAL
    def test_get_current_dataset_returns_dataset(
        self, scheduler: CurriculumScheduler
    ) -> None:
        dataset = scheduler.get_current_dataset()
        assert isinstance(dataset, CurriculumDataset)
    def test_reset_returns_to_basic_level(
        self, scheduler: CurriculumScheduler
    ) -> None:
        scheduler.advance_level()
        scheduler.advance_level()
        scheduler.reset()
        assert scheduler.current_level == CurriculumLevel. BASIC
    def test_progress_increases_with_level(
        self, scheduler: CurriculumScheduler
    ) -> None:
        initial_progress = scheduler.progress
        scheduler.advance_level()
        assert scheduler.progress > initial_progress
