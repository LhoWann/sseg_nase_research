import pytest
import torch
from collections import deque


class TestPlateauDetector:
    
    @pytest.fixture
    def detector(self):
        from training.callbacks.plateau_detector import PlateauDetector
        return PlateauDetector(
            window_size=5,
            plateau_threshold=0.01,
            distillation_gap_threshold=0.1,
        )
    
    def test_initial_check_not_plateau(self, detector):
        detector.update(1.0, 0.5)
        status = detector.check_plateau()
        
        assert status.is_plateau is False
    
    def test_detects_plateau_after_window(self, detector):
        for _ in range(5):
            detector.update(1.0, 0.5)
        
        status = detector.check_plateau()
        
        assert status.is_plateau is True
    
    def test_no_plateau_with_changing_loss(self, detector):
        for i in range(5):
            detector.update(1.0 - i * 0.1, 0.5)
        
        status = detector.check_plateau()
        
        assert status.is_plateau is False
    
    def test_reset_clears_history(self, detector):
        for _ in range(3):
            detector.update(1.0, 0.5)
        
        detector.reset()
        
        status = detector.check_plateau()
        assert status.current_ssl_loss == 0.0


class TestMockEvolutionCallback:
    
    def test_architecture_tracker_exists(self):
        from models.evolution.architecture_tracker import ArchitectureTracker
        from models.evolution.evolution_operators import MutationType
        
        tracker = ArchitectureTracker()
        
        tracker.record_mutation(
            epoch=1,
            level=1,
            mutation_type=MutationType.GROW,
            target_layer=None,
            num_blocks_before=3,
            num_blocks_after=4,
            num_params_before=1000,
            num_params_after=2000,
            ssl_loss_before=0.5,
        )
        
        assert len(tracker.mutation_history) == 1
        assert tracker.mutation_history[0].mutation_type == "GROW"
