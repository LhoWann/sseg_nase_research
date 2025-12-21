import pytest
import torch
import numpy as np


class TestConfidenceInterval:
    
    def test_from_samples_returns_correct_mean(self):
        from evaluation.metrics.confidence_interval import ConfidenceInterval
        
        samples = np.array([70.0, 72.0, 74.0, 76.0, 78.0])
        ci = ConfidenceInterval.from_samples(samples)
        
        assert abs(ci.mean - 74.0) < 1e-6
    
    def test_from_samples_returns_valid_interval(self):
        from evaluation.metrics.confidence_interval import ConfidenceInterval
        
        samples = np.array([70.0, 72.0, 74.0, 76.0, 78.0])
        ci = ConfidenceInterval.from_samples(samples)
        
        assert ci.lower < ci.mean < ci.upper
    
    def test_margin_is_positive(self):
        from evaluation.metrics.confidence_interval import ConfidenceInterval
        
        samples = np.array([70.0, 72.0, 74.0, 76.0, 78.0])
        ci = ConfidenceInterval.from_samples(samples)
        
        assert ci.margin > 0


class TestAccuracyMetrics:
    
    def test_compute_accuracy_perfect(self):
        from evaluation.metrics.accuracy_metrics import compute_accuracy
        
        predictions = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        
        accuracy = compute_accuracy(predictions, targets)
        
        assert accuracy == 100.0
    
    def test_compute_accuracy_zero(self):
        from evaluation.metrics.accuracy_metrics import compute_accuracy
        
        predictions = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([4, 3, 2, 1, 0])
        
        accuracy = compute_accuracy(predictions, targets)
        
        # Only middle element (2) is correct
        assert accuracy == 20.0
    
    def test_compute_accuracy_partial(self):
        from evaluation.metrics.accuracy_metrics import compute_accuracy
        
        predictions = torch.tensor([0, 1, 0, 1, 0])
        targets = torch.tensor([0, 1, 1, 0, 0])
        
        accuracy = compute_accuracy(predictions, targets)
        
        # 3 out of 5 correct
        assert accuracy == 60.0


class TestComplexityMetrics:
    
    def test_count_parameters(self):
        from evaluation.metrics.complexity_metrics import count_parameters
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        total, trainable = count_parameters(model)
        
        # 10*5 weights + 5 bias = 55
        assert total == 55
        assert trainable == 55
