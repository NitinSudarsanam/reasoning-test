"""Tests for checkpoint manager functionality."""

import os
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from training.checkpoint_manager import CheckpointManager, CheckpointMetadata
from models.generator import LLMGenerator
from models.discriminator import LLMDiscriminator


class MockModel(nn.Module):
    """Mock model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


class MockLLM:
    """Mock LLM wrapper for testing."""
    def __init__(self, model_name="test-model", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = MockModel()


def test_checkpoint_manager_init():
    """Test checkpoint manager initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        
        assert manager.checkpoint_dir == Path(tmpdir)
        assert manager.checkpoint_dir.exists()
        assert manager.best_score == float('-inf')
        assert manager.best_checkpoint_path is None
        
    print("✓ test_checkpoint_manager_init passed")


def test_save_checkpoint():
    """Test saving a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        
        # Create mock models
        generator = MockLLM()
        discriminator = MockLLM()
        
        # Save checkpoint
        metrics = {
            'generator_reward': 0.75,
            'discriminator_reward': 0.68,
            'test_validity': 0.92
        }
        
        checkpoint_path = manager.save_checkpoint(
            generator=generator,
            discriminator=discriminator,
            stage=3,
            epoch=10,
            metrics=metrics,
            is_best=False
        )
        
        # Verify checkpoint file exists
        assert Path(checkpoint_path).exists()
        assert "stage_3_epoch_10" in checkpoint_path
        
        # Verify metadata file exists
        metadata_path = Path(checkpoint_path).with_suffix('.json')
        assert metadata_path.exists()
        
    print("✓ test_save_checkpoint passed")


def test_load_checkpoint():
    """Test loading a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        
        # Create and save checkpoint
        generator = MockLLM()
        discriminator = MockLLM()
        
        metrics = {
            'generator_reward': 0.75,
            'discriminator_reward': 0.68,
            'test_validity': 0.92
        }
        
        checkpoint_path = manager.save_checkpoint(
            generator=generator,
            discriminator=discriminator,
            stage=3,
            epoch=10,
            metrics=metrics,
            is_best=False
        )
        
        # Create new models and load checkpoint
        new_generator = MockLLM()
        new_discriminator = MockLLM()
        
        metadata = manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            generator=new_generator,
            discriminator=new_discriminator
        )
        
        # Verify metadata
        assert metadata['stage'] == 3
        assert metadata['epoch'] == 10
        assert 'metrics' in metadata
        assert metadata['metrics']['generator_reward'] == 0.75
        
    print("✓ test_load_checkpoint passed")


def test_best_checkpoint_tracking():
    """Test best checkpoint tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        
        generator = MockLLM()
        discriminator = MockLLM()
        
        # Save first checkpoint (should be best)
        metrics1 = {
            'generator_reward': 0.60,
            'discriminator_reward': 0.50,
            'test_validity': 0.80
        }
        
        is_best1 = manager.should_save_as_best(metrics1)
        assert is_best1 is True
        
        manager.save_checkpoint(
            generator=generator,
            discriminator=discriminator,
            stage=1,
            epoch=5,
            metrics=metrics1,
            is_best=is_best1
        )
        
        # Save second checkpoint with better score (should be new best)
        metrics2 = {
            'generator_reward': 0.80,
            'discriminator_reward': 0.70,
            'test_validity': 0.90
        }
        
        is_best2 = manager.should_save_as_best(metrics2)
        assert is_best2 is True
        
        manager.save_checkpoint(
            generator=generator,
            discriminator=discriminator,
            stage=2,
            epoch=10,
            metrics=metrics2,
            is_best=is_best2
        )
        
        # Save third checkpoint with worse score (should not be best)
        metrics3 = {
            'generator_reward': 0.50,
            'discriminator_reward': 0.40,
            'test_validity': 0.70
        }
        
        is_best3 = manager.should_save_as_best(metrics3)
        assert is_best3 is False
        
        # Verify best checkpoint exists
        best_checkpoint = manager.get_best_checkpoint()
        assert best_checkpoint is not None
        assert Path(best_checkpoint).exists()
        
    print("✓ test_best_checkpoint_tracking passed")


def test_get_latest_checkpoint():
    """Test getting the latest checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        
        generator = MockLLM()
        discriminator = MockLLM()
        
        # Save multiple checkpoints
        for stage in [1, 2, 3]:
            for epoch in [5, 10]:
                metrics = {
                    'generator_reward': 0.70,
                    'discriminator_reward': 0.60,
                    'test_validity': 0.85
                }
                
                manager.save_checkpoint(
                    generator=generator,
                    discriminator=discriminator,
                    stage=stage,
                    epoch=epoch,
                    metrics=metrics,
                    is_best=False
                )
        
        # Get latest checkpoint
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert "stage_3_epoch_10" in latest
        
    print("✓ test_get_latest_checkpoint passed")


def test_list_checkpoints():
    """Test listing checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        
        generator = MockLLM()
        discriminator = MockLLM()
        
        # Save checkpoints for different stages
        for stage in [1, 2, 3]:
            metrics = {
                'generator_reward': 0.70,
                'discriminator_reward': 0.60,
                'test_validity': 0.85
            }
            
            manager.save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                stage=stage,
                epoch=10,
                metrics=metrics,
                is_best=False
            )
        
        # List all checkpoints
        all_checkpoints = manager.list_checkpoints()
        assert len(all_checkpoints) >= 3
        
        # List checkpoints for specific stage
        stage_2_checkpoints = manager.list_checkpoints(stage=2)
        assert len(stage_2_checkpoints) >= 1
        assert all("stage_2" in cp for cp in stage_2_checkpoints)
        
    print("✓ test_list_checkpoints passed")


def test_checkpoint_score_computation():
    """Test checkpoint score computation."""
    manager = CheckpointManager()
    
    metrics = {
        'generator_reward': 0.80,
        'discriminator_reward': 0.70,
        'test_validity': 0.90
    }
    
    score = manager.compute_checkpoint_score(metrics)
    
    # Score should be 0.7 * 0.80 + 0.3 * 0.90 = 0.56 + 0.27 = 0.83
    expected_score = 0.7 * 0.80 + 0.3 * 0.90
    assert abs(score - expected_score) < 0.001
    
    print("✓ test_checkpoint_score_computation passed")


def test_checkpoint_corruption_handling():
    """Test handling of corrupted checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        
        # Create a corrupted checkpoint file
        corrupted_path = Path(tmpdir) / "corrupted.pt"
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid checkpoint")
        
        # Try to load corrupted checkpoint
        generator = MockLLM()
        discriminator = MockLLM()
        
        try:
            manager.load_checkpoint(
                checkpoint_path=str(corrupted_path),
                generator=generator,
                discriminator=discriminator
            )
            assert False, "Should have raised an error"
        except RuntimeError as e:
            assert "Corrupted checkpoint" in str(e)
        
    print("✓ test_checkpoint_corruption_handling passed")


def run_all_tests():
    """Run all checkpoint tests."""
    print("\n" + "="*60)
    print("RUNNING CHECKPOINT TESTS")
    print("="*60 + "\n")
    
    test_checkpoint_manager_init()
    test_save_checkpoint()
    test_load_checkpoint()
    test_best_checkpoint_tracking()
    test_get_latest_checkpoint()
    test_list_checkpoints()
    test_checkpoint_score_computation()
    test_checkpoint_corruption_handling()
    
    print("\n" + "="*60)
    print("ALL CHECKPOINT TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
