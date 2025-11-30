"""Tests for inference engine functionality."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from inference.inference_engine import InferenceEngine, InferenceResult
from sandbox.sandbox import Sandbox, ExecutionResult
from data.problem_dataset import Problem


class MockModel(nn.Module):
    """Mock model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def eval(self):
        """Set to eval mode."""
        return self


class MockGenerator:
    """Mock generator for testing."""
    def __init__(self, model_name="test-gen", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = MockModel()
    
    def generate_stage_output(self, problem, previous_stages, stage_id, **kwargs):
        """Generate mock stage output."""
        return f"Stage {stage_id} reasoning for: {problem[:30]}..."
    
    def generate_code(self, problem, reasoning_chain, **kwargs):
        """Generate mock code."""
        return f"def solution():\n    # Generated for: {problem[:30]}...\n    return 42"


class MockDiscriminator:
    """Mock discriminator for testing."""
    def __init__(self, model_name="test-disc", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = MockModel()
    
    def generate_tests(self, problem, generator_code, **kwargs):
        """Generate mock tests."""
        return "assert solution() == 42"


class MockSandbox:
    """Mock sandbox for testing."""
    def __init__(self, timeout=5):
        self.timeout = timeout
    
    def execute_tests(self, code, tests):
        """Mock test execution."""
        return ExecutionResult(
            passed=True,
            num_passed=1,
            num_total=1,
            errors=[],
            stdout="",
            stderr="",
            timed_out=False
        )


def test_inference_engine_init():
    """Test inference engine initialization."""
    generator = MockGenerator()
    discriminator = MockDiscriminator()
    sandbox = MockSandbox()
    
    engine = InferenceEngine(
        generator=generator,
        discriminator=discriminator,
        sandbox=sandbox,
        device="cpu"
    )
    
    assert engine.generator is not None
    assert engine.discriminator is not None
    assert engine.sandbox is not None
    assert engine.device == "cpu"
    
    print("✓ test_inference_engine_init passed")


def test_solve_problem():
    """Test solving a single problem."""
    generator = MockGenerator()
    discriminator = MockDiscriminator()
    sandbox = MockSandbox()
    
    engine = InferenceEngine(
        generator=generator,
        discriminator=discriminator,
        sandbox=sandbox
    )
    
    result = engine.solve_problem(
        problem_description="Write a function that returns 42",
        function_signature="def solution():",
        execute_tests=False
    )
    
    assert isinstance(result, InferenceResult)
    assert result.problem_description == "Write a function that returns 42"
    assert len(result.reasoning_chain) == 4  # Stages 1-4
    assert result.generated_code is not None
    assert "def solution()" in result.generated_code
    assert result.inference_time > 0
    assert len(result.stage_times) == 5
    
    print("✓ test_solve_problem passed")


def test_solve_problem_with_tests():
    """Test solving a problem with test execution."""
    generator = MockGenerator()
    discriminator = MockDiscriminator()
    sandbox = MockSandbox()
    
    engine = InferenceEngine(
        generator=generator,
        discriminator=discriminator,
        sandbox=sandbox
    )
    
    result = engine.solve_problem(
        problem_description="Write a function that returns 42",
        execute_tests=True,
        test_cases=["assert solution() == 42"]
    )
    
    assert result.execution_result is not None
    assert result.execution_result.passed is True
    assert result.execution_result.num_passed == 1
    assert result.execution_result.num_total == 1
    
    print("✓ test_solve_problem_with_tests passed")


def test_solve_batch():
    """Test batch inference."""
    generator = MockGenerator()
    discriminator = MockDiscriminator()
    sandbox = MockSandbox()
    
    engine = InferenceEngine(
        generator=generator,
        discriminator=discriminator,
        sandbox=sandbox
    )
    
    problems = [
        Problem(
            id="test1",
            description="Problem 1",
            function_signature="def func1():",
            baseline_tests=["assert func1() == 1"],
            reference_solution="def func1(): return 1",
            difficulty="easy",
            tags=["test"]
        ),
        Problem(
            id="test2",
            description="Problem 2",
            function_signature="def func2():",
            baseline_tests=["assert func2() == 2"],
            reference_solution="def func2(): return 2",
            difficulty="easy",
            tags=["test"]
        )
    ]
    
    results = engine.solve_batch(problems, execute_tests=False)
    
    assert len(results) == 2
    assert all(isinstance(r, InferenceResult) for r in results)
    assert results[0].problem_description == "Problem 1"
    assert results[1].problem_description == "Problem 2"
    
    print("✓ test_solve_batch passed")


def test_get_reasoning_chain():
    """Test getting reasoning chain only."""
    generator = MockGenerator()
    discriminator = MockDiscriminator()
    sandbox = MockSandbox()
    
    engine = InferenceEngine(
        generator=generator,
        discriminator=discriminator,
        sandbox=sandbox
    )
    
    reasoning_chain = engine.get_reasoning_chain(
        problem_description="Write a function"
    )
    
    assert len(reasoning_chain) == 5  # All 5 stages
    assert all(isinstance(s, str) for s in reasoning_chain)
    assert "Stage 1" in reasoning_chain[0]
    assert "def solution()" in reasoning_chain[4]  # Final code
    
    print("✓ test_get_reasoning_chain passed")


def test_from_checkpoint():
    """Test loading inference engine from checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock checkpoint
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
        
        # Create mock model state dicts
        mock_state_dict = MockModel().state_dict()
        
        checkpoint_data = {
            'stage': 5,
            'epoch': 10,
            'generator_state_dict': mock_state_dict,
            'discriminator_state_dict': mock_state_dict,
            'metrics': {
                'generator_reward': 0.75,
                'test_validity': 0.90
            },
            'config': {
                'generator_model': 'test-gen',
                'discriminator_model': 'test-disc',
                'device': 'cpu'
            }
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Note: This test would require actual model loading
        # For now, we just verify the checkpoint file exists
        assert checkpoint_path.exists()
        
        # Load checkpoint data to verify structure
        loaded_data = torch.load(checkpoint_path)
        assert loaded_data['stage'] == 5
        assert 'generator_state_dict' in loaded_data
        assert 'discriminator_state_dict' in loaded_data
        
    print("✓ test_from_checkpoint passed")


def test_inference_result_structure():
    """Test InferenceResult dataclass structure."""
    result = InferenceResult(
        problem_description="Test problem",
        reasoning_chain=["stage1", "stage2", "stage3", "stage4"],
        generated_code="def test(): pass",
        execution_result=None,
        inference_time=1.5,
        stage_times=[0.3, 0.3, 0.3, 0.3, 0.3]
    )
    
    assert result.problem_description == "Test problem"
    assert len(result.reasoning_chain) == 4
    assert result.generated_code == "def test(): pass"
    assert result.execution_result is None
    assert result.inference_time == 1.5
    assert len(result.stage_times) == 5
    
    print("✓ test_inference_result_structure passed")


def test_inference_with_empty_problem():
    """Test inference with empty problem description."""
    generator = MockGenerator()
    discriminator = MockDiscriminator()
    sandbox = MockSandbox()
    
    engine = InferenceEngine(
        generator=generator,
        discriminator=discriminator,
        sandbox=sandbox
    )
    
    result = engine.solve_problem(
        problem_description="",
        execute_tests=False
    )
    
    # Should still generate output even with empty problem
    assert isinstance(result, InferenceResult)
    assert result.problem_description == ""
    assert len(result.reasoning_chain) == 4
    
    print("✓ test_inference_with_empty_problem passed")


def run_all_tests():
    """Run all inference tests."""
    print("\n" + "="*60)
    print("RUNNING INFERENCE TESTS")
    print("="*60 + "\n")
    
    test_inference_engine_init()
    test_solve_problem()
    test_solve_problem_with_tests()
    test_solve_batch()
    test_get_reasoning_chain()
    test_from_checkpoint()
    test_inference_result_structure()
    test_inference_with_empty_problem()
    
    print("\n" + "="*60)
    print("ALL INFERENCE TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
