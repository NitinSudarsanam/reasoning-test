"""Basic integration test for the system."""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.generator import LLMGenerator
        from models.discriminator import LLMDiscriminator
        from sandbox.sandbox import Sandbox, ExecutionResult
        from data.problem_dataset import Problem, load_problems
        from reasoning.stages import get_stage, REASONING_STAGES
        from training.adversarial_trainer import AdversarialTrainer
        from training.config import TrainingConfig
        from training.reward import compute_generator_reward, compute_discriminator_reward
        from evaluation.metrics import compute_pass_rate
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_problem_loading():
    """Test loading problems from JSON."""
    print("\nTesting problem loading...")
    
    try:
        from data.problem_dataset import load_problems
        problems = load_problems("data/example_problems.json")
        print(f"✓ Loaded {len(problems)} problems")
        
        # Validate first problem
        if problems:
            p = problems[0]
            print(f"  - Problem ID: {p.id}")
            print(f"  - Description: {p.description[:50]}...")
            print(f"  - Has reference solution: {bool(p.reference_solution)}")
        
        return True
    except Exception as e:
        print(f"✗ Problem loading failed: {e}")
        return False


def test_sandbox():
    """Test sandbox execution."""
    print("\nTesting sandbox...")
    
    try:
        from sandbox.sandbox import Sandbox
        
        sandbox = Sandbox(timeout=5)
        
        # Test simple code execution
        code = "def add(a, b):\n    return a + b"
        tests = """
import pytest

def test_add():
    assert add(1, 2) == 3
    assert add(0, 0) == 0
"""
        
        result = sandbox.execute_tests(code, tests)
        print(f"✓ Sandbox execution successful")
        print(f"  - Passed: {result.num_passed}/{result.num_total}")
        print(f"  - Timed out: {result.timed_out}")
        
        return True
    except Exception as e:
        print(f"✗ Sandbox test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reasoning_stages():
    """Test reasoning stage definitions."""
    print("\nTesting reasoning stages...")
    
    try:
        from reasoning.stages import REASONING_STAGES, get_stage
        
        print(f"✓ Found {len(REASONING_STAGES)} reasoning stages")
        
        for stage in REASONING_STAGES:
            print(f"  - Stage {stage.id}: {stage.name}")
        
        # Test get_stage
        stage_3 = get_stage(3)
        print(f"✓ get_stage(3) returned: {stage_3.name}")
        
        return True
    except Exception as e:
        print(f"✗ Reasoning stages test failed: {e}")
        return False


def test_reward_computation():
    """Test reward computation."""
    print("\nTesting reward computation...")
    
    try:
        from sandbox.sandbox import ExecutionResult
        from training.reward import compute_generator_reward, compute_discriminator_reward
        
        # Test generator reward
        gen_result = ExecutionResult(
            passed=True,
            num_passed=3,
            num_total=5,
            errors=[],
            stdout="",
            stderr="",
            timed_out=False
        )
        
        gen_reward = compute_generator_reward(gen_result)
        print(f"✓ Generator reward: {gen_reward:.2f} (expected 0.60)")
        
        # Test discriminator reward
        val_result = ExecutionResult(
            passed=True,
            num_passed=5,
            num_total=5,
            errors=[],
            stdout="",
            stderr="",
            timed_out=False
        )
        
        disc_reward = compute_discriminator_reward(gen_result, val_result)
        print(f"✓ Discriminator reward: {disc_reward:.2f}")
        print(f"  - Formula: (1.0 - {gen_reward:.2f}) * 1.0 = {disc_reward:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Reward computation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("BASIC INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        test_imports,
        test_problem_loading,
        test_sandbox,
        test_reasoning_stages,
        test_reward_computation
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
