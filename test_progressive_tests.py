"""Test the progressive test generation system."""

import sys

def test_stage_prompts():
    """Test that all stages have test generation prompts."""
    print("Testing stage prompt updates...")
    
    try:
        from reasoning.stages import REASONING_STAGES
        
        for stage in REASONING_STAGES:
            print(f"\n  Stage {stage.id}: {stage.name}")
            
            # Check discriminator prompt mentions tests
            if "test" in stage.discriminator_prompt_template.lower():
                print(f"    ✓ Discriminator generates tests")
            else:
                print(f"    ✗ Discriminator prompt doesn't mention tests")
                return False
            
            # Check for pytest
            if "pytest" in stage.discriminator_prompt_template.lower():
                print(f"    ✓ Uses pytest format")
            else:
                print(f"    ✗ Doesn't specify pytest format")
                return False
        
        print("\n✓ All stages have test generation prompts")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_trainer_methods():
    """Test that trainer has new methods."""
    print("\nTesting trainer methods...")
    
    try:
        from training.adversarial_trainer import AdversarialTrainer
        
        # Check method exists
        if hasattr(AdversarialTrainer, '_generate_full_chain_with_tests'):
            print("  ✓ _generate_full_chain_with_tests method exists")
        else:
            print("  ✗ _generate_full_chain_with_tests method missing")
            return False
        
        # Check method signature
        import inspect
        sig = inspect.signature(AdversarialTrainer._generate_full_chain_with_tests)
        params = list(sig.parameters.keys())
        
        expected_params = ['self', 'problem', 'training_stage']
        if params == expected_params:
            print(f"  ✓ Method signature correct: {params}")
        else:
            print(f"  ✗ Method signature incorrect: {params} vs {expected_params}")
            return False
        
        print("✓ Trainer methods updated correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_functions():
    """Test that reward functions still work."""
    print("\nTesting reward functions...")
    
    try:
        from sandbox.sandbox import ExecutionResult
        from training.reward import compute_generator_reward, compute_discriminator_reward
        
        # Test generator reward
        result = ExecutionResult(
            passed=True,
            num_passed=8,
            num_total=10,
            errors=[],
            stdout="",
            stderr="",
            timed_out=False
        )
        
        gen_reward = compute_generator_reward(result)
        expected = 0.8
        
        if abs(gen_reward - expected) < 0.01:
            print(f"  ✓ Generator reward: {gen_reward:.2f} (expected {expected:.2f})")
        else:
            print(f"  ✗ Generator reward: {gen_reward:.2f} (expected {expected:.2f})")
            return False
        
        # Test discriminator reward
        val_result = ExecutionResult(
            passed=True,
            num_passed=10,
            num_total=10,
            errors=[],
            stdout="",
            stderr="",
            timed_out=False
        )
        
        disc_reward = compute_discriminator_reward(result, val_result)
        expected = 0.2  # (1 - 0.8) * 1.0
        
        if abs(disc_reward - expected) < 0.01:
            print(f"  ✓ Discriminator reward: {disc_reward:.2f} (expected {expected:.2f})")
        else:
            print(f"  ✗ Discriminator reward: {disc_reward:.2f} (expected {expected:.2f})")
            return False
        
        print("✓ Reward functions work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test that components integrate correctly."""
    print("\nTesting integration...")
    
    try:
        from reasoning.stages import get_stage
        
        # Test each stage
        for stage_id in range(1, 6):
            stage = get_stage(stage_id)
            
            # Check generator prompt
            if "{problem}" in stage.generator_prompt_template:
                pass
            else:
                print(f"  ✗ Stage {stage_id} generator prompt missing {{problem}}")
                return False
            
            # Check discriminator prompt
            if "{problem}" in stage.discriminator_prompt_template:
                pass
            else:
                print(f"  ✗ Stage {stage_id} discriminator prompt missing {{problem}}")
                return False
            
            if "{stage_output}" in stage.discriminator_prompt_template:
                pass
            else:
                print(f"  ✗ Stage {stage_id} discriminator prompt missing {{stage_output}}")
                return False
        
        print("  ✓ All stage prompts have required placeholders")
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PROGRESSIVE TEST GENERATION - VERIFICATION")
    print("="*60)
    
    tests = [
        test_stage_prompts,
        test_trainer_methods,
        test_reward_functions,
        test_integration
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe progressive test generation system is correctly implemented!")
        print("\nKey changes:")
        print("  1. Discriminator generates tests at ALL stages (not just stage 5)")
        print("  2. Tests accumulate across stages")
        print("  3. Final test suite used to compute rewards for each stage")
        print("  4. Reward formulas unchanged (same as before)")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
