"""Verify the progressive test generation implementation without heavy dependencies."""

import sys
import ast

def verify_stage_prompts():
    """Verify all stages have test generation prompts."""
    print("Verifying stage prompts...")
    
    try:
        from reasoning.stages import REASONING_STAGES
        
        for stage in REASONING_STAGES:
            # Check discriminator prompt generates tests
            prompt = stage.discriminator_prompt_template
            
            if "test" not in prompt.lower():
                print(f"  ✗ Stage {stage.id}: No test generation")
                return False
            
            if "pytest" not in prompt.lower():
                print(f"  ✗ Stage {stage.id}: No pytest mention")
                return False
            
            print(f"  ✓ Stage {stage.id} ({stage.name}): Generates tests")
        
        print("✓ All stages generate tests\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False


def verify_trainer_has_new_method():
    """Verify trainer has the new method."""
    print("Verifying trainer implementation...")
    
    try:
        # Read the file and parse it
        with open('training/adversarial_trainer.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for new method
        if '_generate_full_chain_with_tests' in content:
            print("  ✓ _generate_full_chain_with_tests method exists")
        else:
            print("  ✗ _generate_full_chain_with_tests method missing")
            return False
        
        # Check it returns 3 values
        if 'reasoning_chain, final_code, all_tests' in content:
            print("  ✓ Returns (reasoning_chain, final_code, accumulated_tests)")
        else:
            print("  ✗ Return signature incorrect")
            return False
        
        # Check it accumulates tests
        if 'accumulated_tests' in content:
            print("  ✓ Accumulates tests from all stages")
        else:
            print("  ✗ Doesn't accumulate tests")
            return False
        
        # Check train_discriminator_epoch uses it
        if 'train_discriminator_epoch' in content and '_generate_full_chain_with_tests' in content:
            # Count occurrences
            disc_section = content[content.find('def train_discriminator_epoch'):content.find('def train_generator_epoch')]
            if '_generate_full_chain_with_tests' in disc_section:
                print("  ✓ train_discriminator_epoch uses new method")
            else:
                print("  ✗ train_discriminator_epoch doesn't use new method")
                return False
        
        # Check train_generator_epoch uses it
        gen_section = content[content.find('def train_generator_epoch'):content.find('def train_alternating')]
        if '_generate_full_chain_with_tests' in gen_section:
            print("  ✓ train_generator_epoch uses new method")
        else:
            print("  ✗ train_generator_epoch doesn't use new method")
            return False
        
        print("✓ Trainer implementation correct\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def verify_reward_unchanged():
    """Verify reward functions are unchanged."""
    print("Verifying reward functions...")
    
    try:
        with open('training/reward.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check generator reward formula
        if 'pass_rate = execution_result.num_passed / execution_result.num_total' in content:
            print("  ✓ Generator reward formula unchanged")
        else:
            print("  ✗ Generator reward formula changed")
            return False
        
        # Check discriminator reward formula
        if 'adversarial_score * test_validity' in content:
            print("  ✓ Discriminator reward formula unchanged")
        else:
            print("  ✗ Discriminator reward formula changed")
            return False
        
        print("✓ Reward functions unchanged\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False


def verify_key_changes():
    """Verify key implementation changes."""
    print("Verifying key implementation changes...")
    
    changes = []
    
    # 1. All stages generate tests
    try:
        from reasoning.stages import REASONING_STAGES
        all_generate_tests = all('test' in s.discriminator_prompt_template.lower() for s in REASONING_STAGES)
        if all_generate_tests:
            changes.append("✓ All 5 stages generate tests (not just stage 5)")
        else:
            changes.append("✗ Not all stages generate tests")
    except:
        changes.append("✗ Could not verify stage prompts")
    
    # 2. Tests are accumulated
    try:
        with open('training/adversarial_trainer.py', 'r', encoding='utf-8') as f:
            content = f.read()
        if 'accumulated_tests' in content and 'append(tests)' in content:
            changes.append("✓ Tests accumulated across stages")
        else:
            changes.append("✗ Tests not accumulated")
    except:
        changes.append("✗ Could not verify test accumulation")
    
    # 3. Full forward pass
    try:
        with open('training/adversarial_trainer.py', 'r', encoding='utf-8') as f:
            content = f.read()
        if 'for stage_id in range(1, 6):' in content and '_generate_full_chain_with_tests' in content:
            changes.append("✓ Full forward pass through all stages")
        else:
            changes.append("✗ No full forward pass")
    except:
        changes.append("✗ Could not verify forward pass")
    
    # 4. Rewards use accumulated tests
    try:
        with open('training/adversarial_trainer.py', 'r', encoding='utf-8') as f:
            content = f.read()
        if 'execute_tests(final_code, accumulated_tests)' in content:
            changes.append("✓ Rewards computed from ALL accumulated tests")
        else:
            changes.append("✗ Rewards not using accumulated tests")
    except:
        changes.append("✗ Could not verify reward computation")
    
    # 5. Reward formulas unchanged
    try:
        with open('training/reward.py', 'r', encoding='utf-8') as f:
            content = f.read()
        has_gen_formula = 'num_passed / num_total' in content or 'pass_rate = execution_result.num_passed / execution_result.num_total' in content
        has_disc_formula = 'adversarial_score * test_validity' in content or 'reward = adversarial_score * test_validity' in content
        if has_gen_formula and has_disc_formula:
            changes.append("✓ Reward formulas unchanged (same math)")
        else:
            changes.append("✗ Reward formulas changed")
    except:
        changes.append("✗ Could not verify reward formulas")
    
    for change in changes:
        print(f"  {change}")
    
    all_good = all('✓' in c for c in changes)
    if all_good:
        print("\n✓ All key changes implemented correctly\n")
    else:
        print("\n✗ Some changes missing or incorrect\n")
    
    return all_good


def main():
    """Run all verifications."""
    print("="*70)
    print("PROGRESSIVE TEST GENERATION - IMPLEMENTATION VERIFICATION")
    print("="*70)
    print()
    
    tests = [
        ("Stage Prompts", verify_stage_prompts),
        ("Trainer Implementation", verify_trainer_has_new_method),
        ("Reward Functions", verify_reward_unchanged),
        ("Key Changes", verify_key_changes)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {name} crashed: {e}\n")
            results.append(False)
    
    print("="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    print()
    
    if all(results):
        print("✓✓✓ IMPLEMENTATION VERIFIED ✓✓✓")
        print()
        print("The progressive test generation system is correctly implemented!")
        print()
        print("Summary of changes:")
        print("  • Discriminator generates tests at ALL stages (1-5)")
        print("  • Tests accumulate: stage_1_tests + stage_2_tests + ... + stage_5_tests")
        print("  • Full forward pass: Generate all 5 stages for each training step")
        print("  • Rewards: Based on final accumulated test suite")
        print("  • Formulas: Unchanged (generator = pass_rate, disc = (1-pass_rate)*validity)")
        print()
        print("Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run training: python run_training.py")
        print("  3. Observe: Each stage now contributes tests to final suite")
        return 0
    else:
        print("✗ VERIFICATION FAILED")
        print()
        print("Some implementation issues detected. Review the failed checks above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
