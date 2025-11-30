"""Comprehensive test to see what the model generates and why tests fail."""

from models.generator import LLMGenerator
from reasoning.stages import get_stage
from data.problem_dataset import load_problems
from sandbox.sandbox_simple import execute_tests_simple

# Load model
print("Loading model...")
generator = LLMGenerator(model_name="Salesforce/codegen-350M-mono", device="cpu")

# Load problems
problems = load_problems("data/custom_problems.json")

# Test first 2 problems
for i, problem in enumerate(problems[:2]):
    print("\n" + "="*80)
    print(f"PROBLEM {i+1}: {problem.id}")
    print("="*80)
    print(f"Description: {problem.description[:100]}...")
    print(f"\nFunction Signature:")
    print(problem.function_signature)
    print(f"\nBaseline Tests:")
    for test in problem.baseline_tests:
        print(f"  - {test[:80]}...")
    
    # Generate code
    print(f"\nGenerating code (this takes ~30-60 seconds on CPU)...")
    stage_5 = get_stage(5)
    code = generator.generate_code(
        problem=problem.description,
        reasoning_chain=[],
        prompt_template=stage_5.generator_prompt_template,
        max_new_tokens=256,
        temperature=0.7,
        function_signature=problem.function_signature
    )
    
    print("\n" + "-"*80)
    print("GENERATED CODE:")
    print("-"*80)
    print(code)
    print("-"*80)
    
    # Test it
    print("\nTesting generated code...")
    test_code = "\n".join(problem.baseline_tests)
    result = execute_tests_simple(code, test_code, timeout=5)
    
    print(f"\nRESULTS:")
    print(f"  Pass rate: {result.num_passed}/{result.num_total}")
    print(f"  Passed: {result.passed}")
    
    if not result.passed:
        print(f"\n  ERRORS:")
        for j, error in enumerate(result.errors[:5], 1):
            if error.strip():
                print(f"    {j}. {error[:200]}")
    
    print("\n" + "="*80)
    
    # Compare with reference solution
    print("\nREFERENCE SOLUTION (for comparison):")
    print("-"*80)
    print(problem.reference_solution[:300] + "...")
    print("-"*80)

print("\n\nTest complete!")
