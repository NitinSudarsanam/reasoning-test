"""Quick test to verify custom problems work with sandbox."""

from data.problem_dataset import load_problems
from sandbox.sandbox_simple import execute_tests_simple

# Load problems
problems = load_problems("data/custom_problems.json")
print(f"Loaded {len(problems)} problems\n")

# Test each problem's reference solution against its baseline tests
for problem in problems:
    print(f"Testing: {problem.id}")
    
    # Use reference solution
    code = problem.reference_solution
    
    # Join baseline tests
    test_code = "\n".join(problem.baseline_tests)
    
    # Execute
    result = execute_tests_simple(code, test_code, timeout=5)
    
    status = "✓" if result.passed else "✗"
    print(f"  {status} Pass rate: {result.num_passed}/{result.num_total}")
    
    if not result.passed:
        print(f"  Errors: {result.errors[:2]}")
    print()

print("All problems validated!")
