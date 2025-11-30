"""Test the simple sandbox."""

from data.problem_dataset import load_problems
from sandbox.sandbox_simple import execute_tests_simple

problems = load_problems("data/custom_problems.json")

print("Testing simple sandbox...\n")

for i, problem in enumerate(problems[:2], 1):
    print(f"Problem {i}: {problem.id}")
    
    # Combine tests
    test_code = "\n".join(problem.baseline_tests)
    
    # Execute
    result = execute_tests_simple(problem.reference_solution, test_code)
    
    print(f"  Result: {result.num_passed}/{result.num_total} passed")
    print(f"  Passed: {result.passed}")
    
    if result.errors:
        print(f"  Errors: {result.errors[:2]}")
    
    print()
