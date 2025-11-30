"""Quick test to verify problems are valid."""

import json
from data.problem_dataset import load_problems
from sandbox.sandbox import Sandbox

# Load problems
problems = load_problems("data/custom_problems.json")

print(f"Loaded {len(problems)} problems\n")

# Test each problem
sandbox = Sandbox(timeout=5)

for i, problem in enumerate(problems, 1):
    print(f"Problem {i}: {problem.id}")
    print(f"  Description: {problem.description[:60]}...")
    print(f"  Signature: {problem.function_signature}")
    print(f"  Tests: {len(problem.baseline_tests)} test cases")
    
    # Try to execute reference solution with tests
    test_code = "\n".join(problem.baseline_tests)
    result = sandbox.execute_tests(problem.reference_solution, test_code)
    
    print(f"  Reference solution: {result.num_passed}/{result.num_total} tests passed")
    
    if not result.passed:
        print(f"  Errors: {result.errors[:2]}")
    
    print()

print("Done!")
