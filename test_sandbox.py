"""Test sandbox execution with actual problems."""

from data.problem_dataset import load_problems
from sandbox.sandbox import Sandbox

problems = load_problems("data/custom_problems.json")
sandbox = Sandbox(timeout=5)

print("Testing sandbox with reference solutions...\n")

for i, problem in enumerate(problems[:2], 1):  # Test first 2
    print(f"Problem {i}: {problem.id}")
    print(f"  Tests: {len(problem.baseline_tests)}")
    
    # Combine tests
    test_code = "\n".join(problem.baseline_tests)
    print(f"  Test code length: {len(test_code)} chars")
    
    # Execute
    result = sandbox.execute_tests(problem.reference_solution, test_code)
    
    print(f"  Result: {result.num_passed}/{result.num_total} passed")
    print(f"  Passed: {result.passed}")
    
    if result.errors:
        print(f"  Errors:")
        for error in result.errors[:5]:
            print(f"    {error}")
    
    print(f"  Stderr: {result.stderr[:500]}")
    
    print()
