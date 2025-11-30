"""Debug script to check problem loading."""

from data.problem_dataset import load_problems

problems = load_problems("data/custom_problems.json")

print(f"Loaded {len(problems)} problems\n")

for i, problem in enumerate(problems, 1):
    print(f"Problem {i}: {problem.id}")
    print(f"  Description: {problem.description[:60]}...")
    print(f"  Baseline tests: {len(problem.baseline_tests)}")
    
    if problem.baseline_tests:
        print(f"  First test: {problem.baseline_tests[0]}")
    else:
        print(f"  WARNING: No baseline tests!")
    
    print(f"  Reference solution: {len(problem.reference_solution)} chars")
    print()
