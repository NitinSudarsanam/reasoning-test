"""Manual test to see what's wrong."""

import tempfile
from pathlib import Path

# Get first problem
from data.problem_dataset import load_problems
problems = load_problems("data/custom_problems.json")
problem = problems[0]

print(f"Testing: {problem.id}\n")

# Create temp dir
with tempfile.TemporaryDirectory() as tmpdir:
    # Write solution
    code_file = Path(tmpdir) / "solution.py"
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(problem.reference_solution)
    
    # Wrap tests
    test_lines = [line.strip() for line in problem.baseline_tests]
    test_functions = []
    for i, line in enumerate(test_lines):
        test_functions.append(f"def test_case_{i}():\n    {line}\n")
    tests_wrapped = "\n".join(test_functions)
    
    # Write test file
    test_file = Path(tmpdir) / "test_solution.py"
    test_content = f"""
import sys
sys.path.insert(0, r'{tmpdir}')
from solution import *

{tests_wrapped}
"""
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("Test file content:")
    print(test_content)
    print("\n" + "="*60 + "\n")
    
    # Try to run pytest
    import subprocess
    result = subprocess.run(
        ['pytest', str(test_file), '-v'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
