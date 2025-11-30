"""Validate Groq-generated problems and fix issues."""

import json
from pathlib import Path
from sandbox.sandbox_simple import execute_tests_simple

def validate_problems(filepath: str):
    """Validate that problems have working reference solutions.
    
    Args:
        filepath: Path to problems JSON file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = data['problems']
    valid_problems = []
    
    print(f"Validating {len(problems)} problems...\n")
    
    for i, problem in enumerate(problems, 1):
        print(f"Problem {i}: {problem['id']}")
        
        # Check required fields
        if not all(k in problem for k in ['id', 'description', 'function_signature', 
                                           'baseline_tests', 'reference_solution']):
            print(f"  ✗ Missing required fields")
            continue
        
        # Test reference solution against baseline tests
        test_code = "\n".join(problem['baseline_tests'])
        result = execute_tests_simple(problem['reference_solution'], test_code, timeout=5)
        
        print(f"  Tests: {result.num_passed}/{result.num_total} passed")
        
        if result.passed:
            print(f"  ✓ Valid - reference solution passes all tests")
            valid_problems.append(problem)
        else:
            print(f"  ✗ Invalid - reference solution fails tests")
            if result.errors:
                print(f"     Error: {result.errors[0][:100]}")
    
    print(f"\n{'='*60}")
    print(f"Valid problems: {len(valid_problems)}/{len(problems)}")
    print(f"{'='*60}\n")
    
    # Save valid problems
    if valid_problems:
        output_file = filepath.replace('.json', '_validated.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"problems": valid_problems}, f, indent=2)
        
        print(f"✓ Saved {len(valid_problems)} valid problems to {output_file}")
        
        if len(valid_problems) < len(problems):
            print(f"\nRemoved {len(problems) - len(valid_problems)} invalid problems")
            print("You can use the validated file for training:")
            print(f"  python run_training.py --problems-file {output_file}")
    else:
        print("✗ No valid problems found!")
        print("Recommendation: Use pre-made problems instead:")
        print("  python generate_custom_problems.py")
    
    return valid_problems


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate problem file")
    parser.add_argument(
        '--file',
        type=str,
        default="data/custom_problems.json",
        help="Path to problems JSON file"
    )
    
    args = parser.parse_args()
    
    validate_problems(args.file)
