"""Validate project structure without requiring heavy dependencies."""

import os
from pathlib import Path


def check_file_exists(filepath):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {filepath}")
        return True
    else:
        print(f"✗ {filepath} - MISSING")
        return False


def check_directory_structure():
    """Check that all required directories and files exist."""
    print("="*60)
    print("PROJECT STRUCTURE VALIDATION")
    print("="*60)
    
    required_files = [
        # Core modules
        "models/__init__.py",
        "models/generator.py",
        "models/discriminator.py",
        
        # Reasoning
        "reasoning/__init__.py",
        "reasoning/stages.py",
        
        # Sandbox
        "sandbox/__init__.py",
        "sandbox/sandbox.py",
        
        # Training
        "training/__init__.py",
        "training/adversarial_trainer.py",
        "training/rl_loop.py",
        "training/reward.py",
        "training/config.py",
        "training/multi_attempt.py",
        
        # Evaluation
        "evaluation/__init__.py",
        "evaluation/metrics.py",
        
        # Data
        "data/__init__.py",
        "data/problem_dataset.py",
        "data/example_problems.json",
        
        # Main files
        "run_training.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        
        # Tests
        "test_basic.py",
    ]
    
    print("\nChecking files...")
    results = [check_file_exists(f) for f in required_files]
    
    print(f"\n{'='*60}")
    print(f"Files found: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All required files present!")
        return True
    else:
        print("✗ Some files are missing")
        return False


def check_python_syntax():
    """Check Python files for syntax errors."""
    print(f"\n{'='*60}")
    print("SYNTAX VALIDATION")
    print("="*60)
    
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    
    print(f"\nChecking {len(python_files)} Python files...")
    
    errors = []
    for filepath in python_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                compile(f.read(), filepath, 'exec')
            print(f"✓ {filepath}")
        except SyntaxError as e:
            print(f"✗ {filepath} - SYNTAX ERROR: {e}")
            errors.append((filepath, e))
    
    print(f"\n{'='*60}")
    if errors:
        print(f"✗ Found {len(errors)} syntax errors")
        return False
    else:
        print("✓ All Python files have valid syntax!")
        return True


def check_json_files():
    """Check JSON files are valid."""
    print(f"\n{'='*60}")
    print("JSON VALIDATION")
    print("="*60)
    
    import json
    
    json_files = ["data/example_problems.json"]
    
    errors = []
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {filepath} - Valid JSON")
            
            # Check structure
            if 'problems' in data:
                print(f"  - Contains {len(data['problems'])} problems")
        except Exception as e:
            print(f"✗ {filepath} - ERROR: {e}")
            errors.append((filepath, e))
    
    print(f"\n{'='*60}")
    if errors:
        print(f"✗ Found {len(errors)} JSON errors")
        return False
    else:
        print("✓ All JSON files are valid!")
        return True


def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("ADVERSARIAL RL SYSTEM - STRUCTURE VALIDATION")
    print("="*60 + "\n")
    
    results = []
    
    # Check structure
    results.append(check_directory_structure())
    
    # Check syntax
    results.append(check_python_syntax())
    
    # Check JSON
    results.append(check_json_files())
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Checks passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓✓✓ PROJECT STRUCTURE IS VALID ✓✓✓")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python run_training.py")
        return 0
    else:
        print("\n✗ Some validation checks failed")
        return 1


if __name__ == "__main__":
    exit(main())
