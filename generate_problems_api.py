"""Generate custom coding problems using OpenAI or Anthropic APIs."""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional


def generate_problem_with_openai(
    client,
    difficulty: str,
    topic: str,
    model: str = "gpt-3.5-turbo"
) -> Dict:
    """Generate a problem using OpenAI API.
    
    Args:
        client: OpenAI client instance
        difficulty: Problem difficulty (easy/medium/hard)
        topic: Problem topic/domain
        model: Model to use (gpt-3.5-turbo, gpt-3.5-turbo, etc.)
        
    Returns:
        Dictionary with problem details
    """
    
    prompt = f"""Generate a novel coding problem that is unlikely to appear in standard coding interview datasets.

Requirements:
- Difficulty: {difficulty}
- Topic: {topic}
- Must be a unique variation or combination of concepts
- Avoid common LeetCode/HackerRank patterns

Provide the following in this EXACT format:

PROBLEM_ID: <snake_case_id>

DESCRIPTION:
<2-3 sentence clear problem description>

SIGNATURE:
<Python function signature with type hints>

TESTS:
<5 test cases as Python assert statements, one per line>

SOLUTION:
```python
<complete working Python solution>
```

Make the problem interesting and include edge cases in tests."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at creating novel coding problems."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=1500
    )
    
    generated_text = response.choices[0].message.content
    problem = parse_generated_problem(generated_text)
    problem['difficulty'] = difficulty
    problem['tags'] = [topic]
    
    return problem


def generate_problem_with_anthropic(
    client,
    difficulty: str,
    topic: str,
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict:
    """Generate a problem using Anthropic API.
    
    Args:
        client: Anthropic client instance
        difficulty: Problem difficulty (easy/medium/hard)
        topic: Problem topic/domain
        model: Model to use
        
    Returns:
        Dictionary with problem details
    """
    
    prompt = f"""Generate a novel coding problem that is unlikely to appear in standard coding interview datasets.

Requirements:
- Difficulty: {difficulty}
- Topic: {topic}
- Must be a unique variation or combination of concepts
- Avoid common LeetCode/HackerRank patterns

Provide the following in this EXACT format:

PROBLEM_ID: <snake_case_id>

DESCRIPTION:
<2-3 sentence clear problem description>

SIGNATURE:
<Python function signature with type hints>

TESTS:
<5 test cases as Python assert statements, one per line>

SOLUTION:
```python
<complete working Python solution>
```

Make the problem interesting and include edge cases in tests."""

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        temperature=0.8,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    generated_text = response.content[0].text
    problem = parse_generated_problem(generated_text)
    problem['difficulty'] = difficulty
    problem['tags'] = [topic]
    
    return problem


def parse_generated_problem(text: str) -> Dict:
    """Parse LLM-generated problem text into structured format.
    
    Args:
        text: Generated text from LLM
        
    Returns:
        Dictionary with problem fields
    """
    problem = {}
    
    # Extract problem ID
    if "PROBLEM_ID:" in text:
        problem_id = text.split("PROBLEM_ID:")[1].split("\n")[0].strip()
        problem['id'] = problem_id
    else:
        problem['id'] = "generated_problem"
    
    # Extract description
    if "DESCRIPTION:" in text and "SIGNATURE:" in text:
        desc = text.split("DESCRIPTION:")[1].split("SIGNATURE:")[0].strip()
        problem['description'] = desc
    else:
        problem['description'] = "Generated problem"
    
    # Extract signature
    if "SIGNATURE:" in text and "TESTS:" in text:
        sig = text.split("SIGNATURE:")[1].split("TESTS:")[0].strip()
        problem['function_signature'] = sig
    else:
        problem['function_signature'] = "def solution():"
    
    # Extract tests
    if "TESTS:" in text and "SOLUTION:" in text:
        tests_text = text.split("TESTS:")[1].split("SOLUTION:")[0].strip()
        tests = [t.strip() for t in tests_text.split("\n") if t.strip() and "assert" in t]
        problem['baseline_tests'] = tests[:5]  # Max 5 tests
    else:
        problem['baseline_tests'] = []
    
    # Extract solution
    if "SOLUTION:" in text:
        solution = text.split("SOLUTION:")[1].strip()
        # Extract code block if present
        if "```python" in solution:
            solution = solution.split("```python")[1].split("```")[0].strip()
        elif "```" in solution:
            solution = solution.split("```")[1].split("```")[0].strip()
        problem['reference_solution'] = solution
    else:
        problem['reference_solution'] = "def solution(): pass"
    
    return problem


def generate_problem_with_groq(
    client,
    difficulty: str,
    topic: str,
    model: str = "llama-3.3-70b-versatile"
) -> Dict:
    """Generate a problem using Groq API (free).
    
    Args:
        client: Groq client instance
        difficulty: Problem difficulty (easy/medium/hard)
        topic: Problem topic/domain
        model: Model to use
        
    Returns:
        Dictionary with problem details
    """
    
    prompt = f"""Generate a novel coding problem that is unlikely to appear in standard coding interview datasets.

Requirements:
- Difficulty: {difficulty}
- Topic: {topic}
- Must be a unique variation or combination of concepts
- Avoid common LeetCode/HackerRank patterns

Provide the following in this EXACT format:

PROBLEM_ID: <snake_case_id>

DESCRIPTION:
<2-3 sentence clear problem description>

SIGNATURE:
<Python function signature with type hints>

TESTS:
<5 test cases as Python assert statements, one per line>

SOLUTION:
```python
<complete working Python solution>
```

Make the problem interesting and include edge cases in tests."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at creating novel coding problems."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=1500
    )
    
    generated_text = response.choices[0].message.content
    problem = parse_generated_problem(generated_text)
    problem['difficulty'] = difficulty
    problem['tags'] = [topic]
    
    return problem


def generate_problem_set(
    provider: str = "openai",
    model: Optional[str] = None,
    num_problems: int = 5,
    api_key: Optional[str] = None
) -> List[Dict]:
    """Generate a set of problems using an API.
    
    Args:
        provider: API provider (openai, anthropic, or groq)
        model: Model name (optional, uses defaults)
        num_problems: Number of problems to generate
        api_key: API key (optional, reads from env)
        
    Returns:
        List of problem dictionaries
    """
    print(f"\n{'='*60}")
    print(f"GENERATING PROBLEMS WITH {provider.upper()} API")
    print(f"{'='*60}\n")
    
    # Initialize client
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("Error: openai package not installed")
            print("Install with: pip install openai")
            return []
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment")
            print("Set it with: export OPENAI_API_KEY='your-key'")
            return []
        
        client = OpenAI(api_key=api_key)
        model = model or "gpt-3.5-turbo"
        
    elif provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            print("Error: anthropic package not installed")
            print("Install with: pip install anthropic")
            return []
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found in environment")
            print("Set it with: export ANTHROPIC_API_KEY='your-key'")
            return []
        
        client = Anthropic(api_key=api_key)
        model = model or "claude-3-5-sonnet-20241022"
    
    elif provider == "groq":
        try:
            from groq import Groq
        except ImportError:
            print("Error: groq package not installed")
            print("Install with: pip install groq")
            return []
        
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Error: GROQ_API_KEY not found in environment")
            print("Get free key at: https://console.groq.com/")
            print("Set it with: export GROQ_API_KEY='your-key'")
            return []
        
        client = Groq(api_key=api_key)
        model = model or "llama-3.3-70b-versatile"
    
    else:
        print(f"Error: Unknown provider '{provider}'")
        print("Supported: openai, anthropic, groq")
        return []
    
    print(f"Using model: {model}")
    print(f"Generating {num_problems} problems...\n")
    
    # Define problem specifications
    problem_specs = [
        ("easy", "array manipulation"),
        ("easy", "string processing"),
        ("medium", "data structures"),
        ("medium", "graph algorithms"),
        ("medium", "dynamic programming"),
        ("hard", "advanced algorithms"),
        ("hard", "system design"),
    ]
    
    problems = []
    
    for i, (difficulty, topic) in enumerate(problem_specs[:num_problems], 1):
        print(f"Generating problem {i}/{num_problems}: {difficulty} - {topic}")
        
        try:
            if provider == "openai":
                problem = generate_problem_with_openai(client, difficulty, topic, model)
            elif provider == "anthropic":
                problem = generate_problem_with_anthropic(client, difficulty, topic, model)
            else:  # groq
                problem = generate_problem_with_groq(client, difficulty, topic, model)
            
            problems.append(problem)
            print(f"  ✓ Generated: {problem['id']}")
            print(f"    {problem['description'][:60]}...")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    print(f"\n✓ Successfully generated {len(problems)} problems\n")
    
    return problems


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate coding problems using OpenAI or Anthropic API"
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default="groq",
        choices=["openai", "anthropic", "groq"],
        help="API provider to use (groq is FREE!)"
    )
    parser.add_argument(
        '--model',
        type=str,
        help="Model name (default: llama-3.3-70b-versatile for Groq, gpt-4 for OpenAI, claude-3-5-sonnet for Anthropic)"
    )
    parser.add_argument(
        '--num-problems',
        type=int,
        default=5,
        help="Number of problems to generate"
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help="API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="data/custom_problems.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Generate problems
    problems = generate_problem_set(
        provider=args.provider,
        model=args.model,
        num_problems=args.num_problems,
        api_key=args.api_key
    )
    
    if not problems:
        print("No problems generated. Exiting.")
        return 1
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"problems": problems}, f, indent=2)
    
    print(f"✓ Saved {len(problems)} problems to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("GENERATED PROBLEMS SUMMARY")
    print(f"{'='*60}")
    for p in problems:
        print(f"\n{p['id']} ({p['difficulty']})")
        print(f"  Description: {p['description'][:80]}...")
        print(f"  Tests: {len(p['baseline_tests'])} test cases")
        print(f"  Solution: {len(p['reference_solution'])} chars")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print(f"  1. Review problems in {output_path}")
    print(f"  2. Run training: python run_training.py --problems-file {output_path}")
    print(f"  3. Or use automated: run_cpu_benchmark.bat")


if __name__ == "__main__":
    exit(main())
