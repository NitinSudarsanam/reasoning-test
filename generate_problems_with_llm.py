"""Generate custom coding problems using a larger LLM for better quality."""

import json
import argparse
from pathlib import Path
from typing import List, Dict

from models.generator import LLMGenerator


def generate_problem_with_llm(
    generator: LLMGenerator,
    difficulty: str,
    topic: str,
    avoid_common: bool = True
) -> Dict:
    """Generate a single coding problem using LLM.
    
    Args:
        generator: LLM generator instance
        difficulty: Problem difficulty (easy/medium/hard)
        topic: Problem topic/domain
        avoid_common: Whether to avoid common problem patterns
        
    Returns:
        Dictionary with problem details
    """
    
    prompt = f"""Generate a novel coding problem with the following requirements:

Difficulty: {difficulty}
Topic: {topic}
{"Avoid common LeetCode/HackerRank patterns. Create unique variations." if avoid_common else ""}

Provide:
1. A clear problem description (2-3 sentences)
2. Function signature in Python
3. 3-5 test cases covering edge cases
4. A correct reference solution in Python
5. A unique problem ID (snake_case)

Format your response as:

PROBLEM_ID: <id>

DESCRIPTION:
<description>

SIGNATURE:
<function signature>

TESTS:
<test case 1>
<test case 2>
<test case 3>

SOLUTION:
<complete working solution>

Make the problem interesting and non-trivial. Include edge cases in tests."""

    # Generate problem
    response = generator.model.generate(
        generator.tokenizer(prompt, return_tensors="pt").input_ids.to(generator.device),
        max_new_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )
    
    generated_text = generator.tokenizer.decode(response[0], skip_special_tokens=True)
    
    # Parse response (simplified - you may need more robust parsing)
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
    # Simple parsing - extract sections
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


def generate_problem_set_with_llm(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    num_problems: int = 5,
    device: str = "cpu"
) -> List[Dict]:
    """Generate a set of problems using a larger LLM.
    
    Args:
        model_name: HuggingFace model name for generation
        num_problems: Number of problems to generate
        device: Device to run on
        
    Returns:
        List of problem dictionaries
    """
    print(f"\n{'='*60}")
    print(f"GENERATING PROBLEMS WITH {model_name}")
    print(f"{'='*60}\n")
    
    generator = LLMGenerator(model_name=model_name, device=device)
    
    # Define problem specifications
    problem_specs = [
        ("easy", "array manipulation"),
        ("easy", "string processing"),
        ("medium", "data structures"),
        ("medium", "graph algorithms"),
        ("hard", "dynamic programming"),
    ]
    
    problems = []
    
    for i, (difficulty, topic) in enumerate(problem_specs[:num_problems], 1):
        print(f"Generating problem {i}/{num_problems}: {difficulty} - {topic}")
        
        try:
            problem = generate_problem_with_llm(
                generator=generator,
                difficulty=difficulty,
                topic=topic,
                avoid_common=True
            )
            problems.append(problem)
            print(f"  ✓ Generated: {problem['id']}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    print(f"\n✓ Successfully generated {len(problems)} problems\n")
    
    return problems


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate coding problems using a larger LLM"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Model to use for generation"
    )
    parser.add_argument(
        '--num-problems',
        type=int,
        default=5,
        help="Number of problems to generate"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="data/llm_generated_problems.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Generate problems
    problems = generate_problem_set_with_llm(
        model_name=args.model,
        num_problems=args.num_problems,
        device=args.device
    )
    
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
        print(f"  - {p['id']} ({p['difficulty']})")
        print(f"    {p['description'][:60]}...")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
