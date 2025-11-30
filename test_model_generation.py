"""Test what the model actually generates."""

from models.generator import LLMGenerator
from reasoning.stages import get_stage
from data.problem_dataset import load_problems
from sandbox.sandbox_simple import execute_tests_simple

# Load model
print("Loading model...")
generator = LLMGenerator(model_name="Salesforce/codegen-350M-mono", device="cpu")

# Load first problem
problems = load_problems("data/custom_problems.json")
problem = problems[0]

print(f"\nProblem: {problem.id}")
print(f"Description: {problem.description[:100]}...")
print()

# Generate code using stage 5 prompt
stage_5 = get_stage(5)
print("Generating code...")
code = generator.generate_code(
    problem=problem.description,
    reasoning_chain=[],
    prompt_template=stage_5.generator_prompt_template,
    max_new_tokens=512,
    temperature=0.7
)

print("\n" + "="*60)
print("GENERATED CODE:")
print("="*60)
print(code)
print("="*60)

# Test it
print("\nTesting generated code...")
test_code = "\n".join(problem.baseline_tests)
result = execute_tests_simple(code, test_code, timeout=5)

print(f"Pass rate: {result.num_passed}/{result.num_total}")
if not result.passed:
    print(f"\nErrors:")
    for error in result.errors[:5]:
        print(f"  {error}")
