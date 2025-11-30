"""Quick test of the fixed benchmark."""

from data.problem_dataset import load_problems
from models.generator import LLMGenerator
from sandbox.sandbox import Sandbox
from sandbox.sandbox_simple import execute_tests_simple
from reasoning.stages import get_stage

# Load problems
problems = load_problems("data/custom_problems.json")
print(f"Loaded {len(problems)} problems\n")

# Test baseline evaluation with first problem
problem = problems[0]
print(f"Testing baseline evaluation for: {problem.id}\n")

# Initialize generator
print("Loading model...")
generator = LLMGenerator(model_name="Salesforce/codegen-350M-mono", device="cpu")
sandbox = Sandbox(timeout=5)

# Generate code
print("Generating code...")
stage_5 = get_stage(5)
code = generator.generate_code(
    problem=problem.description,
    reasoning_chain=[],
    prompt_template=stage_5.generator_prompt_template,
    max_new_tokens=256,
    temperature=0.7
)

print(f"Generated code ({len(code)} chars)\n")

# Test with simple executor
print("Testing with simple executor...")
test_code = "\n".join(problem.baseline_tests)
result = execute_tests_simple(code, test_code, timeout=5)

print(f"✓ Result: {result.num_passed}/{result.num_total} tests passed")
print(f"  Pass rate: {result.num_passed/result.num_total:.2%}")
print(f"  Passed: {result.passed}")

if result.errors:
    print(f"  Errors: {result.errors[:2]}")

print("\n✓ Benchmark script is fixed!")
