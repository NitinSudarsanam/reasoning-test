"""Debug what the model actually generates."""

from models.generator import LLMGenerator
from reasoning.stages import get_stage
from data.problem_dataset import load_problems

# Load model
print("Loading model...")
generator = LLMGenerator(model_name="Salesforce/codegen-350M-mono", device="cpu")

# Load first problem
problems = load_problems("data/custom_problems.json")
problem = problems[2]  # string_compression_with_frequency - simpler

print(f"\nProblem: {problem.id}")
print(f"Expected signature: {problem.function_signature}")
print()

# Generate code using stage 5 prompt
stage_5 = get_stage(5)

# Show the prompt
prompt = stage_5.generator_prompt_template.format(
    problem=problem.description,
    previous_stages=""
)
print("="*60)
print("PROMPT:")
print("="*60)
print(prompt)
print("="*60)

print("\nGenerating code (this takes ~30 seconds on CPU)...")
code = generator.generate_code(
    problem=problem.description,
    reasoning_chain=[],
    prompt_template=stage_5.generator_prompt_template,
    max_new_tokens=256,  # Shorter for speed
    temperature=0.7
)

print("\n" + "="*60)
print("RAW GENERATED CODE:")
print("="*60)
print(repr(code))
print("="*60)

print("\n" + "="*60)
print("FORMATTED CODE:")
print("="*60)
print(code)
print("="*60)

# Check if it matches expected format
print("\nFormat Check:")
print(f"  Has function def: {'def ' in code}")
print(f"  Has expected name: {'compress_string' in code}")
print(f"  Starts with def: {code.strip().startswith('def ')}")
print(f"  Has markdown: {'```' in code}")
