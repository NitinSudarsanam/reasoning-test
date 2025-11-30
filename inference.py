"""Inference script for multi-stage reasoning with trained models."""

import argparse
import json
from pathlib import Path

from models.generator import LLMGenerator
from models.discriminator import LLMDiscriminator
from sandbox.sandbox import Sandbox
from training.adversarial_trainer import AdversarialTrainer
from training.config import TrainingConfig
from reasoning.stages import get_stage


def load_trained_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained models from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load models on
        
    Returns:
        Tuple of (generator, discriminator, config)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load config if available
    config_file = checkpoint_path / "training_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = TrainingConfig(
            generator_model=config_dict['generator_model'],
            discriminator_model=config_dict['discriminator_model'],
            device=device
        )
    else:
        # Default config
        config = TrainingConfig(device=device)
    
    # Initialize models
    print(f"Loading models from {checkpoint_path}...")
    generator = LLMGenerator(config.generator_model, device)
    discriminator = LLMDiscriminator(config.discriminator_model, device)
    
    # Create trainer to use load_checkpoint method
    sandbox = Sandbox()
    trainer = AdversarialTrainer(generator, discriminator, sandbox, config)
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)
    
    return generator, discriminator, config


def generate_multi_stage_reasoning(
    generator: LLMGenerator,
    problem: str,
    config: TrainingConfig
):
    """Generate multi-stage reasoning for a problem.
    
    Args:
        generator: Trained generator model
        problem: Problem description
        config: Training configuration
        
    Returns:
        Dictionary with reasoning chain and final code
    """
    print("\n" + "="*60)
    print("MULTI-STAGE REASONING")
    print("="*60 + "\n")
    
    reasoning_chain = []
    
    # Generate each stage
    for stage_id in range(1, 6):
        stage = get_stage(stage_id)
        
        print(f"Stage {stage_id}: {stage.name}")
        print("-" * 60)
        
        if stage_id == 5:
            # Final code generation
            output = generator.generate_code(
                problem=problem,
                reasoning_chain=reasoning_chain,
                prompt_template=stage.generator_prompt_template,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p
            )
        else:
            # Intermediate reasoning stage
            output = generator.generate_stage_output(
                problem=problem,
                previous_stages=reasoning_chain,
                stage_id=stage_id,
                prompt_template=stage.generator_prompt_template,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p
            )
        
        reasoning_chain.append(output)
        print(output)
        print()
    
    return {
        'problem': problem,
        'reasoning_chain': reasoning_chain,
        'final_code': reasoning_chain[-1] if reasoning_chain else ""
    }


def generate_tests_for_code(
    discriminator: LLMDiscriminator,
    problem: str,
    code: str,
    config: TrainingConfig,
    num_tests: int = 5
):
    """Generate test cases for code.
    
    Args:
        discriminator: Trained discriminator model
        problem: Problem description
        code: Generated code
        config: Training configuration
        num_tests: Number of tests to generate
        
    Returns:
        Generated test cases
    """
    print("="*60)
    print("GENERATING TEST CASES")
    print("="*60 + "\n")
    
    stage_5 = get_stage(5)
    tests = discriminator.generate_tests(
        problem=problem,
        generator_code=code,
        num_tests=num_tests,
        prompt_template=stage_5.discriminator_prompt_template,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature
    )
    
    print(tests)
    print()
    
    return tests


def run_tests(code: str, tests: str):
    """Run tests against code.
    
    Args:
        code: Generated code
        tests: Test cases
        
    Returns:
        Execution result
    """
    print("="*60)
    print("RUNNING TESTS")
    print("="*60 + "\n")
    
    sandbox = Sandbox()
    result = sandbox.execute_tests(code, tests)
    
    print(f"Tests passed: {result.num_passed}/{result.num_total}")
    print(f"Pass rate: {result.num_passed/result.num_total*100:.1f}%")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    print()
    
    return result


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained multi-stage reasoning models"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="checkpoints/best",
        help="Path to checkpoint directory (default: checkpoints/best)"
    )
    parser.add_argument(
        '--problem',
        type=str,
        help="Problem description (if not provided, will use example)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on"
    )
    parser.add_argument(
        '--generate-tests',
        action='store_true',
        help="Generate and run tests for the code"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Output file to save results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Default problem if not provided
    if not args.problem:
        args.problem = """Given an array of integers nums and an integer target, 
return indices of the two numbers such that they add up to target. 
You may assume that each input would have exactly one solution, 
and you may not use the same element twice."""
    
    print("\n" + "="*60)
    print("MULTI-STAGE REASONING INFERENCE")
    print("="*60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"\nProblem: {args.problem}")
    
    try:
        # Load trained models
        generator, discriminator, config = load_trained_model(
            args.checkpoint,
            args.device
        )
        
        # Generate multi-stage reasoning
        result = generate_multi_stage_reasoning(generator, args.problem, config)
        
        # Optionally generate and run tests
        if args.generate_tests:
            tests = generate_tests_for_code(
                discriminator,
                args.problem,
                result['final_code'],
                config
            )
            
            test_result = run_tests(result['final_code'], tests)
            result['tests'] = tests
            result['test_result'] = {
                'passed': test_result.passed,
                'num_passed': test_result.num_passed,
                'num_total': test_result.num_total,
                'errors': test_result.errors
            }
        
        # Save results if output file specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Results saved to {output_path}")
        
        print("\n" + "="*60)
        print("INFERENCE COMPLETE")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
