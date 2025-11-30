"""Benchmark script to measure improvement from adversarial RL training."""

import json
import argparse
from pathlib import Path
from typing import List, Dict

from models.generator import LLMGenerator
from sandbox.sandbox import Sandbox
from sandbox.sandbox_simple import execute_tests_simple
from data.problem_dataset import load_problems, Problem
from reasoning.stages import get_stage
from inference.inference_engine import InferenceEngine


def evaluate_baseline_model(
    model_name: str,
    problems: List[Problem],
    device: str = "cpu"
) -> Dict:
    """Evaluate untrained baseline model.
    
    Args:
        model_name: HuggingFace model name
        problems: List of problems to evaluate
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING BASELINE MODEL: {model_name}")
    print(f"{'='*60}\n")
    
    generator = LLMGenerator(model_name=model_name, device=device)
    sandbox = Sandbox(timeout=5)
    
    results = []
    
    for i, problem in enumerate(problems, 1):
        print(f"Problem {i}/{len(problems)}: {problem.id}")
        
        # Generate code directly (no multi-stage reasoning)
        stage_5 = get_stage(5)
        code = generator.generate_code(
            problem=problem.description,
            reasoning_chain=[],
            prompt_template=stage_5.generator_prompt_template,
            max_new_tokens=512,
            temperature=0.7,
            function_signature=problem.function_signature
        )
        
        # Test against baseline tests using simple executor
        test_code = "\n".join(problem.baseline_tests)
        execution_result = execute_tests_simple(code, test_code, timeout=sandbox.timeout)
        
        passed = execution_result.passed
        pass_rate = execution_result.num_passed / execution_result.num_total if execution_result.num_total > 0 else 0
        
        results.append({
            'problem_id': problem.id,
            'passed': passed,
            'pass_rate': pass_rate,
            'num_passed': execution_result.num_passed,
            'num_total': execution_result.num_total,
            'errors': execution_result.errors[:3]  # First 3 errors
        })
        
        status = "✓" if passed else "✗"
        print(f"  {status} Pass rate: {pass_rate:.2%} ({execution_result.num_passed}/{execution_result.num_total})")
    
    # Compute aggregate metrics
    total_passed = sum(1 for r in results if r['passed'])
    avg_pass_rate = sum(r['pass_rate'] for r in results) / len(results)
    
    metrics = {
        'model': model_name,
        'total_problems': len(problems),
        'problems_passed': total_passed,
        'problem_pass_rate': total_passed / len(problems),
        'avg_test_pass_rate': avg_pass_rate,
        'results': results
    }
    
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Problems Passed: {total_passed}/{len(problems)} ({metrics['problem_pass_rate']:.2%})")
    print(f"Average Test Pass Rate: {avg_pass_rate:.2%}")
    print(f"{'='*60}\n")
    
    return metrics


def evaluate_trained_model(
    checkpoint_path: str,
    problems: List[Problem],
    device: str = "cpu"
) -> Dict:
    """Evaluate trained model with multi-stage reasoning.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        problems: List of problems to evaluate
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING TRAINED MODEL: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Load trained model
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    results = []
    
    for i, problem in enumerate(problems, 1):
        print(f"Problem {i}/{len(problems)}: {problem.id}")
        
        # Use multi-stage reasoning
        result = engine.solve_problem(
            problem_description=problem.description,
            function_signature=problem.function_signature,
            execute_tests=False,  # We'll use simple executor below
            test_cases=problem.baseline_tests,
            max_new_tokens=512,
            temperature=0.7
        )
        
        # Execute tests with simple executor for accurate counts
        test_code = "\n".join(problem.baseline_tests)
        execution_result = execute_tests_simple(result.generated_code, test_code, timeout=5)
        
        passed = execution_result.passed
        pass_rate = execution_result.num_passed / execution_result.num_total if execution_result.num_total > 0 else 0
        
        results.append({
            'problem_id': problem.id,
            'passed': passed,
            'pass_rate': pass_rate,
            'num_passed': execution_result.num_passed,
            'num_total': execution_result.num_total,
            'inference_time': result.inference_time,
            'errors': execution_result.errors[:3]
        })
        
        status = "✓" if passed else "✗"
        print(f"  {status} Pass rate: {pass_rate:.2%} ({results[-1]['num_passed']}/{results[-1]['num_total']})")
        print(f"  Time: {result.inference_time:.2f}s")
    
    # Compute aggregate metrics
    total_passed = sum(1 for r in results if r['passed'])
    avg_pass_rate = sum(r['pass_rate'] for r in results) / len(results)
    avg_time = sum(r['inference_time'] for r in results) / len(results)
    
    metrics = {
        'checkpoint': checkpoint_path,
        'total_problems': len(problems),
        'problems_passed': total_passed,
        'problem_pass_rate': total_passed / len(problems),
        'avg_test_pass_rate': avg_pass_rate,
        'avg_inference_time': avg_time,
        'results': results
    }
    
    print(f"\n{'='*60}")
    print("TRAINED MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Problems Passed: {total_passed}/{len(problems)} ({metrics['problem_pass_rate']:.2%})")
    print(f"Average Test Pass Rate: {avg_pass_rate:.2%}")
    print(f"Average Inference Time: {avg_time:.2f}s")
    print(f"{'='*60}\n")
    
    return metrics


def compare_results(baseline_metrics: Dict, trained_metrics: Dict):
    """Compare baseline and trained model results."""
    print(f"\n{'='*60}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*60}\n")
    
    baseline_problem_pass = baseline_metrics['problem_pass_rate']
    trained_problem_pass = trained_metrics['problem_pass_rate']
    problem_improvement = trained_problem_pass - baseline_problem_pass
    
    baseline_test_pass = baseline_metrics['avg_test_pass_rate']
    trained_test_pass = trained_metrics['avg_test_pass_rate']
    test_improvement = trained_test_pass - baseline_test_pass
    
    print(f"Problem Pass Rate:")
    print(f"  Baseline: {baseline_problem_pass:.2%}")
    print(f"  Trained:  {trained_problem_pass:.2%}")
    print(f"  Improvement: {problem_improvement:+.2%}")
    print()
    
    print(f"Average Test Pass Rate:")
    print(f"  Baseline: {baseline_test_pass:.2%}")
    print(f"  Trained:  {trained_test_pass:.2%}")
    print(f"  Improvement: {test_improvement:+.2%}")
    print()
    
    # Per-problem comparison
    print("Per-Problem Improvements:")
    for baseline_result, trained_result in zip(
        baseline_metrics['results'], 
        trained_metrics['results']
    ):
        problem_id = baseline_result['problem_id']
        baseline_rate = baseline_result['pass_rate']
        trained_rate = trained_result['pass_rate']
        improvement = trained_rate - baseline_rate
        
        status = "✓" if improvement > 0 else ("=" if improvement == 0 else "✗")
        print(f"  {status} {problem_id}: {baseline_rate:.2%} → {trained_rate:.2%} ({improvement:+.2%})")
    
    print(f"\n{'='*60}\n")
    
    return {
        'problem_pass_improvement': problem_improvement,
        'test_pass_improvement': test_improvement,
        'baseline': baseline_metrics,
        'trained': trained_metrics
    }


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(
        description="Benchmark improvement from adversarial RL training"
    )
    
    parser.add_argument(
        '--baseline-model',
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Baseline model to evaluate"
    )
    parser.add_argument(
        '--trained-checkpoint',
        type=str,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        '--problems-file',
        type=str,
        default="data/custom_problems.json",
        help="Path to problems JSON file"
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
        default="benchmark_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help="Only evaluate baseline model"
    )
    
    args = parser.parse_args()
    
    # Load problems
    print(f"Loading problems from {args.problems_file}...")
    problems = load_problems(args.problems_file)
    print(f"Loaded {len(problems)} problems\n")
    
    # Evaluate baseline
    baseline_metrics = evaluate_baseline_model(
        model_name=args.baseline_model,
        problems=problems,
        device=args.device
    )
    
    if args.baseline_only:
        # Save baseline results
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'baseline': baseline_metrics}, f, indent=2)
        print(f"Baseline results saved to {output_path}")
        return
    
    if not args.trained_checkpoint:
        print("Error: --trained-checkpoint required (or use --baseline-only)")
        return 1
    
    # Evaluate trained model
    trained_metrics = evaluate_trained_model(
        checkpoint_path=args.trained_checkpoint,
        problems=problems,
        device=args.device
    )
    
    # Compare results
    comparison = compare_results(baseline_metrics, trained_metrics)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    exit(main())
