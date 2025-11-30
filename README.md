# Adversarial RL Multi-Stage Reasoning System

A GAN-inspired adversarial reinforcement learning system that trains a generator LLM and discriminator LLM to solve coding problems through multi-stage reasoning.

## Overview

This system implements adversarial training where:
- **Generator LLM** produces multi-stage reasoning (informal → structured → pseudocode → constraints → code)
- **Discriminator LLM** generates adversarial test cases to challenge the generator
- Both models compete in a zero-sum game using reinforcement learning (PPO)
- Training proceeds stage-by-stage from bottom to top

### Key Features

- **Multi-Stage Reasoning Pipeline**: 5 progressive stages from informal reasoning to executable code
- **Adversarial Competition**: Generator maximizes test pass rate, discriminator minimizes it
- **Test Validation**: Discriminator tests validated against ground truth solutions
- **Secure Sandbox**: Safe code execution with timeout and error handling
- **CPU/GPU Support**: Works on CPU for development, GPU for faster training
- **Free Cloud GPU**: Compatible with Google Colab, Kaggle, Lightning AI

## Architecture

```
Problem → Generator → [Stage 1-5 Reasoning] → Code
                                                  ↓
                                            Sandbox Tests
                                                  ↑
Problem → Discriminator → Adversarial Tests ─────┘
                                                  ↓
                                          Reward Computation
                                                  ↓
                                            PPO Training
```

### Training Process

For each stage (1-5):
1. **N Discriminator Steps**: Train discriminator with frozen generator
2. **N Generator Steps**: Train generator with frozen discriminator  
3. **K Alternating Steps**: Rapid competition between both models

### Reward Functions

**Generator Reward**:
```
reward = test_pass_rate
```

**Discriminator Reward**:
```
reward = (1.0 - generator_pass_rate) * test_validity_score
```

Where `test_validity_score` is the percentage of discriminator tests that pass against ground truth.

## Installation

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/adversarial-rl-reasoning.git
cd adversarial-rl-reasoning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

```python
# Upload your code or clone from GitHub
!git clone https://github.com/yourusername/adversarial-rl-reasoning.git
%cd adversarial-rl-reasoning

# Install dependencies
!pip install -r requirements.txt

# Run training
!python run_training.py --device cuda
```

### Kaggle Setup

1. Create new notebook
2. Add dataset or upload code
3. Enable GPU in settings
4. Run:
```python
!pip install -r requirements.txt
!python run_training.py --device cuda
```

## Usage

### Basic Training

```bash
python run_training.py
```

This will:
- Train all 5 stages sequentially
- Save checkpoints after each stage
- Track and save best performing model
- Save results to `output/training_results.json`

### Checkpoint Management

The system automatically saves checkpoints during training:

```
checkpoints/
├── checkpoint_stage_1_epoch_10.pt     # Stage 1 checkpoint
├── checkpoint_stage_1_epoch_10.json   # Stage 1 metadata
├── checkpoint_stage_2_epoch_10.pt     # Stage 2 checkpoint
├── checkpoint_stage_2_epoch_10.json   # Stage 2 metadata
├── ...
├── checkpoint_best.pt                 # Best performing model
└── checkpoint_best.json               # Best model metadata
```

**Checkpoint Contents**:
- Generator and discriminator model weights
- Training metrics (rewards, test validity)
- Stage and epoch information
- Training configuration
- Timestamp

**Best Checkpoint Tracking**:
The system automatically tracks the best checkpoint based on a combined score:
```
score = 0.7 * generator_reward + 0.3 * test_validity
```

### Resume Training

Resume training from the latest checkpoint:

```bash
python run_training.py --resume
```

Resume from a specific checkpoint:

```bash
python run_training.py --resume-from checkpoints/checkpoint_stage_3_epoch_10.pt
```

### Inference with Trained Models

After training, use your trained models to solve new problems:

**Single Problem Inference**:
```bash
# Basic inference
python run_inference.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --problem "Write a function that finds the longest palindromic substring"

# With function signature
python run_inference.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --problem "Find longest palindrome" \
  --signature "def longest_palindrome(s: str) -> str:"

# Execute tests
python run_inference.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --problem "Return sum of two numbers" \
  --tests "assert add(2,3)==5;assert add(0,0)==0" \
  --execute-tests

# Save results
python run_inference.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --problem "Your problem" \
  --output results.json
```

**Batch Inference**:
```bash
# Process multiple problems from file
python run_inference.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --batch \
  --problems-file data/example_problems.json \
  --execute-tests \
  --output batch_results.json
```

**Inference Options**:
- `--checkpoint`: Path to checkpoint file (required)
- `--problem`: Problem description for single problem mode
- `--problem-file`: Path to file containing problem description
- `--signature`: Function signature
- `--tests`: Test cases separated by semicolons
- `--batch`: Enable batch mode for multiple problems
- `--problems-file`: JSON file with problems (for batch mode)
- `--execute-tests`: Execute generated code against tests
- `--device`: Device to use (cpu/cuda)
- `--max-tokens`: Max tokens per stage (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling (default: 0.9)
- `--output`: Save results to JSON file
- `--no-reasoning`: Hide reasoning chain, show only code

### Custom Configuration

```bash
python run_training.py \
  --generator-model "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  --discriminator-model "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  --device cuda \
  --n-discriminator-steps 10 \
  --n-generator-steps 10 \
  --k-alternating-steps 5 \
  --learning-rate 1e-5 \
  --problems-file data/example_problems.json \
  --output-dir output
```

### Arguments

- `--generator-model`: HuggingFace model for generator (default: Qwen2.5-Coder-1.5B-Instruct)
- `--discriminator-model`: HuggingFace model for discriminator (default: Qwen2.5-Coder-1.5B-Instruct)
- `--device`: Device to use (cpu or cuda)
- `--n-discriminator-steps`: Training steps for discriminator per stage (default: 2)
- `--n-generator-steps`: Training steps for generator per stage (default: 2)
- `--k-alternating-steps`: Alternating competition steps per stage (default: 2)
- `--learning-rate`: Learning rate (default: 1e-5)
- `--problems-file`: Path to problems JSON file
- `--output-dir`: Directory to save results

## Project Structure

```
adversarial-rl-reasoning/
├── models/
│   ├── generator.py          # Generator LLM wrapper
│   └── discriminator.py      # Discriminator LLM wrapper
├── reasoning/
│   └── stages.py             # 5-stage reasoning definitions
├── sandbox/
│   └── sandbox.py            # Secure code execution
├── training/
│   ├── adversarial_trainer.py    # Main training orchestration
│   ├── checkpoint_manager.py     # Checkpoint saving/loading
│   ├── rl_loop.py                # PPO training loop
│   ├── reward.py                 # Reward computation
│   ├── config.py                 # Training configuration
│   └── multi_attempt.py          # Multi-attempt support
├── inference/
│   └── inference_engine.py   # Inference with trained models
├── evaluation/
│   └── metrics.py            # Evaluation metrics
├── data/
│   ├── problem_dataset.py    # Problem loading
│   └── example_problems.json # Sample problems
├── run_training.py           # Training entry point
├── run_inference.py          # Inference entry point
├── test_checkpoint.py        # Checkpoint tests
├── test_inference.py         # Inference tests
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Reasoning Stages

### Stage 1: Informal Reasoning
High-level intuitive understanding of the problem

### Stage 2: Structured Reasoning
Organized breakdown with clear steps and edge cases

### Stage 3: Pseudocode
Algorithm expressed in pseudocode notation

### Stage 4: Constraints and Invariants
Explicit constraints, invariants, and complexity analysis

### Stage 5: Executable Code
Final Python implementation

## Extending the System

### Adding New Problems

Edit `data/example_problems.json`:

```json
{
  "problems": [
    {
      "id": "your_problem",
      "description": "Problem description",
      "function_signature": "def your_function(args):",
      "baseline_tests": ["assert your_function(1) == 2"],
      "reference_solution": "def your_function(x):\n    return x + 1",
      "difficulty": "easy",
      "tags": ["math"]
    }
  ]
}
```

### Adding New Reasoning Stages

Edit `reasoning/stages.py` and add a new `ReasoningStage` to the `REASONING_STAGES` list.

### Using Different Models

Any HuggingFace causal LM can be used:

```bash
python run_training.py \
  --generator-model "codellama/CodeLlama-7b-hf" \
  --discriminator-model "codellama/CodeLlama-7b-hf"
```

## Transferring to Cloud GPU

### Method 1: GitHub (Recommended)

```bash
# On your computer
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main

# In Colab/Kaggle
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo
!pip install -r requirements.txt
!python run_training.py --device cuda
```

### Method 2: Google Drive (Colab)

```python
# Upload to Google Drive, then in Colab:
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/your-project
!pip install -r requirements.txt
!python run_training.py --device cuda
```

### Method 3: Direct Upload

```python
# In Colab
from google.colab import files
uploaded = files.upload()  # Upload your zip file
!unzip your-project.zip
%cd your-project
!pip install -r requirements.txt
!python run_training.py --device cuda
```

## Free GPU Options

### Google Colab
- **Free Tier**: T4 GPU (12GB), 12GB RAM
- **Session Limit**: ~12 hours
- **Best For**: Quick experiments and testing

### Kaggle Notebooks
- **Free Tier**: P100/T4 GPU (16GB), 30GB RAM
- **Quota**: 30 hours/week
- **Best For**: Longer training runs

### Lightning AI Studio
- **Free Tier**: Limited GPU hours
- **Best For**: Development and testing

## Performance

### Expected Training Time

- **CPU (16GB RAM)**: ~2-4 hours for 1 iteration (5 stages, 2 steps each)
- **GPU (T4)**: ~30-60 minutes for 1 iteration
- **GPU (A100)**: ~15-30 minutes for 1 iteration

### Memory Requirements

- **Minimum**: 16GB RAM (CPU mode)
- **Recommended**: 24GB GPU memory for larger models
- **1.5B Model**: ~6GB GPU memory

## Troubleshooting

### Out of Memory

```bash
# Use smaller batch size or fewer steps
python run_training.py --n-discriminator-steps 1 --n-generator-steps 1
```

### Slow Training

```bash
# Use GPU if available
python run_training.py --device cuda

# Or reduce model size
python run_training.py --generator-model "Qwen/Qwen2.5-Coder-0.5B"
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Checkpoint Errors

**Corrupted Checkpoint**:
```bash
# Use a different checkpoint
python run_training.py --resume-from checkpoints/checkpoint_stage_2_epoch_10.pt

# Or start fresh
python run_training.py
```

**Missing Checkpoint**:
```bash
# List available checkpoints
ls checkpoints/

# Resume from specific stage
python run_training.py --resume-from checkpoints/checkpoint_stage_3_epoch_10.pt
```

**Checkpoint Load Failure**:
- Ensure checkpoint was saved with same model architecture
- Check device compatibility (CPU vs CUDA)
- Verify checkpoint file is not corrupted

### Inference Errors

**Model Load Failure**:
```bash
# Specify model names explicitly
python run_inference.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --problem "Your problem"
```

**Generation Timeout**:
```bash
# Increase max tokens or adjust temperature
python run_inference.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --problem "Your problem" \
  --max-tokens 1024 \
  --temperature 0.5
```

**Test Execution Failure**:
- Check test case syntax
- Verify generated code is valid Python
- Review error messages in execution result

## Citation

If you use this code in your research, please cite:

```bibtex
@software{adversarial_rl_reasoning,
  title={Adversarial RL Multi-Stage Reasoning System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/adversarial-rl-reasoning}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or pull request.

## Acknowledgments

- Built with HuggingFace Transformers
- Inspired by GAN training dynamics
- Uses PPO for reinforcement learning
