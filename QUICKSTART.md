# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Note**: If you encounter NumPy version issues, use:
```bash
pip install "numpy<2.0.0"
```

## Running Training

### Basic Training (CPU)

```bash
python run_training.py
```

This will run a minimal training iteration with:
- 2 discriminator steps per stage
- 2 generator steps per stage
- 2 alternating steps per stage
- All 5 reasoning stages

Expected time: ~2-4 hours on CPU

### Quick Test (Minimal Steps)

```bash
python run_training.py \
  --n-discriminator-steps 1 \
  --n-generator-steps 1 \
  --k-alternating-steps 1
```

Expected time: ~30-60 minutes on CPU

### GPU Training (if available)

```bash
python run_training.py --device cuda
```

Expected time: ~30-60 minutes on GPU

## Validation

Before running training, validate the project structure:

```bash
python validate_structure.py
```

This checks:
- All required files exist
- Python syntax is valid
- JSON files are well-formed

## Running on Google Colab

1. Upload your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

2. In a new Colab notebook:
```python
# Clone repository
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo

# Install dependencies
!pip install -r requirements.txt

# Run training with GPU
!python run_training.py --device cuda
```

## Expected Output

Training will show progress for each stage:

```
============================================================
Training Stage 1: Informal Reasoning
============================================================

Training discriminator at stage 1 for 2 steps...
Discriminator: 100%|██████████| 2/2 [00:30<00:00, 15.2s/it]

Training generator at stage 1 for 2 steps...
Generator: 100%|██████████| 2/2 [00:25<00:00, 12.8s/it]

Alternating training at stage 1 for 2 steps...
Alternating: 100%|██████████| 2/2 [00:20<00:00, 10.1s/it]

Stage 1 Summary:
  Generator Reward: 0.4500
  Discriminator Reward: 0.5200
```

Results are saved to `output/training_results.json`

## Troubleshooting

### Out of Memory

Reduce the number of steps:
```bash
python run_training.py --n-discriminator-steps 1 --n-generator-steps 1
```

### Slow Training

Use GPU if available, or reduce model size:
```bash
python run_training.py --device cuda
```

### Import Errors

Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

After successful training:

1. Check results in `output/training_results.json`
2. Modify problems in `data/example_problems.json`
3. Adjust training parameters in `run_training.py`
4. Extend reasoning stages in `reasoning/stages.py`

## Getting Help

- Check README.md for detailed documentation
- Review the design document in `.kiro/specs/adversarial-rl-reasoning/design.md`
- Open an issue on GitHub
