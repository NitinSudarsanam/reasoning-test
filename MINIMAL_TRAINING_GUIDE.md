# Minimal Training Guide

## Quick Start

Run the minimal training to see what's happening:

```batch
run_minimal.bat
```

## What It Does

1. **Loads 2 problems** (instead of 5) - faster
2. **Runs 1 training iteration** (instead of multiple) - minimal
3. **Saves all generated code** to `training_output/TIMESTAMP/`
4. **Takes 5-10 minutes** (instead of 30+)

## Output Structure

```
training_output/
└── 20241130_143022/          # Timestamp of run
    ├── baseline/              # Code BEFORE training
    │   ├── string_compression_with_frequency.py
    │   └── matrix_spiral_sum.py
    ├── trained/               # Code AFTER training
    │   ├── string_compression_with_frequency.py
    │   └── matrix_spiral_sum.py
    ├── checkpoints/           # Model checkpoints
    │   └── checkpoint_best.pt
    └── summary.json           # Training metrics
```

## What to Check

### 1. Baseline Code
Look at `baseline/*.py` to see what the model generates initially.

**Expected:** Buggy or incomplete code (0-20% accuracy)

### 2. Trained Code
Look at `trained/*.py` to see what the model generates after training.

**Expected:** Better code (20-40% accuracy)

### 3. Summary
Check `summary.json` for metrics:
```json
{
  "config": {
    "model": "Salesforce/codegen-350M-mono",
    "problems": ["string_compression_with_frequency", "matrix_spiral_sum"]
  },
  "metrics": {
    "stage_1": {...},
    "stage_5": {...}
  }
}
```

## Configuration

Edit `run_minimal_training.py` to change:

```python
# Use different model
config = TrainingConfig(
    generator_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",  # Larger model
    ...
)

# Use more problems
problems = all_problems[:5]  # Use 5 instead of 2

# More training iterations
config = TrainingConfig(
    k_alternating_steps=3,  # 3 iterations instead of 1
    ...
)
```

## Troubleshooting

**If it's too slow:**
- Use fewer problems (1 instead of 2)
- Reduce `max_new_tokens` to 128

**If code quality is bad:**
- Use a larger model (Qwen2.5-Coder-1.5B)
- Increase training iterations

**If you want more detail:**
- Check the console output during training
- Look at individual `.py` files in the output folders

## Next Steps

Once you verify it works:
1. Run the full benchmark: `run_cpu_benchmark_auto.bat`
2. Use more problems and iterations
3. Try different models
