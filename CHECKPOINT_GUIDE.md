# Checkpoint and Inference Guide

## Checkpoint System

The system automatically saves checkpoints during training to preserve your progress and enable inference.

## Checkpoint Structure

After training, you'll have:

```
checkpoints/
├── stage_1/
│   ├── generator.pt          # Generator after stage 1
│   ├── discriminator.pt      # Discriminator after stage 1
│   └── metrics.json          # Stage 1 metrics
├── stage_2/
│   ├── generator.pt
│   ├── discriminator.pt
│   └── metrics.json
├── stage_3/
│   ├── generator.pt
│   ├── discriminator.pt
│   └── metrics.json
├── stage_4/
│   ├── generator.pt
│   ├── discriminator.pt
│   └── metrics.json
├── stage_5/
│   ├── generator.pt
│   ├── discriminator.pt
│   └── metrics.json
├── best/
│   ├── generator.pt          # Best generator (highest reward)
│   ├── discriminator.pt      # Best discriminator
│   └── metrics.json          # Best metrics
└── final/
    ├── generator.pt          # Final generator after all training
    ├── discriminator.pt      # Final discriminator
    └── training_config.json  # Training configuration
```

## When Checkpoints Are Saved

### During Training

1. **After each stage**: Checkpoint saved to `checkpoints/stage_N/`
2. **When reward improves**: Best checkpoint updated in `checkpoints/best/`
3. **After all training**: Final checkpoint saved to `checkpoints/final/`

### What's Saved

Each checkpoint contains:
- **Model weights**: The trained neural network parameters
- **Optimizer state**: For resuming training
- **Stage ID**: Which stage this checkpoint is from
- **Metrics**: Performance metrics (if available)

## Using Checkpoints for Inference

### Basic Inference

```bash
# Use best checkpoint (recommended)
python inference.py --checkpoint checkpoints/best --problem "Implement a function that reverses a string"
```

Output:
```
Stage 1: Informal Reasoning
------------------------------------------------------------
We need to reverse a string. We can use two pointers...

Stage 2: Structured Reasoning
------------------------------------------------------------
1. Problem: Reverse string in-place
2. Approach: Two pointers from ends...

Stage 3: Pseudocode
------------------------------------------------------------
left = 0, right = len(s) - 1
while left < right:
    swap s[left] and s[right]...

Stage 4: Constraints and Invariants
------------------------------------------------------------
Time: O(n), Space: O(1)...

Stage 5: Executable Code
------------------------------------------------------------
def reverse_string(s):
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
```

### Generate and Test Code

```bash
python inference.py \
  --checkpoint checkpoints/best \
  --problem "Implement two_sum" \
  --generate-tests
```

This will:
1. Generate multi-stage reasoning
2. Generate test cases with discriminator
3. Run tests against generated code
4. Show pass/fail results

### Save Results

```bash
python inference.py \
  --checkpoint checkpoints/best \
  --problem "Your problem" \
  --output results.json
```

Creates `results.json`:
```json
{
  "problem": "Your problem",
  "reasoning_chain": [
    "Stage 1 output...",
    "Stage 2 output...",
    "Stage 3 output...",
    "Stage 4 output...",
    "Stage 5 code..."
  ],
  "final_code": "def solution(): ..."
}
```

### Use Different Checkpoints

```bash
# Use stage 3 checkpoint
python inference.py --checkpoint checkpoints/stage_3 --problem "Your problem"

# Use final checkpoint
python inference.py --checkpoint checkpoints/final --problem "Your problem"

# Use specific stage
python inference.py --checkpoint checkpoints/stage_2 --problem "Your problem"
```

## Inference Options

### Command-Line Arguments

```bash
python inference.py \
  --checkpoint checkpoints/best \      # Checkpoint to load
  --problem "Problem description" \    # Problem to solve
  --device cuda \                      # Use GPU
  --generate-tests \                   # Generate and run tests
  --output results.json                # Save to file
```

### Arguments

- `--checkpoint`: Path to checkpoint directory (default: `checkpoints/best`)
- `--problem`: Problem description (default: example problem)
- `--device`: Device to use (`cpu` or `cuda`)
- `--generate-tests`: Generate tests and run them
- `--output`: Save results to JSON file

## Comparing Checkpoints

### Check Metrics

```bash
# View stage 1 metrics
cat checkpoints/stage_1/metrics.json

# View best metrics
cat checkpoints/best/metrics.json

# View final metrics
cat checkpoints/final/training_config.json
```

### Compare Performance

```python
import json

# Load metrics from different stages
with open('checkpoints/stage_1/metrics.json') as f:
    stage_1 = json.load(f)

with open('checkpoints/stage_5/metrics.json') as f:
    stage_5 = json.load(f)

print(f"Stage 1 reward: {stage_1['generator']['avg_reward']}")
print(f"Stage 5 reward: {stage_5['generator']['avg_reward']}")
```

## Resuming Training

To resume training from a checkpoint:

```python
from training.adversarial_trainer import AdversarialTrainer
from training.config import TrainingConfig

# Create trainer
trainer = AdversarialTrainer(generator, discriminator, sandbox, config)

# Load checkpoint
trainer.load_checkpoint('checkpoints/stage_3', load_optimizer=True)

# Continue training from stage 4
for stage_id in range(4, 6):
    trainer.train_stage(stage_id, problems)
```

## Checkpoint Management

### Disk Space

Each checkpoint is ~12GB (6GB generator + 6GB discriminator)

Total space for all checkpoints:
- 5 stage checkpoints: ~60GB
- 1 best checkpoint: ~12GB
- 1 final checkpoint: ~12GB
- **Total: ~84GB**

### Cleanup

If disk space is limited, keep only what you need:

```bash
# Keep only best and final
rm -rf checkpoints/stage_*

# Keep only final
rm -rf checkpoints/stage_* checkpoints/best

# Keep only best
rm -rf checkpoints/stage_* checkpoints/final
```

### Backup

To backup your trained models:

```bash
# Compress checkpoints
tar -czf checkpoints_backup.tar.gz checkpoints/

# Upload to cloud storage
# Google Drive, Dropbox, etc.
```

## Using Checkpoints in Code

### Load and Use Generator

```python
from models.generator import LLMGenerator
from training.config import TrainingConfig
import torch

# Initialize generator
config = TrainingConfig(device="cpu")
generator = LLMGenerator(config.generator_model, config.device)

# Load checkpoint
checkpoint = torch.load('checkpoints/best/generator.pt', map_location='cpu')
generator.model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
from reasoning.stages import get_stage

stage_1 = get_stage(1)
output = generator.generate_stage_output(
    problem="Your problem",
    previous_stages=[],
    stage_id=1,
    prompt_template=stage_1.generator_prompt_template
)

print(output)
```

## Best Practices

1. **Use `best` checkpoint for inference** - highest quality
2. **Use `final` checkpoint for continued training** - most recent
3. **Save checkpoints to cloud** - don't lose your work
4. **Test inference before long training** - verify it works
5. **Monitor disk space** - checkpoints are large

## Troubleshooting

### Checkpoint not found
```bash
# Check if checkpoint exists
ls checkpoints/best/

# If missing, use final or stage checkpoint
python inference.py --checkpoint checkpoints/final
```

### Out of memory during inference
```bash
# Use CPU
python inference.py --checkpoint checkpoints/best --device cpu
```

### Checkpoint from different model
```bash
# Error: Model architecture mismatch
# Solution: Use same model architecture as training
```

---

Now you can train models and use them for inference! 🎉
