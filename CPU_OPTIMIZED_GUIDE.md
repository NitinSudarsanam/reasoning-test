# CPU-Only Training Guide: Realistic Timings

## TL;DR: 45-90 minutes total for full benchmark

---

## Realistic CPU Timings

### Using Qwen2.5-Coder-0.5B (500M params) - RECOMMENDED

| Step | Time | What's Happening |
|------|------|------------------|
| **1. Generate Problems** | 2 min | Use pre-made (no model loading) |
| **2. Baseline Eval** | 8-12 min | Load model + 5 problems × 1 generation each |
| **3. Training** | 30-60 min | 5 problems × 5 stages × (5+5+3) steps |
| **4. Final Eval** | 8-12 min | Load trained model + 5 problems |
| **TOTAL** | **48-86 min** | ~1-1.5 hours |

### Using CodeGen-350M (350M params) - FASTER

| Step | Time | What's Happening |
|------|------|------------------|
| **1. Generate Problems** | 2 min | Use pre-made |
| **2. Baseline Eval** | 5-8 min | Smaller model = faster |
| **3. Training** | 20-40 min | Faster per step |
| **4. Final Eval** | 5-8 min | Faster inference |
| **TOTAL** | **32-58 min** | ~30-60 minutes |

---

## Time Breakdown: Training (The Slow Part)

### Per Training Step (0.5B model, CPU):
- **Load models**: 30-60 seconds (first time only)
- **Generate reasoning (5 stages)**: 15-30 seconds
- **Generate tests**: 10-20 seconds
- **Execute tests**: 1-2 seconds
- **Compute gradients & update**: 5-10 seconds
- **Total per step**: ~30-60 seconds

### Full Training Calculation:
```
5 problems × 5 stages × (5 disc + 5 gen + 3 alt) steps = 325 total steps
325 steps × 45 seconds/step = 14,625 seconds = ~24 minutes (optimistic)
325 steps × 60 seconds/step = 19,500 seconds = ~33 minutes (realistic)

Add model loading overhead: +5-10 minutes
Total: 30-45 minutes for training
```

---

## CPU-Optimized Configuration

### Minimal Training (Fastest - 20 minutes)

```bash
python run_training.py \
  --generator-model "Salesforce/codegen-350M-mono" \
  --discriminator-model "Salesforce/codegen-350M-mono" \
  --problems-file data/custom_problems.json \
  --n-discriminator-steps 2 \
  --n-generator-steps 2 \
  --k-alternating-steps 1 \
  --device cpu
```

**Time:** ~20-30 minutes
**Quality:** Still shows improvement, but less robust

### Balanced Training (Recommended - 45 minutes)

```bash
python run_training.py \
  --generator-model "Qwen/Qwen2.5-Coder-0.5B" \
  --discriminator-model "Qwen/Qwen2.5-Coder-0.5B" \
  --problems-file data/custom_problems.json \
  --n-discriminator-steps 3 \
  --n-generator-steps 3 \
  --k-alternating-steps 2 \
  --device cpu
```

**Time:** ~35-50 minutes
**Quality:** Good balance of speed and results

### Full Training (Best Results - 90 minutes)

```bash
python run_training.py \
  --generator-model "Qwen/Qwen2.5-Coder-0.5B" \
  --discriminator-model "Qwen/Qwen2.5-Coder-0.5B" \
  --problems-file data/custom_problems.json \
  --n-discriminator-steps 5 \
  --n-generator-steps 5 \
  --k-alternating-steps 3 \
  --device cpu
```

**Time:** ~60-90 minutes
**Quality:** Best results, most robust

---

## Speed Optimization Tips for CPU

### 1. Use Fewer Problems (Faster)

Train on 3 problems instead of 5:

```bash
# Edit custom_problems.json to only include first 3 problems
python run_training.py \
  --problems-file data/custom_problems_small.json \
  --n-discriminator-steps 3 \
  --n-generator-steps 3 \
  --k-alternating-steps 2
```

**Time saved:** ~40% faster (30 min instead of 50 min)

### 2. Reduce Max Tokens (Faster)

The config already has `max_new_tokens=512`. You can reduce it:

Edit `training/config.py`:
```python
max_new_tokens: int = 256  # Instead of 512
```

**Time saved:** ~30% faster per generation

### 3. Use Smaller Model (Much Faster)

```bash
# CodeGen-350M is ~40% faster than Qwen-0.5B
python run_training.py \
  --generator-model "Salesforce/codegen-350M-mono" \
  --discriminator-model "Salesforce/codegen-350M-mono"
```

**Time saved:** ~40% faster overall

### 4. Train Fewer Stages (Fastest)

Train only stages 3-5 (skip informal reasoning):

Edit the training script to start from stage 3:
```python
for stage_id in range(3, 6):  # Instead of range(1, 6)
```

**Time saved:** ~40% faster (only 3 stages instead of 5)

---

## Recommended CPU Workflow

### Quick Demo (30 minutes total)

```bash
# 1. Use pre-made problems (instant)
python generate_custom_problems.py

# 2. Baseline eval (5 min)
python benchmark_improvement.py \
  --baseline-model "Salesforce/codegen-350M-mono" \
  --problems-file data/custom_problems.json \
  --baseline-only \
  --device cpu

# 3. Fast training (20 min)
python run_training.py \
  --generator-model "Salesforce/codegen-350M-mono" \
  --discriminator-model "Salesforce/codegen-350M-mono" \
  --problems-file data/custom_problems.json \
  --n-discriminator-steps 2 \
  --n-generator-steps 2 \
  --k-alternating-steps 1 \
  --device cpu

# 4. Final eval (5 min)
python benchmark_improvement.py \
  --baseline-model "Salesforce/codegen-350M-mono" \
  --trained-checkpoint checkpoints/checkpoint_best.pt \
  --problems-file data/custom_problems.json \
  --device cpu
```

**Total: ~30 minutes**

### Balanced Quality (60 minutes total)

```bash
# 1. Use pre-made problems (instant)
python generate_custom_problems.py

# 2. Baseline eval (8 min)
python benchmark_improvement.py \
  --baseline-model "Qwen/Qwen2.5-Coder-0.5B" \
  --problems-file data/custom_problems.json \
  --baseline-only \
  --device cpu

# 3. Balanced training (45 min)
python run_training.py \
  --generator-model "Qwen/Qwen2.5-Coder-0.5B" \
  --discriminator-model "Qwen/Qwen2.5-Coder-0.5B" \
  --problems-file data/custom_problems.json \
  --n-discriminator-steps 3 \
  --n-generator-steps 3 \
  --k-alternating-steps 2 \
  --device cpu

# 4. Final eval (8 min)
python benchmark_improvement.py \
  --baseline-model "Qwen/Qwen2.5-Coder-0.5B" \
  --trained-checkpoint checkpoints/checkpoint_best.pt \
  --problems-file data/custom_problems.json \
  --device cpu
```

**Total: ~60 minutes**

---

## What to Do While Training

Training is CPU-intensive but you can still use your computer:

✅ **Can do:**
- Browse web
- Write documents
- Watch videos (may be slightly laggy)
- Listen to music

⚠️ **Avoid:**
- Running other CPU-intensive tasks
- Compiling code
- Video editing
- Gaming

💡 **Tip:** Run training overnight or during lunch break

---

## Progress Monitoring

The training script shows progress:

```
============================================================
Training Stage 1: Informal Reasoning
============================================================

Training discriminator at stage 1 for 3 steps...
Discriminator: 100%|████████████| 3/3 [01:30<00:00, 30.2s/it]

Training generator at stage 1 for 3 steps...
Generator: 100%|████████████| 3/3 [01:25<00:00, 28.5s/it]

Alternating training at stage 1 for 2 steps...
Alternating: 100%|████████████| 2/2 [00:55<00:00, 27.8s/it]

Stage 1 Summary:
  Generator Reward: 0.3500
  Discriminator Reward: 0.6200

✓ Saved checkpoint: checkpoints/checkpoint_stage_1_epoch_3.pt
```

You can see:
- Current stage (1-5)
- Progress bars with time estimates
- Rewards (improving over time)
- Checkpoints being saved

---

## Hardware Requirements (CPU)

### Minimum:
- **RAM:** 8GB (for 350M model)
- **CPU:** 4 cores
- **Disk:** 5GB free
- **Time:** 30-60 minutes

### Recommended:
- **RAM:** 16GB (for 0.5B model)
- **CPU:** 8 cores (faster)
- **Disk:** 10GB free
- **Time:** 45-90 minutes

### Your CPU Specs Matter:

| CPU | 0.5B Training Time |
|-----|-------------------|
| Intel i3 / Ryzen 3 | 90-120 min |
| Intel i5 / Ryzen 5 | 60-90 min |
| Intel i7 / Ryzen 7 | 45-60 min |
| Intel i9 / Ryzen 9 | 30-45 min |

---

## Automated CPU-Optimized Script

I'll create a script that automatically uses CPU-optimized settings:

```bash
# Windows
run_cpu_benchmark.bat

# Linux/Mac
./run_cpu_benchmark.sh
```

This will:
- Use CodeGen-350M (fastest)
- Use minimal training steps (2-2-1)
- Complete in ~30 minutes
- Still show clear improvement

---

## Bottom Line

**For CPU-only:**
- **Fastest:** 30 minutes (CodeGen-350M, minimal steps)
- **Balanced:** 60 minutes (Qwen-0.5B, moderate steps)
- **Best:** 90 minutes (Qwen-0.5B, full steps)

**My recommendation for you:**
Use the **Balanced** approach (60 minutes) with Qwen-0.5B and 3-3-2 steps. This gives you:
- ✅ Credible model size
- ✅ Clear improvement
- ✅ Reasonable time
- ✅ Good for demonstration

Start it, grab lunch, come back to results! 🍕
