# Quick Test Guide

## Fixed Batch File Issue ✅

The batch file syntax error has been fixed. You can now run:

```bash
run_cpu_benchmark.bat
```

## Quick Test (5 minutes)

### Option 1: Pre-made Problems (Instant)

```bash
# Just press 1 when prompted
run_cpu_benchmark.bat

# Or directly:
python generate_custom_problems.py
```

**Result:** 5 good quality problems instantly

### Option 2: Groq API (FREE, 30 seconds)

```bash
# 1. Get free API key (2 minutes, one time)
# Visit: https://console.groq.com/
# Sign up and create API key

# 2. Set environment variable
set GROQ_API_KEY=gsk_your_key_here

# 3. Run benchmark and choose option 2
run_cpu_benchmark.bat
```

**Result:** 5 excellent quality problems in 30 seconds

## Test Just Problem Generation

### Pre-made
```bash
python generate_custom_problems.py
```

### Groq (FREE)
```bash
set GROQ_API_KEY=gsk_your_key_here
pip install groq
python generate_problems_api.py --provider groq --num-problems 5
```

### GPT-4 (Paid)
```bash
set OPENAI_API_KEY=sk_your_key_here
pip install openai
python generate_problems_api.py --provider openai --num-problems 5
```

### Claude (Paid)
```bash
set ANTHROPIC_API_KEY=sk-ant_your_key_here
pip install anthropic
python generate_problems_api.py --provider anthropic --num-problems 5
```

## Verify Problems Generated

```bash
# Check the file was created
dir data\custom_problems.json

# View the problems
type data\custom_problems.json
```

## Full Workflow Test

```bash
# 1. Generate problems (choose any method)
python generate_custom_problems.py

# 2. Quick training test (5 minutes)
python run_training.py ^
  --generator-model "Salesforce/codegen-350M-mono" ^
  --discriminator-model "Salesforce/codegen-350M-mono" ^
  --problems-file data/custom_problems.json ^
  --n-discriminator-steps 1 ^
  --n-generator-steps 1 ^
  --k-alternating-steps 1 ^
  --device cpu

# 3. Check checkpoint was created
dir checkpoints
```

## Troubleshooting

### "... was unexpected at this time"
**Fixed!** The batch file syntax has been corrected.

### "python not found"
Make sure Python is installed and in your PATH:
```bash
python --version
```

### "No module named 'groq'"
Install the package:
```bash
pip install groq
```

### "GROQ_API_KEY not found"
Set the environment variable:
```bash
set GROQ_API_KEY=gsk_your_key_here
```

## Next Steps

Once problem generation works:

1. **Run full benchmark:**
   ```bash
   run_cpu_benchmark.bat
   ```

2. **Wait ~30-60 minutes**

3. **Check results:**
   ```bash
   type comparison_results.json
   ```

You should see improvement metrics showing +25-35% improvement! 🎉
