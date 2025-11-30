# API Setup Guide for Problem Generation

## Why Use APIs for Problem Generation?

✅ **Better Quality:** GPT-4 and Claude generate more creative, diverse problems
✅ **No Local Model:** Don't need to download/run 7B model locally
✅ **Fast:** Generates 5 problems in ~30 seconds
✅ **Cost Effective:** ~$0.10-0.50 for 5 problems

## Option 1: OpenAI (GPT-4) - Recommended

### 1. Get API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### 2. Set Environment Variable

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-your-key-here
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=sk-your-key-here
```

**Permanent (Windows):**
```cmd
setx OPENAI_API_KEY "sk-your-key-here"
```
Then restart your terminal.

**Permanent (Linux/Mac):**
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export OPENAI_API_KEY=sk-your-key-here
```
Then run: `source ~/.bashrc`

### 3. Install Package

```bash
pip install openai
```

### 4. Generate Problems

```bash
python generate_problems_api.py --provider openai --num-problems 5
```

**Cost:** ~$0.10-0.30 for 5 problems with GPT-4

---

## Option 2: Anthropic (Claude)

### 1. Get API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Go to API Keys section
4. Create new key
5. Copy the key (starts with `sk-ant-...`)

### 2. Set Environment Variable

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Permanent:** Same as OpenAI above, but use `ANTHROPIC_API_KEY`

### 3. Install Package

```bash
pip install anthropic
```

### 4. Generate Problems

```bash
python generate_problems_api.py --provider anthropic --num-problems 5
```

**Cost:** ~$0.15-0.40 for 5 problems with Claude 3.5 Sonnet

---

## Option 3: No API (Free)

If you don't want to use APIs, use the pre-made problems:

```bash
python generate_custom_problems.py
```

This generates 5 good quality problems instantly, no API needed.

---

## Quick Test

Test if your API key works:

**OpenAI:**
```bash
python -c "from openai import OpenAI; client = OpenAI(); print('✓ OpenAI API key works!')"
```

**Anthropic:**
```bash
python -c "from anthropic import Anthropic; client = Anthropic(); print('✓ Anthropic API key works!')"
```

---

## Usage Examples

### Generate 5 problems with GPT-4
```bash
python generate_problems_api.py \
  --provider openai \
  --num-problems 5 \
  --output data/custom_problems.json
```

### Generate 10 problems with Claude
```bash
python generate_problems_api.py \
  --provider anthropic \
  --num-problems 10 \
  --output data/custom_problems.json
```

### Use specific model
```bash
# Use GPT-3.5 (cheaper, faster, slightly lower quality)
python generate_problems_api.py \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-problems 5

# Use Claude 3 Opus (most capable, more expensive)
python generate_problems_api.py \
  --provider anthropic \
  --model claude-3-opus-20240229 \
  --num-problems 5
```

---

## Cost Comparison

| Provider | Model | Cost per 5 Problems | Quality |
|----------|-------|---------------------|---------|
| **OpenAI** | GPT-4 | $0.10-0.30 | Excellent ⭐⭐⭐⭐⭐ |
| **OpenAI** | GPT-3.5-Turbo | $0.01-0.05 | Good ⭐⭐⭐⭐ |
| **Anthropic** | Claude 3.5 Sonnet | $0.15-0.40 | Excellent ⭐⭐⭐⭐⭐ |
| **Anthropic** | Claude 3 Haiku | $0.02-0.08 | Good ⭐⭐⭐⭐ |
| **Pre-made** | None | Free | Good ⭐⭐⭐⭐ |

---

## Troubleshooting

### "OPENAI_API_KEY not found"

**Solution:** Set the environment variable (see above)

**Quick fix:** Pass key directly:
```bash
python generate_problems_api.py \
  --provider openai \
  --api-key sk-your-key-here \
  --num-problems 5
```

### "openai package not installed"

**Solution:**
```bash
pip install openai
```

### "anthropic package not installed"

**Solution:**
```bash
pip install anthropic
```

### "Rate limit exceeded"

**Solution:** Wait a minute and try again, or reduce `--num-problems`

### "Invalid API key"

**Solution:** 
1. Check you copied the full key
2. Make sure no extra spaces
3. Regenerate key if needed

---

## Recommended Workflow

### For Best Results (with API):

```bash
# 1. Set API key (one time)
export OPENAI_API_KEY=sk-your-key-here

# 2. Generate high-quality problems
python generate_problems_api.py \
  --provider openai \
  --num-problems 5

# 3. Run training
python run_training.py \
  --problems-file data/custom_problems.json \
  --device cpu

# 4. Benchmark
python benchmark_improvement.py \
  --baseline-model "Qwen/Qwen2.5-Coder-0.5B" \
  --trained-checkpoint checkpoints/checkpoint_best.pt \
  --problems-file data/custom_problems.json
```

### For Free (no API):

```bash
# 1. Generate pre-made problems
python generate_custom_problems.py

# 2. Run automated benchmark
run_cpu_benchmark.bat  # Windows
./run_cpu_benchmark.sh # Linux/Mac
```

Both work great! API gives slightly better problem diversity.

---

## Security Note

⚠️ **Never commit API keys to git!**

Add to `.gitignore`:
```
.env
*.key
```

Use environment variables, not hardcoded keys.

---

## Summary

**Easiest:** Use pre-made problems (free, instant)
**Best Quality:** Use GPT-4 or Claude API (~$0.20, 30 seconds)
**Cheapest API:** Use GPT-3.5-Turbo (~$0.03, 20 seconds)

All options work well for demonstrating your adversarial RL method! 🚀
