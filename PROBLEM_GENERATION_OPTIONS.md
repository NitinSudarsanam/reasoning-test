# Problem Generation Options

## Three Ways to Generate Problems

### Option 1: Pre-Made Problems (Fastest, Free) ⚡

**Best for:** Quick start, no setup needed

```bash
python generate_custom_problems.py
```

**Pros:**
- ✅ Instant (no API calls)
- ✅ Free
- ✅ Good quality
- ✅ No dependencies

**Cons:**
- ⚠️ Fixed set of 5 problems
- ⚠️ Less variety

**Output:** 5 hand-crafted problems in `data/custom_problems.json`

---

### Option 2: GPT-4 API (Best Quality) 🏆

**Best for:** Highest quality, most creative problems

**Setup:**
```bash
# 1. Install package
pip install openai

# 2. Set API key
export OPENAI_API_KEY=sk-your-key-here  # Linux/Mac
set OPENAI_API_KEY=sk-your-key-here     # Windows

# 3. Generate
python generate_problems_api.py --provider openai --num-problems 5
```

**Pros:**
- ✅ Excellent quality
- ✅ Highly creative
- ✅ Customizable (any number of problems)
- ✅ Different topics/difficulties

**Cons:**
- ⚠️ Requires API key
- ⚠️ Costs ~$0.10-0.30 for 5 problems
- ⚠️ Needs internet

**Output:** Custom-generated problems in `data/custom_problems.json`

---

### Option 3: Claude API (Excellent Alternative) 🎯

**Best for:** Alternative to GPT-4, excellent quality

**Setup:**
```bash
# 1. Install package
pip install anthropic

# 2. Set API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here  # Linux/Mac
set ANTHROPIC_API_KEY=sk-ant-your-key-here     # Windows

# 3. Generate
python generate_problems_api.py --provider anthropic --num-problems 5
```

**Pros:**
- ✅ Excellent quality
- ✅ Very creative
- ✅ Good at edge cases
- ✅ Customizable

**Cons:**
- ⚠️ Requires API key
- ⚠️ Costs ~$0.15-0.40 for 5 problems
- ⚠️ Needs internet

**Output:** Custom-generated problems in `data/custom_problems.json`

---

## Comparison Table

| Method | Time | Cost | Quality | Setup |
|--------|------|------|---------|-------|
| **Pre-made** | Instant | Free | Good ⭐⭐⭐⭐ | None |
| **GPT-4** | 30 sec | $0.20 | Excellent ⭐⭐⭐⭐⭐ | API key |
| **Claude** | 30 sec | $0.25 | Excellent ⭐⭐⭐⭐⭐ | API key |

---

## Detailed Examples

### Pre-Made Problems

```bash
# Generate 5 pre-made problems
python generate_custom_problems.py

# Output:
# ✓ Generated 5 custom problems
# ✓ Saved to data/custom_problems.json
#
# Problems:
#   - circular_buffer_median (medium)
#   - bitwise_range_query (medium)
#   - string_compression_with_frequency (easy)
#   - matrix_spiral_sum (medium)
#   - interval_merge_with_priority (hard)
```

### GPT-4 Generation

```bash
# Generate 5 problems with GPT-4
python generate_problems_api.py \
  --provider openai \
  --num-problems 5 \
  --output data/custom_problems.json

# Generate 10 problems (more variety)
python generate_problems_api.py \
  --provider openai \
  --num-problems 10

# Use GPT-3.5 (cheaper, faster)
python generate_problems_api.py \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-problems 5
```

### Claude Generation

```bash
# Generate 5 problems with Claude
python generate_problems_api.py \
  --provider anthropic \
  --num-problems 5 \
  --output data/custom_problems.json

# Use Claude 3 Haiku (cheaper, faster)
python generate_problems_api.py \
  --provider anthropic \
  --model claude-3-haiku-20240307 \
  --num-problems 5
```

---

## Which Should You Use?

### Use Pre-Made If:
- ✅ You want to start immediately
- ✅ You don't have API keys
- ✅ You're just testing the system
- ✅ You want zero cost

### Use GPT-4 If:
- ✅ You want best quality
- ✅ You have OpenAI API access
- ✅ You want custom problem topics
- ✅ You're doing research/publication

### Use Claude If:
- ✅ You want excellent quality
- ✅ You have Anthropic API access
- ✅ You prefer Claude's style
- ✅ You're doing research/publication

---

## Automated Workflow

The `run_cpu_benchmark` scripts now ask which method you want:

```bash
run_cpu_benchmark.bat  # Windows
./run_cpu_benchmark.sh # Linux/Mac
```

**You'll see:**
```
Choose problem generation method:
  1. Use pre-made problems (instant, good quality)
  2. Generate with GPT-4 API (best quality, requires API key)
  3. Generate with Claude API (excellent quality, requires API key)

Enter choice (1-3, default=1):
```

Just press Enter for pre-made (fastest), or choose 2/3 for API generation.

---

## API Key Setup (Quick)

### OpenAI (GPT-4)

1. Get key: https://platform.openai.com/api-keys
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   ```
3. Install: `pip install openai`

### Anthropic (Claude)

1. Get key: https://console.anthropic.com/
2. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```
3. Install: `pip install anthropic`

See **API_SETUP_GUIDE.md** for detailed instructions.

---

## My Recommendation

**For quick testing:** Use pre-made problems (Option 1)
- Zero setup, instant results
- Good enough to demonstrate the method

**For research/publication:** Use GPT-4 or Claude (Option 2/3)
- Higher quality, more diverse
- Better for showing robust improvement
- Worth the small cost (~$0.20)

**Best of both worlds:**
1. Start with pre-made to test everything works
2. Then generate with API for final results

---

## Example Workflow

### Quick Test (5 minutes)
```bash
# Use pre-made
python generate_custom_problems.py
python run_training.py --problems-file data/custom_problems.json --device cpu
```

### Research Quality (35 minutes)
```bash
# Generate with GPT-4
export OPENAI_API_KEY=sk-your-key-here
python generate_problems_api.py --provider openai --num-problems 5

# Train
python run_training.py --problems-file data/custom_problems.json --device cpu

# Benchmark
python benchmark_improvement.py \
  --baseline-model "Qwen/Qwen2.5-Coder-0.5B" \
  --trained-checkpoint checkpoints/checkpoint_best.pt \
  --problems-file data/custom_problems.json
```

---

## Summary

✅ **All three options work great!**
✅ **Pre-made is fastest** (instant, free)
✅ **APIs give best quality** (~30 sec, ~$0.20)
✅ **Choose based on your needs**

Start with pre-made, upgrade to API if you want better quality! 🚀
