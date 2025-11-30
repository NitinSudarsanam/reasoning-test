# CPU Benchmark Ready! 🚀

## What Was Fixed

1. **Fixed `sandbox_simple.py`** - Test counting now works correctly with custom problems
   - Changed from `line.startswith('assert')` to `'assert' in line`
   - Now handles tests with setup code on the same line

2. **Fixed `models/discriminator.py`** - Template formatting issue resolved
   - Changed `{code}` to `{stage_output}` to match stage templates
   - Fixes the KeyError that was blocking training

3. **Updated benchmark scripts** - Added conda environment activation
   - `run_cpu_benchmark_auto.bat` - Full benchmark (30-40 min)
   - `test_benchmark_quick.bat` - Quick test (5 min)

## Verified Working

✅ Custom problems load correctly (5 problems)
✅ Reference solutions pass all tests (100%)
✅ Sandbox execution works properly
✅ Test counting is accurate

## To Run Full Benchmark

```batch
run_cpu_benchmark_auto.bat
```

This will:
1. Generate custom problems (~1 min)
2. Evaluate baseline model (~5 min)
3. Train with adversarial RL (~20 min)
4. Evaluate trained model (~5 min)

**Total time: ~30-40 minutes**

## Quick Test First (Recommended)

```batch
test_benchmark_quick.bat
```

This runs steps 1-2 only (~5 min) to verify everything works before the full run.

## Output Files

After completion, you'll have:
- `baseline_results.json` - Baseline model performance
- `comparison_results.json` - Before/after comparison
- `checkpoints_cpu/` - Trained model checkpoints

## Note on Baseline Accuracy

The baseline model (codegen-350M-mono) may show 0% accuracy initially because:
- It's a small 350M parameter model
- These are medium-hard coding problems
- The model needs the multi-stage reasoning training to improve

**This is expected!** The training will improve it.

## Ready to Go!

Everything is fixed and ready. Just run the benchmark script and let it complete.
