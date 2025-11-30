# System Architecture Overview - Progressive Test Generation

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     run_training.py                          │
│                    (Main Entry Point)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              AdversarialTrainer                              │
│         (training/adversarial_trainer.py)                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Generator    │  │Discriminator │  │   Sandbox    │     │
│  │    LLM       │  │     LLM      │  │   Executor   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         │  Reasoning       │  Tests at        │             │
│         │  at each stage   │  each stage      │             │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────────┐                       │
│              │ Accumulated Tests    │                       │
│              │ (stages 1-5 combined)│                       │
│              └──────────┬───────────┘                       │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────────┐                       │
│              │   Reward System      │                       │
│              │  (reward.py)         │                       │
│              └──────────┬───────────┘                       │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────────┐                       │
│              │    PPO Training      │                       │
│              │   (rl_loop.py)       │                       │
│              └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Progressive Test Generation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Stage N                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1 (Frozen if N>1):                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Generator → Informal Reasoning                     │    │
│  │ Discriminator → Basic Tests (2-3)                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Stage 2 (Frozen if N>2):                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Generator → Structured Reasoning                   │    │
│  │ Discriminator → Edge Case Tests (2-3)              │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Stage N (TRAINING):                                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Generator → Stage N Reasoning ← GRADIENTS          │    │
│  │ Discriminator → Stage N Tests ← GRADIENTS          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Stages N+1 to 5 (Trainable):                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Generator → Remaining Stages                       │    │
│  │ Discriminator → Remaining Tests                    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Accumulate ALL Tests:                              │    │
│  │ tests = stage_1 + stage_2 + ... + stage_5          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Execute Against Final Code:                        │    │
│  │ result = sandbox.execute(final_code, all_tests)    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Compute Rewards:                                   │    │
│  │ gen_reward = pass_rate                             │    │
│  │ disc_reward = (1 - pass_rate) * validity           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Update Stage N Weights with PPO                    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Stage Reasoning with Test Accumulation

```
Problem: "Implement two_sum function"

┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Informal Reasoning                                  │
├─────────────────────────────────────────────────────────────┤
│ Generator Output:                                            │
│   "Use hash map to store complements..."                    │
│                                                              │
│ Discriminator Output (Tests):                               │
│   def test_basic(): assert two_sum([2,7,11,15], 9) == [0,1] │
│   def test_empty(): assert two_sum([], 5) == []             │
│                                                              │
│ Accumulated Tests: 2 tests                                  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Structured Reasoning                                │
├─────────────────────────────────────────────────────────────┤
│ Generator Output:                                            │
│   "1. Create map 2. Iterate once 3. Return indices"         │
│                                                              │
│ Discriminator Output (Tests):                               │
│   def test_duplicates(): assert two_sum([3,3], 6) == [0,1]  │
│   def test_no_solution(): assert two_sum([1,2,3], 10) == [] │
│                                                              │
│ Accumulated Tests: 2 + 2 = 4 tests                          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Pseudocode                                          │
├─────────────────────────────────────────────────────────────┤
│ Generator Output:                                            │
│   "for i, num: if target-num in seen: return..."            │
│                                                              │
│ Discriminator Output (Tests):                               │
│   def test_same_element(): assert two_sum([5,5,5], 10)...   │
│   def test_order(): assert two_sum([1,2,3,4], 7) == [2,3]   │
│                                                              │
│ Accumulated Tests: 4 + 2 = 6 tests                          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Constraints                                         │
├─────────────────────────────────────────────────────────────┤
│ Generator Output:                                            │
│   "Time: O(n), Space: O(n), handles negatives"              │
│                                                              │
│ Discriminator Output (Tests):                               │
│   def test_negatives(): assert two_sum([-1,-2], -3)...      │
│   def test_large(): assert two_sum(range(10000), ...)       │
│                                                              │
│ Accumulated Tests: 6 + 2 = 8 tests                          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Executable Code                                     │
├─────────────────────────────────────────────────────────────┤
│ Generator Output:                                            │
│   def two_sum(nums, target):                                │
│       seen = {}                                              │
│       for i, num in enumerate(nums):                         │
│           complement = target - num                          │
│           if complement in seen:                             │
│               return [seen[complement], i]                   │
│           seen[num] = i                                      │
│       return []                                              │
│                                                              │
│ Discriminator Output (Tests):                               │
│   def test_zero(): assert two_sum([0,4,3,0], 0) == [0,3]    │
│   def test_tricky(): assert two_sum([3,2,4], 6) == [1,2]    │
│                                                              │
│ Accumulated Tests: 8 + 2 = 10 tests TOTAL                   │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Execute ALL 10 Tests Against Final Code                     │
├─────────────────────────────────────────────────────────────┤
│ Result: 8/10 tests passed (80%)                             │
│                                                              │
│ Rewards:                                                     │
│   Generator: 0.80 (80% pass rate)                           │
│   Discriminator: 0.20 (20% failure rate * 100% validity)    │
└─────────────────────────────────────────────────────────────┘
```

## Training Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Pipeline Training                    │
│                    (train_full_pipeline)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For stage_id in [1, 2, 3, 4, 5]:                          │
│                                                              │
│    ┌──────────────────────────────────────────────────┐    │
│    │ Phase 1: Train Discriminator (N steps)          │    │
│    │   - Freeze generator                             │    │
│    │   - Unfreeze discriminator                       │    │
│    │   - Generate full chain with tests               │    │
│    │   - Execute accumulated tests                    │    │
│    │   - Update discriminator weights                 │    │
│    └──────────────────────────────────────────────────┘    │
│                         ↓                                    │
│    ┌──────────────────────────────────────────────────┐    │
│    │ Phase 2: Train Generator (N steps)              │    │
│    │   - Freeze discriminator                         │    │
│    │   - Unfreeze generator                           │    │
│    │   - Generate full chain with tests               │    │
│    │   - Execute accumulated tests                    │    │
│    │   - Update generator weights                     │    │
│    └──────────────────────────────────────────────────┘    │
│                         ↓                                    │
│    ┌──────────────────────────────────────────────────┐    │
│    │ Phase 3: Alternating (K steps)                  │    │
│    │   - Step 0: Train generator                      │    │
│    │   - Step 1: Train discriminator                  │    │
│    │   - Step 2: Train generator                      │    │
│    │   - ...                                           │    │
│    │   - Rapid back-and-forth competition             │    │
│    └──────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Adversarial Competition Dynamics

```
┌─────────────────────────────────────────────────────────────┐
│                    Competition Cycle                         │
│                                                              │
│  Generator Goal: MAXIMIZE test pass rate                    │
│  Discriminator Goal: MINIMIZE test pass rate                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Generator produces better reasoning/code        │    │
│  │    → Pass rate increases (e.g., 70% → 80%)         │    │
│  │    → Generator reward increases                     │    │
│  │    → Discriminator reward decreases                 │    │
│  └────────────────────────────────────────────────────┘    │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 2. Discriminator generates harder tests            │    │
│  │    → Pass rate decreases (e.g., 80% → 75%)         │    │
│  │    → Discriminator reward increases                 │    │
│  │    → Generator reward decreases                     │    │
│  └────────────────────────────────────────────────────┘    │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 3. Generator improves to pass harder tests         │    │
│  │    → Pass rate increases (e.g., 75% → 85%)         │    │
│  │    → Generator reward increases                     │    │
│  │    → Discriminator reward decreases                 │    │
│  └────────────────────────────────────────────────────┘    │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 4. Cycle continues...                               │    │
│  │    → Both models improve through competition        │    │
│  │    → Pass rate stabilizes at high level             │    │
│  │    → System reaches equilibrium                     │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Result: Both models get better through adversarial         │
│          competition, leading to high-quality code and      │
│          comprehensive test coverage                         │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Generator (`models/generator.py`)
- Produces reasoning at each stage (1-4)
- Produces executable code at stage 5
- Provides log probabilities for RL
- **Goal**: Maximize test pass rate

### Discriminator (`models/discriminator.py`)
- Generates tests at each stage (1-5)
- Different test types per stage
- Provides log probabilities for RL
- **Goal**: Minimize pass rate while keeping tests valid

### Sandbox (`sandbox/sandbox.py`)
- Executes code securely
- Runs accumulated tests
- Validates tests against ground truth
- Returns structured results

### Reward System (`training/reward.py`)
- Computes generator reward: `pass_rate`
- Computes discriminator reward: `(1 - pass_rate) * validity`
- Creates adversarial competition
- Penalizes invalid tests

### PPO Loop (`training/rl_loop.py`)
- Implements policy gradient optimization
- Computes clipped surrogate objective
- Updates model weights
- Handles gradient clipping

### Adversarial Trainer (`training/adversarial_trainer.py`)
- Orchestrates full training pipeline
- Manages model freezing/unfreezing
- Implements N+N+K pattern
- Accumulates tests across stages

## Data Flow

```
1. Problem Loading
   ┌──────────────┐
   │   Problem    │ ← Load from JSON with ground truth
   │  "two_sum"   │
   └──────┬───────┘
          │
          ▼
2. Full Chain Generation
   ┌──────────────────────────────────────┐
   │  _generate_full_chain_with_tests()   │
   │                                      │
   │  For each stage 1-5:                 │
   │    - Generate reasoning/code         │
   │    - Generate tests                  │
   │    - Accumulate tests                │
   └──────────────┬───────────────────────┘
                  │
                  ▼
3. Test Accumulation
   ┌──────────────────────────────────────┐
   │  all_tests = stage_1_tests +         │
   │              stage_2_tests +         │
   │              stage_3_tests +         │
   │              stage_4_tests +         │
   │              stage_5_tests           │
   └──────────────┬───────────────────────┘
                  │
                  ▼
4. Execution
   ┌──────────────────────────────────────┐
   │  Sandbox                             │
   │  - Run all_tests against final_code  │
   │  - Run all_tests against ground_truth│
   └──────────────┬───────────────────────┘
                  │
                  ▼
5. Reward Computation
   ┌──────────────────────────────────────┐
   │  Generator: pass_rate                │
   │  Discriminator: (1-pass_rate)*valid  │
   └──────────────┬───────────────────────┘
                  │
                  ▼
6. Weight Update
   ┌──────────────────────────────────────┐
   │  PPO Algorithm                       │
   │  - Compute policy loss               │
   │  - Backward pass                     │
   │  - Update weights                    │
   └──────────────────────────────────────┘
```

## Key Design Patterns

### 1. Strategy Pattern
- Different test generation strategies per stage
- Same interface for all stages

### 2. Template Method Pattern
- `train_stage()` defines structure
- Subcomponents handle specifics

### 3. Accumulator Pattern
- Tests accumulated across stages
- Final suite used for rewards

### 4. Freezing Pattern
- Earlier stages frozen when training later ones
- Prevents catastrophic forgetting

### 5. Adversarial Pattern
- Generator and discriminator compete
- Zero-sum game drives improvement

## Performance Characteristics

### Memory Usage
```
┌─────────────────────────────────────┐
│  Generator Model: 6 GB              │
├─────────────────────────────────────┤
│  Discriminator Model: 6 GB          │
├─────────────────────────────────────┤
│  Optimizer States: 4 GB             │
├─────────────────────────────────────┤
│  Activations/Gradients: 4 GB        │
├─────────────────────────────────────┤
│  System/Python: 2 GB                │
├─────────────────────────────────────┤
│  TOTAL: ~22 GB                      │
└─────────────────────────────────────┘
```

### Training Time (per stage)
```
CPU (16GB RAM):
  - Discriminator epoch (N=2): ~20-30 seconds
  - Generator epoch (N=2): ~20-30 seconds
  - Alternating (K=2): ~20-30 seconds
  - Total per stage: ~60-90 seconds
  - Full pipeline (5 stages): ~5-8 minutes

GPU (T4):
  - Discriminator epoch (N=2): ~5-10 seconds
  - Generator epoch (N=2): ~5-10 seconds
  - Alternating (K=2): ~5-10 seconds
  - Total per stage: ~15-30 seconds
  - Full pipeline (5 stages): ~1-2 minutes
```

### Test Suite Growth
```
Stage 1: 2-3 tests (basic)
Stage 2: 2-3 tests (edge cases)
Stage 3: 2-3 tests (algorithmic)
Stage 4: 2-3 tests (constraints)
Stage 5: 2-3 tests (adversarial)
─────────────────────────────────
Total: 10-15 tests per problem
```

## System Benefits

### 1. Comprehensive Testing
- Tests from all stages combined
- Different aspects tested at different stages
- Progressive difficulty

### 2. Strategic Learning
- Discriminator learns WHEN to test WHAT
- Generator learns how each stage affects outcome
- Both develop stage-specific skills

### 3. Grounded Rewards
- All rewards from actual execution
- No ambiguous quality scores
- Objective, measurable

### 4. Curriculum Learning
- Natural easy→hard progression
- Earlier stages provide foundation
- Later stages build on solid base

### 5. Adversarial Improvement
- Competition drives both models to improve
- Zero-sum game prevents stagnation
- Continuous pressure to get better

---

This architecture enables robust, comprehensive testing through progressive test generation across all reasoning stages!
