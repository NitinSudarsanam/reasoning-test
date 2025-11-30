# In-Depth Python File Explanations

This document provides detailed explanations of each Python file in the adversarial RL multi-stage reasoning system with **progressive test generation**.

---

## System Overview

The system implements **adversarial reinforcement learning** where:
- **Generator** produces multi-stage reasoning (informal → structured → pseudocode → constraints → code)
- **Discriminator** generates test cases at **every stage** (not just the final stage)
- Tests **accumulate** across stages to form a comprehensive test suite
- **Rewards** computed from executing the full accumulated test suite
- Both models trained using **PPO** in alternating fashion

### Key Innovation: Progressive Test Generation

Unlike traditional systems that only test final code, this system:
1. Generates tests at each reasoning stage (stages 1-5)
2. Accumulates tests: `stage_1_tests + stage_2_tests + ... + stage_5_tests`
3. Executes all accumulated tests against final code
4. Uses results to compute rewards for training each stage

---

## 1. `reasoning/stages.py` - Multi-Stage Pipeline with Progressive Tests

### Purpose
Defines the 5-stage reasoning pipeline where **both generator and discriminator have specific roles at each stage**.

### Class: `ReasoningStage`

```python
@dataclass
class ReasoningStage:
    id: int                              # Stage number (1-5)
    name: str                            # Human-readable name
    description: str                     # What this stage represents
    generator_prompt_template: str       # Prompt for generating reasoning
    discriminator_prompt_template: str   # Prompt for generating TESTS
```

### The 5 Stages

#### **Stage 1: Informal Reasoning**
**Generator**: Produces conversational, intuitive understanding
```
"We need to find two numbers that add up to target. 
A hash map would work well here..."
```

**Discriminator**: Generates **basic functionality tests**
```python
def test_basic_case():
    assert two_sum([2,7,11,15], 9) == [0,1]

def test_empty_array():
    assert two_sum([], 5) == []
```

**Focus**: Happy path, empty input, single element cases

---

#### **Stage 2: Structured Reasoning**
**Generator**: Produces organized breakdown with steps
```
1. Problem: Find indices of two numbers summing to target
2. Approach: Use hash map for O(1) lookup
3. Edge cases: duplicates, no solution
```

**Discriminator**: Generates **edge case tests**
```python
def test_duplicates():
    assert two_sum([3,3], 6) == [0,1]

def test_no_solution():
    assert two_sum([1,2,3], 10) == []
```

**Focus**: Boundary conditions, edge cases identified in reasoning

---

#### **Stage 3: Pseudocode**
**Generator**: Produces algorithm in pseudocode
```
for i, num in enumerate(nums):
    complement = target - num
    if complement in seen:
        return [seen[complement], i]
    seen[num] = i
```

**Discriminator**: Generates **algorithmic tests**
```python
def test_same_element_twice():
    assert two_sum([5,5,5], 10) == [0,1]

def test_order_matters():
    assert two_sum([1,2,3,4], 7) == [2,3]
```

**Focus**: Loop boundaries, off-by-one errors, algorithmic corner cases

---

#### **Stage 4: Constraints and Invariants**
**Generator**: Produces formal constraints
```
Time: O(n), Space: O(n)
Invariant: seen contains all previous numbers with indices
Handles: negative numbers, duplicates
```

**Discriminator**: Generates **constraint-testing tests**
```python
def test_negative_numbers():
    assert two_sum([-1,-2,-3], -5) == [1,2]

def test_large_array():
    nums = list(range(10000))
    assert two_sum(nums, 19999) == [9999, 10000]
```

**Focus**: Constraint violations, complexity stress tests

---

#### **Stage 5: Executable Code**
**Generator**: Produces final Python implementation
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**Discriminator**: Generates **adversarial tests**
```python
def test_zero_target():
    assert two_sum([0,4,3,0], 0) == [0,3]

def test_tricky_case():
    assert two_sum([3,2,4], 6) == [1,2]
```

**Focus**: Tricky inputs, unexpected combinations, stress tests

---

### Key Functions

**`get_stage(stage_id: int) -> ReasoningStage`**
- Returns the ReasoningStage object for given ID (1-5)
- Validates ID is in valid range
- Used throughout training to get prompts

**`get_all_stages() -> List[ReasoningStage]`**
- Returns all 5 stages in order
- Used for iteration and validation

### Design Rationale

**Why different test types at each stage?**
- **Progressive difficulty**: Start easy, get harder
- **Strategic learning**: Discriminator learns WHEN to test WHAT
- **Comprehensive coverage**: Different stages catch different bugs
- **Natural curriculum**: Mirrors human testing approach

---

## 2. `training/adversarial_trainer.py` - Main Training Orchestrator

### Purpose
Orchestrates adversarial training with **progressive test accumulation** across all stages.

### Class: `AdversarialTrainer`

#### Key Innovation: `_generate_full_chain_with_tests()`

This is the **core method** that implements progressive test generation:

```python
def _generate_full_chain_with_tests(problem, training_stage):
    """
    Generates ALL 5 stages and accumulates tests from each.
    
    Key behavior:
    - Stages 1 to training_stage-1: Frozen (no gradients)
    - Stages training_stage to 5: Trainable (with gradients)
    - Tests generated at EVERY stage
    - Tests accumulated: stage_1_tests + ... + stage_5_tests
    
    Returns:
        (reasoning_chain, final_code, accumulated_tests)
    """
```

**Process**:
1. Loop through stages 1-5
2. For each stage:
   - Generate reasoning/code (frozen if before training_stage, trainable otherwise)
   - Generate tests for that stage
   - Append tests to accumulated_tests list
3. Combine all tests: `"\n\n".join(accumulated_tests)`
4. Return full chain, final code, and ALL tests

**Example when training stage 2**:
```python
# Stage 1: Frozen (no_grad)
with torch.no_grad():
    stage_1_reasoning = generator.generate_stage_1()
    stage_1_tests = discriminator.generate_tests_stage_1()

# Stage 2: Trainable (gradients flow)
stage_2_reasoning = generator.generate_stage_2()  # ← Training this
stage_2_tests = discriminator.generate_tests_stage_2()  # ← Training this

# Stages 3-5: Trainable
stage_3_reasoning = generator.generate_stage_3()
stage_3_tests = discriminator.generate_tests_stage_3()
# ... continue for 4, 5

# Accumulate ALL tests
all_tests = stage_1_tests + stage_2_tests + stage_3_tests + stage_4_tests + stage_5_tests

# Execute against final code
result = sandbox.execute_tests(final_code, all_tests)

# Reward based on ALL tests
reward = compute_reward(result)
```

---

#### `train_discriminator_epoch(stage_id, problems, n_steps)`

**Trains discriminator at a specific stage with frozen generator.**

**Process**:
1. Freeze generator, unfreeze discriminator
2. For each training step:
   - Call `_generate_full_chain_with_tests(problem, stage_id)`
   - Get accumulated tests from ALL stages
   - Get log probs for THIS stage's tests (for PPO)
   - Execute ALL tests against final code
   - Validate ALL tests against ground truth
   - Compute reward: `(1 - pass_rate) * test_validity`
   - Update discriminator weights with PPO

**Key insight**: Discriminator at stage N learns to generate tests that contribute to overall failure rate, even though tests from all stages are used.

---

#### `train_generator_epoch(stage_id, problems, n_steps)`

**Trains generator at a specific stage with frozen discriminator.**

**Process**:
1. Freeze discriminator, unfreeze generator
2. For each training step:
   - Call `_generate_full_chain_with_tests(problem, stage_id)`
   - Get accumulated tests from ALL stages
   - Get log probs for THIS stage's generation (for PPO)
   - Execute ALL tests against final code
   - Compute reward: `pass_rate`
   - Update generator weights with PPO

**Key insight**: Generator at stage N learns how its output affects final test pass rate, even though it's many steps away from final code.

---

#### `train_stage(stage_id, problems)`

**Implements the N+N+K training pattern for one stage.**

```python
# Phase 1: Train discriminator (N steps)
train_discriminator_epoch(stage_id, problems, N)

# Phase 2: Train generator (N steps)
train_generator_epoch(stage_id, problems, N)

# Phase 3: Alternating competition (K steps)
for k in range(K):
    if k % 2 == 0:
        train_generator_epoch(stage_id, problems, 1)
    else:
        train_discriminator_epoch(stage_id, problems, 1)
```

**Why this pattern?**
- **Phase 1**: Discriminator learns to generate hard tests
- **Phase 2**: Generator learns to pass those tests
- **Phase 3**: Rapid back-and-forth creates intense competition

---

#### `train_full_pipeline(problems)`

**Trains all 5 stages sequentially (1→2→3→4→5).**

```python
for stage_id in [1, 2, 3, 4, 5]:
    train_stage(stage_id, problems)
    # Stage N fully trained before moving to N+1
```

**Why sequential?**
- Later stages depend on earlier ones
- Ensures stable foundation before building on it
- Allows cumulative improvement

**Expected progression**:
```
After stage 1: 60-70% pass rate
After stage 2: 70-80% pass rate (improvement!)
After stage 3: 80-85% pass rate (improvement!)
After stage 4: 85-90% pass rate (improvement!)
After stage 5: 90-95% pass rate (best!)
```

---

### Training Flow Diagram

```
Training Stage 2:
  ┌─────────────────────────────────────┐
  │ Generate Stage 1 (frozen, no_grad)  │
  │   - Reasoning                       │
  │   - Tests                           │
  ├─────────────────────────────────────┤
  │ Generate Stage 2 (trainable) ←──────┼─── Training this!
  │   - Reasoning                       │
  │   - Tests                           │
  ├─────────────────────────────────────┤
  │ Generate Stages 3-5 (trainable)     │
  │   - Reasoning for each              │
  │   - Tests for each                  │
  ├─────────────────────────────────────┤
  │ Accumulate ALL tests                │
  │   tests = s1 + s2 + s3 + s4 + s5    │
  ├─────────────────────────────────────┤
  │ Execute tests against final code    │
  │   result = sandbox.execute(code, tests) │
  ├─────────────────────────────────────┤
  │ Compute reward from ALL tests       │
  │   reward = pass_rate                │
  ├─────────────────────────────────────┤
  │ Update Stage 2 weights with PPO     │
  └─────────────────────────────────────┘
```

---

## 3. `training/reward.py` - Reward Computation

### Purpose
Computes rewards for both generator and discriminator based on test execution results.

### Key Functions

#### `compute_generator_reward(execution_result)`

**Simple and direct**: Reward = test pass rate

```python
reward = num_passed / num_total
```

**Examples**:
- 8/10 tests passed → reward = 0.80
- 10/10 tests passed → reward = 1.00
- 0/10 tests passed → reward = 0.00
- Timeout → reward = 0.00
- No tests → reward = 0.50 (neutral)

**Generator's goal**: Maximize this value

---

#### `compute_discriminator_reward(generator_result, validation_result)`

**Adversarial with validity constraint**:

```python
generator_pass_rate = generator_result.num_passed / generator_result.num_total
test_validity = validation_result.num_passed / validation_result.num_total

reward = (1.0 - generator_pass_rate) * test_validity
```

**Two components**:
1. **Adversarial score**: `1.0 - generator_pass_rate`
   - High when generator fails tests
   - Incentivizes hard tests

2. **Validity score**: `test_validity`
   - High when tests pass ground truth
   - Penalizes invalid/contradictory tests

**Examples**:
```
Generator passes 8/10 tests
Tests pass 10/10 ground truth checks

Adversarial score = 1.0 - 0.8 = 0.2
Validity score = 10/10 = 1.0
Discriminator reward = 0.2 * 1.0 = 0.20

---

Generator passes 8/10 tests
Tests pass 6/10 ground truth checks (some invalid!)

Adversarial score = 1.0 - 0.8 = 0.2
Validity score = 6/10 = 0.6
Discriminator reward = 0.2 * 0.6 = 0.12 (penalized!)
```

**Discriminator's goal**: Minimize generator success while keeping tests valid

---

### Why These Formulas?

**Generator formula is simple**:
- Direct: More passed tests = higher reward
- Unambiguous: No interpretation needed
- Objective: Based on execution, not opinion

**Discriminator formula creates competition**:
- **Adversarial**: Wants generator to fail
- **Constrained**: Must generate valid tests
- **Zero-sum**: Generator's success is discriminator's failure

**Together they create a competitive dynamic**:
- Generator improves → passes more tests → discriminator gets lower reward
- Discriminator improves → generates harder tests → generator gets lower reward
- Both forced to improve through competition

---

## 4. Other Key Files (Brief Overview)

### `models/generator.py` - Generator LLM Wrapper

**Purpose**: Wraps HuggingFace model to generate reasoning and code

**Key methods**:
- `generate_stage_output()`: Generates reasoning for stages 1-4
- `generate_code()`: Generates executable code for stage 5
- `get_log_probs()`: Returns log probabilities for PPO training
- `_sanitize_output()`: Cleans up generated text
- `_extract_code_from_markdown()`: Extracts code from markdown blocks

**Used by**: AdversarialTrainer to generate all reasoning stages

---

### `models/discriminator.py` - Discriminator LLM Wrapper

**Purpose**: Wraps HuggingFace model to generate test cases

**Key methods**:
- `generate_tests()`: Generates pytest test functions
- `generate_critique()`: Generates textual critiques (not heavily used)
- `get_log_probs()`: Returns log probabilities for PPO training
- `_sanitize_test_code()`: Ensures tests are syntactically valid

**Used by**: AdversarialTrainer to generate tests at all stages

---

### `sandbox/sandbox.py` - Secure Code Execution

**Purpose**: Safely executes code and tests in isolated subprocess

**Key methods**:
- `execute_tests(code, tests)`: Runs tests against code
- `validate_tests_against_solution(tests, solution)`: Validates test correctness
- `_run_pytest()`: Executes pytest in subprocess
- `_parse_pytest_output()`: Extracts pass/fail counts

**Security features**:
- Subprocess isolation
- Timeout protection
- Temporary directories (auto-deleted)
- Output capture

---

### `training/rl_loop.py` - PPO Implementation

**Purpose**: Implements Proximal Policy Optimization for weight updates

**Key functions**:
- `compute_policy_loss()`: Computes PPO clipped objective
- `train_step()`: Executes one RL update
- `freeze_model()` / `unfreeze_model()`: Controls gradient flow
- `create_optimizer()`: Creates AdamW optimizer

**PPO algorithm**:
```python
ratio = exp(new_log_prob - old_log_prob)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

---

### `data/problem_dataset.py` - Problem Management

**Purpose**: Loads and validates coding problems with ground truth

**Key components**:
- `Problem` dataclass: Stores problem info
- `load_problems()`: Loads from JSON
- `validate_problem()`: Checks syntax and structure

**Problem structure**:
```json
{
  "id": "two_sum",
  "description": "Find two numbers that add up to target",
  "function_signature": "def two_sum(nums, target):",
  "baseline_tests": ["assert two_sum([2,7], 9) == [0,1]"],
  "reference_solution": "def two_sum(nums, target): ...",
  "difficulty": "easy",
  "tags": ["array", "hash-table"]
}
```

---

### `evaluation/metrics.py` - Evaluation Metrics

**Purpose**: Computes performance metrics

**Key functions**:
- `compute_pass_rate()`: Average test pass rate
- `compute_failure_rate()`: Average test failure rate
- `compute_test_diversity()`: Uniqueness of generated tests

---

### `training/checkpoint_manager.py` - Checkpoint Management

**Purpose**: Manages saving and loading of model checkpoints during training

**Key class**: `CheckpointManager`

**Key methods**:
- `save_checkpoint()`: Saves generator and discriminator weights with metadata
- `load_checkpoint()`: Restores models from saved checkpoint
- `get_best_checkpoint()`: Returns path to best performing checkpoint
- `get_latest_checkpoint()`: Returns path to most recent checkpoint
- `list_checkpoints()`: Lists all available checkpoints
- `compute_checkpoint_score()`: Computes combined score (0.7 * gen_reward + 0.3 * test_validity)
- `should_save_as_best()`: Determines if checkpoint is new best

**Checkpoint structure**:
```python
{
    "stage": 3,
    "epoch": 15,
    "generator_state_dict": {...},
    "discriminator_state_dict": {...},
    "metrics": {
        "generator_reward": 0.75,
        "discriminator_reward": 0.68,
        "test_validity": 0.92
    },
    "timestamp": "2025-11-30T10:30:00",
    "config": {...}
}
```

**Features**:
- Progressive checkpoints: Saves after each stage
- Best checkpoint tracking: Keeps best model based on combined metrics
- Metadata storage: Includes all training information
- Resumable training: Supports loading and continuing from any checkpoint
- Corruption detection: Validates checkpoint integrity before loading

**Used by**: AdversarialTrainer to save checkpoints during training

---

### `inference/inference_engine.py` - Inference Engine

**Purpose**: Uses trained models to solve new problems through multi-stage reasoning

**Key class**: `InferenceEngine`

**Key methods**:
- `from_checkpoint()`: Class method to load engine from saved checkpoint
- `solve_problem()`: Executes multi-stage reasoning on a single problem
- `solve_batch()`: Processes multiple problems efficiently
- `get_reasoning_chain()`: Returns only reasoning without code execution

**InferenceResult dataclass**:
```python
@dataclass
class InferenceResult:
    problem_description: str
    reasoning_chain: List[str]      # Outputs from stages 1-4
    generated_code: str              # Output from stage 5
    execution_result: Optional[ExecutionResult]
    inference_time: float
    stage_times: List[float]         # Time per stage
```

**Inference flow**:
1. Load trained models from checkpoint
2. For each problem:
   - Stage 1: Generate informal reasoning
   - Stage 2: Generate structured reasoning (using stage 1)
   - Stage 3: Generate pseudocode (using stages 1-2)
   - Stage 4: Generate constraints/invariants (using stages 1-3)
   - Stage 5: Generate executable code (using stages 1-4)
   - Optionally: Execute code against test cases
3. Return complete reasoning chain and results

**Features**:
- Models in eval mode (no training)
- Support for single and batch inference
- Optional test execution for validation
- Complete reasoning chain for interpretability
- Timing metrics for performance analysis

**Used by**: run_inference.py for solving new problems with trained models
- `compute_reasoning_coherence()`: Consistency across stages

---

## System Integration

### Complete Training Flow

```
1. Load problems with ground truth solutions
2. Initialize generator and discriminator models
3. Create sandbox for execution
4. Create adversarial trainer

5. For each stage (1 to 5):
   a. Train discriminator epoch:
      - Generate full chain (stages 1-5)
      - Accumulate tests from all stages
      - Execute against final code
      - Compute reward from results
      - Update discriminator weights
   
   b. Train generator epoch:
      - Generate full chain (stages 1-5)
      - Accumulate tests from all stages
      - Execute against final code
      - Compute reward from results
      - Update generator weights
   
   c. Alternating training:
      - Rapid back-and-forth competition
      - Both models improve simultaneously

6. Save results and metrics
```

### Data Flow

```
Problem
  ↓
Generator → Stage 1 reasoning → Discriminator → Stage 1 tests
  ↓                                                ↓
Generator → Stage 2 reasoning → Discriminator → Stage 2 tests
  ↓                                                ↓
Generator → Stage 3 reasoning → Discriminator → Stage 3 tests
  ↓                                                ↓
Generator → Stage 4 reasoning → Discriminator → Stage 4 tests
  ↓                                                ↓
Generator → Stage 5 code → Discriminator → Stage 5 tests
  ↓                                                ↓
  └────────────────────────────────────────────────┘
                         ↓
              Accumulated Test Suite
              (all tests combined)
                         ↓
                  Sandbox Execution
                         ↓
                  Execution Results
                         ↓
                  Reward Computation
                         ↓
                    PPO Updates
```

---

## Key Design Principles

### 1. Progressive Test Generation
- Tests generated at every stage, not just final
- Creates comprehensive test coverage
- Different stages test different aspects

### 2. Test Accumulation
- All tests combined into single suite
- Final code tested against everything
- Ensures all stages contribute

### 3. Forward Training
- Train stages 1→2→3→4→5 sequentially
- Earlier stages frozen when training later ones
- Ensures stable foundation

### 4. Adversarial Competition
- Generator maximizes pass rate
- Discriminator minimizes pass rate
- Zero-sum game drives improvement

### 5. Grounded Rewards
- All rewards from actual test execution
- No ambiguous quality scores
- Objective, measurable outcomes

---

This completes the in-depth explanation of all Python files in the current progressive test generation system!
