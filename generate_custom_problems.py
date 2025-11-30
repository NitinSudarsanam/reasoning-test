"""Generate custom coding problems for training that are unlikely to be in model training data."""

import json
from pathlib import Path


def generate_custom_problems():
    """Generate novel coding problems for benchmarking."""
    
    problems = [
        {
            "id": "circular_buffer_median",
            "description": """Implement a circular buffer that maintains a sliding window of integers and 
can efficiently compute the median of the current window. The buffer has a fixed capacity and 
overwrites the oldest element when full. Support operations: add(value), get_median(), and get_size().""",
            "function_signature": "class CircularBufferMedian:\n    def __init__(self, capacity: int):\n        pass\n    def add(self, value: int) -> None:\n        pass\n    def get_median(self) -> float:\n        pass\n    def get_size(self) -> int:\n        pass",
            "baseline_tests": [
                "cb = CircularBufferMedian(3); cb.add(1); cb.add(2); cb.add(3); assert cb.get_median() == 2.0",
                "cb = CircularBufferMedian(3); cb.add(5); assert cb.get_median() == 5.0",
                "cb = CircularBufferMedian(2); cb.add(1); cb.add(2); cb.add(3); assert cb.get_median() == 2.5"
            ],
            "reference_solution": """class CircularBufferMedian:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.index = 0
    
    def add(self, value: int) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(value)
        else:
            self.buffer[self.index] = value
            self.index = (self.index + 1) % self.capacity
    
    def get_median(self) -> float:
        sorted_buffer = sorted(self.buffer)
        n = len(sorted_buffer)
        if n % 2 == 1:
            return float(sorted_buffer[n // 2])
        else:
            return (sorted_buffer[n // 2 - 1] + sorted_buffer[n // 2]) / 2.0
    
    def get_size(self) -> int:
        return len(self.buffer)""",
            "difficulty": "medium",
            "tags": ["data-structures", "sliding-window", "median"]
        },
        {
            "id": "bitwise_range_query",
            "description": """Given an array of integers, implement a data structure that supports:
1. update(index, value): Update array[index] to value
2. query_xor(left, right): Return XOR of all elements from index left to right (inclusive)
3. query_or(left, right): Return bitwise OR of all elements from index left to right (inclusive)""",
            "function_signature": "class BitwiseRangeQuery:\n    def __init__(self, arr: list[int]):\n        pass\n    def update(self, index: int, value: int) -> None:\n        pass\n    def query_xor(self, left: int, right: int) -> int:\n        pass\n    def query_or(self, left: int, right: int) -> int:\n        pass",
            "baseline_tests": [
                "brq = BitwiseRangeQuery([1, 2, 3, 4]); assert brq.query_xor(0, 3) == (1^2^3^4)",
                "brq = BitwiseRangeQuery([5, 6, 7]); assert brq.query_or(0, 2) == (5|6|7)",
                "brq = BitwiseRangeQuery([1, 2, 3]); brq.update(1, 5); assert brq.query_xor(0, 2) == (1^5^3)"
            ],
            "reference_solution": """class BitwiseRangeQuery:
    def __init__(self, arr: list[int]):
        self.arr = arr[:]
    
    def update(self, index: int, value: int) -> None:
        self.arr[index] = value
    
    def query_xor(self, left: int, right: int) -> int:
        result = 0
        for i in range(left, right + 1):
            result ^= self.arr[i]
        return result
    
    def query_or(self, left: int, right: int) -> int:
        result = 0
        for i in range(left, right + 1):
            result |= self.arr[i]
        return result""",
            "difficulty": "medium",
            "tags": ["bitwise", "range-query", "data-structures"]
        },
        {
            "id": "string_compression_with_frequency",
            "description": """Implement a string compression algorithm that:
1. Groups consecutive identical characters
2. Replaces groups of 3+ chars with: char + count
3. Keeps groups of 1-2 chars as-is
4. Returns the shorter of original or compressed string
Example: "aaabbc" -> "a3bbc" (7 chars -> 5 chars)
Example: "abc" -> "abc" (no compression needed)""",
            "function_signature": "def compress_string(s: str) -> str:",
            "baseline_tests": [
                "assert compress_string('aaabbc') == 'a3bbc'",
                "assert compress_string('abc') == 'abc'",
                "assert compress_string('aabbcc') == 'aabbcc'",
                "assert compress_string('aaaa') == 'a4'"
            ],
            "reference_solution": """def compress_string(s: str) -> str:
    if not s:
        return s
    
    compressed = []
    count = 1
    current = s[0]
    
    for i in range(1, len(s)):
        if s[i] == current:
            count += 1
        else:
            if count >= 3:
                compressed.append(current + str(count))
            else:
                compressed.append(current * count)
            current = s[i]
            count = 1
    
    # Handle last group
    if count >= 3:
        compressed.append(current + str(count))
    else:
        compressed.append(current * count)
    
    compressed_str = ''.join(compressed)
    return compressed_str if len(compressed_str) < len(s) else s""",
            "difficulty": "easy",
            "tags": ["string", "compression", "greedy"]
        },
        {
            "id": "matrix_spiral_sum",
            "description": """Given an m x n matrix of integers, return the sum of all elements 
traversed in spiral order (clockwise from outside to inside). 
Example: [[1,2,3],[4,5,6],[7,8,9]] -> 1+2+3+6+9+8+7+4+5 = 45""",
            "function_signature": "def spiral_sum(matrix: list[list[int]]) -> int:",
            "baseline_tests": [
                "assert spiral_sum([[1,2,3],[4,5,6],[7,8,9]]) == 45",
                "assert spiral_sum([[1,2],[3,4]]) == 10",
                "assert spiral_sum([[1]]) == 1",
                "assert spiral_sum([[1,2,3,4]]) == 10"
            ],
            "reference_solution": """def spiral_sum(matrix: list[list[int]]) -> int:
    if not matrix or not matrix[0]:
        return 0
    
    total = 0
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for col in range(left, right + 1):
            total += matrix[top][col]
        top += 1
        
        # Traverse down
        for row in range(top, bottom + 1):
            total += matrix[row][right]
        right -= 1
        
        # Traverse left
        if top <= bottom:
            for col in range(right, left - 1, -1):
                total += matrix[bottom][col]
            bottom -= 1
        
        # Traverse up
        if left <= right:
            for row in range(bottom, top - 1, -1):
                total += matrix[row][left]
            left += 1
    
    return total""",
            "difficulty": "medium",
            "tags": ["matrix", "spiral", "traversal"]
        },
        {
            "id": "interval_merge_with_priority",
            "description": """Given a list of intervals where each interval has [start, end, priority], 
merge overlapping intervals. When intervals overlap, keep the one with higher priority. 
If priorities are equal, merge them into one interval.
Example: [[1,3,1], [2,4,2], [5,7,1]] -> [[1,2,1], [2,4,2], [5,7,1]]""",
            "function_signature": "def merge_intervals_priority(intervals: list[list[int]]) -> list[list[int]]:",
            "baseline_tests": [
                "assert merge_intervals_priority([[1,3,1], [2,4,2]]) == [[1,2,1], [2,4,2]]",
                "assert merge_intervals_priority([[1,3,1], [2,4,1]]) == [[1,4,1]]",
                "assert merge_intervals_priority([[1,2,1]]) == [[1,2,1]]"
            ],
            "reference_solution": """def merge_intervals_priority(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]
    
    for current in intervals[1:]:
        last = result[-1]
        
        # Check for overlap
        if current[0] <= last[1]:
            # Overlapping
            if current[2] > last[2]:
                # Current has higher priority
                if current[0] > last[0]:
                    result[-1] = [last[0], current[0], last[2]]
                    result.append(current)
                else:
                    result[-1] = current
            elif current[2] == last[2]:
                # Same priority, merge
                result[-1] = [last[0], max(last[1], current[1]), last[2]]
            # else: last has higher priority, keep it
        else:
            # No overlap
            result.append(current)
    
    return result""",
            "difficulty": "hard",
            "tags": ["intervals", "merge", "priority"]
        }
    ]
    
    return {"problems": problems}


def save_problems(output_file: str = "data/custom_problems.json"):
    """Save generated problems to JSON file."""
    problems_data = generate_custom_problems()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(problems_data, f, indent=2)
    
    print(f"✓ Generated {len(problems_data['problems'])} custom problems")
    print(f"✓ Saved to {output_file}")
    print("\nProblems:")
    for p in problems_data['problems']:
        print(f"  - {p['id']} ({p['difficulty']}): {p['description'][:60]}...")


if __name__ == "__main__":
    save_problems()
