"""Test if sandbox_simple works with custom problems."""

from sandbox.sandbox_simple import execute_tests_simple

# Simple test code
code = """
class CircularBufferMedian:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, value):
        if len(self.buffer) < self.capacity:
            self.buffer.append(value)
        else:
            self.buffer[0] = value
    
    def get_median(self):
        s = sorted(self.buffer)
        n = len(s)
        if n % 2 == 1:
            return float(s[n//2])
        return (s[n//2-1] + s[n//2]) / 2.0
    
    def get_size(self):
        return len(self.buffer)
"""

# Tests from custom_problems.json
tests = """cb = CircularBufferMedian(3); cb.add(1); cb.add(2); cb.add(3); assert cb.get_median() == 2.0
cb = CircularBufferMedian(3); cb.add(5); assert cb.get_median() == 5.0
cb = CircularBufferMedian(2); cb.add(1); cb.add(2); cb.add(3); assert cb.get_median() == 2.5"""

result = execute_tests_simple(code, tests)

print(f"Passed: {result.passed}")
print(f"Tests: {result.num_passed}/{result.num_total}")
print(f"Errors: {result.errors}")
