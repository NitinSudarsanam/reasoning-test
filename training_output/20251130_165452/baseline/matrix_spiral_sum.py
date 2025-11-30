# Problem: matrix_spiral_sum
# Description: Given an m x n matrix of integers, return the sum of all elements traversed in spiral order (clockwi...

def spiral_sum(matrix: list[list[int]]) -> int:
    # TODO: Implement the solution here
    
    m = len(matrix)
    n = len(matrix[0])
    
    left = 0
    right = m-1
    top = 0
    bottom = n-1
    
    while right >= left and top >= bottom:
        
        for i in range(left, right+1):
            matrix[top][i] += matrix[bottom][i]
            
        for i in range(top+1, bottom+1):
            matrix[i][right] += matrix[i][right-1]
            
        for i in range(bottom+1, top, -1):
            matrix[i][left] += matrix[i][left+1]
            
        for i in range(left+1, right, -1):
            matrix[top][i] += matrix[top][i-1]
            
        for i in range(top+1, bottom, -1):