def dot_product(matrix_a, matrix_b):
    """
    Compute the dot product between two matrices
    - matrix_a: shape (m, n)
    - matrix_b: shape (n, 1)
    - returns: matrix of shape (m, 1)
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError(f"Matrix dimensions don't match for dot product: {len(matrix_a[0])} != {len(matrix_b)}")
    
    result = [[0] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b)):
            result[i][0] += matrix_a[i][j] * matrix_b[j][0]
    
    return result

def matrix_transpose(matrix):
    """
    Transpose a matrix
    - matrix: shape (m, n)
    - returns: matrix of shape (n, m)
    """
    if not matrix or not matrix[0]:
        return []
    
    rows, cols = len(matrix), len(matrix[0])
    result = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    
    return result