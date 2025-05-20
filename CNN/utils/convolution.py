def colerate2d(X, K):
    """
    Perform a 2D correlation operation between an image and a kernel.
    
    Parameters:
    X (list): Input image of shape (H, W, C).
    K (list): Kernel of shape (kH, kW, C).
    
    Returns:
    list: Correlated image of shape (H-kH+1, W-kW+1, C).
    """
    # Get the dimensions of the input image and kernel
    H = len(X)
    W = len(X[0])
    C = len(X[0][0])
    
    kH = len(K)
    kW = len(K[0])
    
    # Calculate the output dimensions
    out_H = H - kH + 1
    out_W = W - kW + 1
    
    # Initialize the output array
    Y = [[[0 for _ in range(C)] for _ in range(out_W)] for _ in range(out_H)]
    
    # Perform the correlation operation
    for i in range(out_H):
        for j in range(out_W):
            for c in range(C):
                sum_val = 0
                for ki in range(kH):
                    for kj in range(kW):
                        sum_val += X[i+ki][j+kj][c] * K[ki][kj][c]
                Y[i][j][c] = sum_val
    
    return Y

def convolve2d(X, K):
    """
    Perform a 2D convolution operation between an image and a kernel.
    
    Parameters:
    X (list): Input image of shape (H, W, C).
    K (list): Kernel of shape (kH, kW, C).
    
    Returns:
    list: Convolved image of shape (H-kH+1, W-kW+1, C).
    """
    # Get the dimensions of the input image and kernel
    H = len(X)
    W = len(X[0])
    C = len(X[0][0])
    
    kH = len(K)
    kW = len(K[0])
    
    # Calculate the output dimensions
    out_H = H - kH + 1
    out_W = W - kW + 1
    
    # Initialize the output array
    Y = [[[0 for _ in range(C)] for _ in range(out_W)] for _ in range(out_H)]
    
    # Perform the convolution operation (correlation with flipped kernel)
    for i in range(out_H):
        for j in range(out_W):
            for c in range(C):
                sum_val = 0
                for ki in range(kH):
                    for kj in range(kW):
                        # Flip the kernel indices for convolution
                        k_i_flipped = kH - 1 - ki
                        k_j_flipped = kW - 1 - kj
                        sum_val += X[i+ki][j+kj][c] * K[k_i_flipped][k_j_flipped][c]
                Y[i][j][c] = sum_val
    
    return Y