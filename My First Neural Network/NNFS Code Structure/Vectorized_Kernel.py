import numpy as np


def createPattern(kernelSize, input_Num_col):
    Temp_pattern = []
    for K in range(kernelSize):
        for i in range(kernelSize):
            Temp_pattern.append(1)
        if K != kernelSize - 1:
            for i in range(input_Num_col - kernelSize):
                Temp_pattern.append(0)
    return Temp_pattern


def createVectorMatrix(kernelSize, input_Num_row, input_Num_col):
    vectorMatrix = []
    pattern = createPattern(kernelSize, input_Num_col)
    for K in range(input_Num_row - kernelSize + 1):
        for L in range(input_Num_col - kernelSize + 1):
            row = []
            for k in range(K):
                for j in range(input_Num_col):
                    row.append(0)
            for i in range(L):
                row.append(0)
            for i in pattern:
                row.append(i)
            totalLength = input_Num_col * input_Num_row
            for i in range(len(row), totalLength):
                row.append(0)
            vectorMatrix.append(row)
    return np.array(vectorMatrix).T


def create_array(kernel_size, inputNum):
    arr = np.ones(kernel_size, dtype=int)
    remaining_rows = inputNum - kernel_size
    zero_rows = np.zeros(remaining_rows, dtype=int)
    result = np.concatenate((arr, zero_rows))
    A = np.array([np.roll(result, i) for i in range(inputNum - kernel_size + 1)]).T
    return A


def make_vectorKernel(kernelSize, num_rows, num_cols):
    left = create_array(kernelSize, num_rows)
    right = create_array(kernelSize, num_cols)
    return np.kron(left, right)


vec = make_vectorKernel(5, 700, 100)
print(vec.shape)



