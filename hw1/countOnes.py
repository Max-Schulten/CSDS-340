import numpy as np

def countOnesLoop(arr):
    counter = 0
    for i in arr:
        for j in i:
            counter += 1 if j == 1 else 0
    return counter

def countOnesWhere(arr):
    return len(np.where(arr == 1)[0])
