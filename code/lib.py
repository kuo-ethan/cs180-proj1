# Utility functions here
import numpy as np

def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

def normalized_cross_correlation(arr1, arr2):
    normalized1 = arr1 / np.linalg.norm(arr1)
    normalized2 = arr2 / np.linalg.norm(arr2)
    return np.sum(normalized1 * normalized2)