# Utility functions here

# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

import numpy as np
import cv2

def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

def normalized_cross_correlation(arr1, arr2):
    normalized1 = arr1 / np.linalg.norm(arr1)
    normalized2 = arr2 / np.linalg.norm(arr2)
    return np.sum(normalized1 * normalized2)

def translate(im, tx, ty):
    """
    translate an image `tx` to the right, `ty` down
    :param image (h, w, c) array
    :param tx (float) pixels to translate right
    :param ty (float) pixels to translate down
    """
    h, w = im.shape[:2]
    mat = translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)
    print (mat)
    return cv2.warpAffine(im, mat, (w, h))

def align_with_ncc(image, base, displacement_factor):
    H, W = base.shape

    # Find the optimal displacement vector.
    # This involves slicing the image and base for the actual overlap.
    best_score = -float.inf()
    best_x, best_y = None, None
    for x in range(-15, 16):
        for y in range(-15, 16):
            image_overlap = image[:-y, :-x]
            base_overlap = base[y:, x:]
            curr_score = normalized_cross_correlation(image_overlap, base_overlap)
            if curr_score > best_score:
                best_score, best_x, best_y = curr_score, x, y

    return translate(image, x, y), best_x, best_y
    

def align_with_euclidean(image, base, displacement_factor):
    pass

def colorize_with_euclidean(r, g, b, displacement_factor):
    ag = align_with_euclidean(g, b, displacement_factor)
    ar = align_with_euclidean(r, b, displacement_factor)
    return np.dstack([ar, ag, b]) * 255
    
def colorize_with_ncc(r, g, b, displacement_factor):
    ag = align_with_ncc(g, b, displacement_factor)
    ar = align_with_ncc(r, b, displacement_factor)
    return np.dstack([ar, ag, b]) * 255