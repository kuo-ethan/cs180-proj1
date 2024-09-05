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
    # demean1 = arr1 - np.mean(arr1)
    # demean2 = arr2 - np.mean(arr2)
    # return np.mean(np.multiply(demean1, demean2)) / (np.sqrt(np.var(arr1) * np.var(arr2)))

def translate(im, tx, ty):
    """
    translate an image `tx` to the right, `ty` down
    zeroes out empty space
    :param image (h, w, c) array
    :param tx (float) pixels to translate right
    :param ty (float) pixels to translate down
    """
    h, w = im.shape[:2]
    mat = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)
    return cv2.warpAffine(im, mat, (w, h))

def align_with_ncc(image, base, D):
    # Apply Canny edge detection
    image_edges = cv2.Canny(image, 100, 155)
    base_edges = cv2.Canny(base, 100, 155)

    # Find the best alignment vector on Canny edge images
    best_score = -float('inf')
    best_x, best_y = None, None
    for x in range(-D, D+1):
        for y in range(-D, D+1):
            # Determine slices of overlap for image and base
            base_overlap, image_overlap = None, None
            if x >= 0 and y >= 0:
                base_overlap = base_edges[(None if y == 0 else y):, (None if x == 0 else x):]
                image_overlap = image_edges[:(None if y == 0 else -y), :(None if x == 0 else -x)]
            elif x >= 0 and y < 0:
                base_overlap = base_edges[:y, (None if x == 0 else x):]
                image_overlap = image_edges[-y:, :(None if x == 0 else -x)]
            elif x < 0 and y >= 0:
                base_overlap = base_edges[(None if y == 0 else y):, :x]
                image_overlap = image_edges[:(None if y == 0 else -y), -x:]
            elif x < 0 and y < 0:
                base_overlap = base_edges[:y, :x]
                image_overlap = image_edges[-y:, -x:]

            # Compute alignment score based only on overlapping regions
            curr_score = normalized_cross_correlation(base_overlap, image_overlap)

            if curr_score > best_score:
                # print(f'{curr_score}: {x}, {y}')
                best_score, best_x, best_y = curr_score, x, y

    # Translate the image based on the displacement vector
    print(best_x, best_y)
    return translate(image, best_x, best_y)
    

def align_with_euclidean(image, base, D):
    # Apply Canny edge detection
    image_edges = cv2.Canny(image, 100, 155)
    base_edges = cv2.Canny(base, 100, 155)

    # Find the best alignment vector
    best_score = float('inf')
    best_x, best_y = None, None
    for x in range(-D, D+1):
        for y in range(-D, D+1):
            # Determine slices of overlap for image and base
            base_overlap, image_overlap = None, None
            if x >= 0 and y >= 0:
                base_overlap = base_edges[(None if y == 0 else y):, (None if x == 0 else x):]
                image_overlap = image_edges[:(None if y == 0 else -y), :(None if x == 0 else -x)]
            elif x >= 0 and y < 0:
                base_overlap = base_edges[:y, (None if x == 0 else x):]
                image_overlap = image_edges[-y:, :(None if x == 0 else -x)]
            elif x < 0 and y >= 0:
                base_overlap = base_edges[(None if y == 0 else y):, :x]
                image_overlap = image_edges[:(None if y == 0 else -y), -x:]
            elif x < 0 and y < 0:
                base_overlap = base_edges[:y, :x]
                image_overlap = image_edges[-y:, -x:]

            # Compute alignment score based only on overlapping regions
            curr_score = euclidean_distance(base_overlap, image_overlap)
            if curr_score < best_score:
                best_score, best_x, best_y = curr_score, x, y

    # Translate the image based on the displacement vector
    print(best_x, best_y)
    return translate(image, best_x, best_y)

def colorize_with_euclidean(r, g, b, D):
    ag = align_with_euclidean(g, b, D)
    ar = align_with_euclidean(r, b, D)
    return np.dstack([ar, ag, b])
    
def colorize_with_ncc(r, g, b, D):
    ag = align_with_ncc(g, b, D)
    ar = align_with_ncc(r, b, D)
    return np.dstack([ar, ag, b])