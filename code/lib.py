# Utility functions here

# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

import numpy as np
import cv2
from skimage.transform import rescale

# =============== Similarity metrics ==================

def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

def normalized_cross_correlation(arr1, arr2):
    normalized1 = arr1 / np.linalg.norm(arr1)
    normalized2 = arr2 / np.linalg.norm(arr2)
    return np.sum(normalized1 * normalized2)

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

# ================ ALGORITHMS ======================

# Aligns channels using exhaustive search.
def naive_colorize(r, g, b, D, similarity_metric=normalized_cross_correlation, score_fn=lambda a, b: a > b):
    x, y = align_exhaustive(g, b, D, similarity_metric, score_fn)
    ag = translate(g, x, y)
    x, y = align_exhaustive(r, b, D, similarity_metric, score_fn)
    ar = translate(r, x, y)
    return np.dstack([ar, ag, b])

# Aligns channels using image pyramid technique.
def pyramid_colorize(r, g, b, D=2):
    x, y = align_with_pyramid(g, b, D)
    ag = translate(g, x, y)
    x, y = align_with_pyramid(r, b, D)
    ar = translate(r, x, y)
    return np.dstack([ar, ag, b])

# Finds the best displacement vector using exhaustive search.
def align_exhaustive(image, base, D, similarity_metric, score_fn):

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
            curr_score = similarity_metric(base_overlap, image_overlap)

            if score_fn(curr_score, best_score):
                # print(f'{curr_score}: {x}, {y}')
                best_score, best_x, best_y = curr_score, x, y

    return best_x, best_y

# Finds the best displacement vector using image pyramid technique.
def align_with_pyramid(image, base, D, levels=4):

    # Returns a pyramid for the image, from low to high resolution
    def build_pyramid(image, levels):
        pyramid = [image]
        for i in range(levels-1):
            downscaled_image = rescale(pyramid[i-1], 0.5, anti_aliasing=True, preserve_range=True)
            pyramid.append(downscaled_image)
        pyramid.reverse()
        return pyramid
    
    image_pyramid = build_pyramid(image, levels)
    base_pyramid = build_pyramid(base, levels)

    # Given a level in the pyramid, return the best displacement vector
    def align_level(level, max_displacement):
        if level == 0:
            return align_exhaustive(image_pyramid[0], base_pyramid[0], max_displacement)
        x, y = align_level(level-1, max_displacement * 2)
        
        # Scale up the displacement vector
        # Update align_exhaustive to allow starting displacement



    best_x, best_y = align_level(levels-1, D)
    return translate(image, best_x, best_y)
