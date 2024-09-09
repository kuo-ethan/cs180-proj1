# Utility functions here
import numpy as np
import cv2
from skimage.transform import resize

# =============== Parameters ==================
CANNY_LOWER_THRESHOLD = 100
CANNY_UPPER_THRESHOLD = 155

# =============== Similarity metrics ==================
def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

def normalized_cross_correlation(arr1, arr2):
    normalized1 = arr1 / np.linalg.norm(arr1)
    normalized2 = arr2 / np.linalg.norm(arr2)
    return np.sum(normalized1 * normalized2)

# =============== Helpers ==================
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
# Colorizes channels using exhaustive search.
def naive_colorize(r, g, b, D, similarity_metric=normalized_cross_correlation):
    x, y = align_exhaustive(g, b, D, similarity_metric)
    ag = translate(g, x, y)
    x, y = align_exhaustive(r, b, D, similarity_metric)
    ar = translate(r, x, y)
    return np.dstack([ar, ag, b])

# Colorizes channels using image pyramid search.
def pyramid_colorize(r, g, b, D, similarity_metric=normalized_cross_correlation):
    x, y = align_with_pyramid(g, b, D, similarity_metric)
    ag = translate(g, x, y)
    x, y = align_with_pyramid(r, b, D, similarity_metric)
    ar = translate(r, x, y)
    return np.dstack([ar, ag, b])

# Finds the best displacement vector using exhaustive search.
def align_exhaustive(image, base, D, similarity_metric, start_x=0, start_y=0):

    # Apply Canny edge detection
    image_edges = cv2.Canny(image, CANNY_LOWER_THRESHOLD, CANNY_UPPER_THRESHOLD)
    base_edges = cv2.Canny(base, CANNY_LOWER_THRESHOLD, CANNY_UPPER_THRESHOLD)

    # Find the best alignment vector on Canny edge images
    best_score = -float('inf')
    best_x, best_y = None, None
    for x in range(start_x - D, start_x + D + 1):
        for y in range(start_y - D, start_y + D + 1):
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

            if curr_score > best_score:
                best_score, best_x, best_y = curr_score, x, y
    print(best_x, best_y)
    return best_x, best_y

# Finds the best displacement vector using image pyramid technique.
def align_with_pyramid(image, base, D, similarity_metric, levels=4):
    last_level = levels - 1

    # Returns a pyramid for the image, from high to low resolution.
    def build_pyramid(image, levels):
        pyramid = [image]
        for i in range(1, levels):
            im = pyramid[i-1] / 255
            downscaled_image = resize(im, (im.shape[0] / 2, im.shape[1] / 2))
            pyramid.append((downscaled_image * 255).astype(np.uint8))
        return pyramid
    
    image_pyramid = build_pyramid(image, levels)
    base_pyramid = build_pyramid(base, levels)

    # Finds the best displacement vector for a level in the pyramid.
    def align_level(level, d):
        if level == last_level:
            return align_exhaustive(image_pyramid[level], base_pyramid[level], d, similarity_metric)
        prev_x, prev_y = align_level(level+1, d*2)
        return align_exhaustive(image_pyramid[level], base_pyramid[level], d, similarity_metric, prev_x*2, prev_y*2)

    best_x, best_y = align_level(0, D)
    print(best_x, best_y)
    return best_x, best_y
