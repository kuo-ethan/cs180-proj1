# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import sys
from lib import colorize_with_euclidean, colorize_with_ncc

imname = None
D = 15

# Parse user inputs
if len(sys.argv) == 2:
    imname = f'../data/{sys.argv[1]}.jpg'
elif len(sys.argv) == 3:
    imname = f'../data/{sys.argv[1]}.jpg'
    D = int(sys.argv[2])
else:
    print("Error: You need to pass 1 or 2 arguments.")
    print("Usage: python3 script_name.py <image_name> [displacement_factor]")
    sys.exit(1)  # Exit the program with a non-zero exit code indicating an error

# Read in the image and separate color channels (in grayscale)
im = skio.imread(imname)
# im = sk.img_as_float(im)

height = np.floor(im.shape[0] / 3.0).astype(int)
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# Align the channels and create color images
im_out_euclidean = colorize_with_euclidean(r, g, b, D).astype('uint8')
im_out_ncc = colorize_with_ncc(r, g, b, D).astype('uint8')

# Create, save, and display the color image
fname_euclidean = f'../images/euclidean_{sys.argv[1]}.jpg'
fname_ncc = f'../images/ncc_{sys.argv[1]}.jpg'
skio.imsave(fname_euclidean, im_out_euclidean)
skio.imsave(fname_ncc, im_out_ncc)
# skio.imshow(im_out_euclidean)
# skio.imshow(im_out_ncc)
# skio.show()