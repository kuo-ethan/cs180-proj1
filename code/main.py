# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import sys
from lib import naive_colorize, pyramid_colorize

imname = None
D = None
curr_extension = None

extension = {
    'cathedral': 'jpg',
    'church': 'tif',
    'emir': 'tif',
    'harvesters': 'tif',
    'icon': 'tif',
    'lady': 'tif',
    'melons': 'tif',
    'monastery': 'jpg',
    'onion_church': 'tif',
    'sculpture': 'tif',
    'self_portrait': 'tif',
    'three_generations': 'tif',
    'tobolsk': 'jpg',
    'train': 'tif',
}

# Validate command line inputs
if len(sys.argv) < 2 or len(sys.arv) > 3:
    print("Error: You need to pass 1 or 2 arguments.")
    print("Usage: python3 main.py <image_name> [displacement_factor]")
    print("Example: python3 main.py cathedral.jpg 30")
    sys.exit(1)

# Parse inputs
curr_extension = extension[sys.argv[1]]
imname = f'../data/{sys.argv[1]}.{curr_extension}'
if len(sys.argv) == 2:
    # Use default displacement search range
    D = 16
else:
    D = int(sys.argv[2])

# Read in the image (as a uint8) and separate color channels (in grayscale)
im = skio.imread(imname)
if (im.dtype == np.uint16):
   im = (im / 256).astype(np.uint8)

height = np.floor(im.shape[0] / 3.0).astype(int)
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# Align the channels and create color images
im_out = None
if curr_extension == 'jpg':
    im_out = naive_colorize(r, g, b, D)
else:
    im_out = pyramid_colorize(r, g, b)

# Create, save, and display the color image
fname_ncc = f'../images/ncc_{sys.argv[1]}.jpg'
skio.imsave(fname_ncc, im_out)