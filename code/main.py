import numpy as np
import skimage.io as skio
import sys
from utils import naive_colorize, pyramid_colorize

extensions = {
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

# Given a file name and displacement factor, generate and safe the colorized image.
def colorize(file, D):
    extension = extensions[file]
    imname = f'../data/{file}.{extension}'
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
    if extension == 'jpg':
        im_out = naive_colorize(r, g, b, D) # Slow but precise
    else:
        im_out = pyramid_colorize(r, g, b, D) # Fast but rough

    # Create, save, and display the color image
    fname = f'../images/{file}.jpg'
    skio.imsave(fname, im_out)

# Validate command line inputs
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Error: You need to pass 1 or 2 arguments.")
    print("Usage: python3 main.py <image_name> [displacement_factor]")
    print("Example: python3 main.py cathedral.jpg 30")
    sys.exit(1)

# Parse inputs
file = sys.argv[1]
if file == 'ALL':
    for file, extension in extensions.items():
        if extension == 'jpg':
            colorize(file, 15)
        else:
            colorize(file, 3)
elif len(sys.argv) == 2:
    if extensions[file] == 'jpg':
        colorize(file, 15)
    else:
        colorize(file, 3)
else:
    colorize(file, int(sys.argv[2]))

