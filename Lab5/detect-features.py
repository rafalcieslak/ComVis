import argparse
import scipy
import numpy as np
import cv2
import sys
from common.show import *
from common.image import *

parser = argparse.ArgumentParser(description='Generate a list of features from a image file.')
parser.add_argument('image_path', metavar="IMAGE", type=str, help="Path to source image file.")
parser.add_argument('output_path', metavar="OUTPUT", type=str,
                    help="Path to output features file. The features will be saved in text format. Each line will contain a single feature, in format: X Y SCALE")
parser.add_argument('--take', metavar="N", type=int, help="Limit the number of fetures to N most confident features.", default=400)

try:
    args = parser.parse_args()
except SystemExit:
    sys.exit(0)

# Open image file
image = scipy.ndimage.imread(args.image_path)[:,:,0:3]

# Convert image to intensity
I = rgb2Y(image)
# Normalize the image to [0,1]
I = I / I.max()

points = harris_laplace(I)
print("Total %d features" % len(points))

# Use adaptive non-maximal suppression to take some best sparsely placed features
points = ANMS(points, args.take)
print("Using %d features" % len(points))

points = np.array(points)[:,[0,1,3]]
np.savetxt(args.output_path, points, '%d %d %f')
