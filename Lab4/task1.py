import scipy
import numpy as np
import cv2
from common.show import *
from common.image import *

dataset = "nd"

data = {
    "test": {
        "file": "data/test.png",
        "maxima_treshold": 6,
    },
    "nd": {
        "file": "data/Notre Dame/1_o.jpg",
        "zoom": 0.6,
        "maxima_treshold": 0.5,
    }
}

data = data[dataset]
fname = data["file"]

image = scipy.ndimage.imread(fname)[:,:,0:3]
zoom = data["zoom"] if "zoom" in data else 1.0
image = scipy.ndimage.zoom(image, (zoom,zoom,1))

I = rgb2Y(image)
# Normalize the image to [0,1]
I = I / I.max()
print(I.shape)

SIGMA1 = 1.2
SIGMA2 = 4
alpha = 0.05
local_maxima_neigh_size = 5
maxima_treshold = data["maxima_treshold"] if "maxima_treshold" in data else 400000
flat_treshold = 0.01

R = harris_corner_response(I, alpha, SIGMA1, SIGMA2)*(255*255)

# Find local maximas
R_max = scipy.ndimage.maximum_filter(R, local_maxima_neigh_size)
maxima = (R == R_max)
# Apply treshold
maxima[R < maxima_treshold] = False
# Clean flat regions
R_min = scipy.ndimage.minimum_filter(R, local_maxima_neigh_size)
d = ((R_max - R_min) > flat_treshold)
maxima[d == 0] = False

labeled_R, n_obj = scipy.ndimage.label(maxima)
slices = scipy.ndimage.find_objects(labeled_R)
print("Found %d corners" % n_obj)

R2, R2l = rescale(R), np.zeros_like(R)
R2l[maxima > 0] = 1
R2 = combine_channels(R2, np.zeros_like(R2), R2l)


cv2.imshow('I', I)
cv2.imshow('R', rescale(R))
cv2.imshow('R2', R2)

while cv2.waitKey(5) & 0xff != 27:
    pass
cv2.destroyAllWindows()
