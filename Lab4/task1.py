import scipy
import numpy as np
import cv2
from common.show import *
from common.image import *

image = scipy.ndimage.imread("data/test.png")[:,:,0:3]

I = rgb2Y(image)
print(I.shape)

SIGMA1 = 1
SIGMA2 = 4
alpha = 0.05
local_maxima_neigh_size = 5
maxima_treshold = 400000
flat_treshold = 0.01

R = harris_corner_response(I, alpha, SIGMA1, SIGMA2)

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

def rescale(img):
    min, max = img.min(), img.max()
    return (img-min)/(max-min)

R2, R2l = rescale(R), np.zeros_like(R)
R2l[maxima > 0] = 1
R2 = combine_channels(R2, np.zeros_like(R2), R2l)


cv2.imshow('I', I)
cv2.imshow('R', rescale(R))
cv2.imshow('R2', R2)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
