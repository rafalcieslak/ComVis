import scipy
import numpy as np
import cv2
from common.show import *
from common.image import *

# Dataset selection
dataset = "nd"
# If false, basic harris corner detector will be in use
use_harris_laplace = True
# This sigma is only meaningful for basic harris mode
harris_sigma = 4

# Datasets definition
data = {
    "test": {
        "file": "data/test.png",
        "take": 30
    },
    "nd": {
        "file": "data/Notre Dame/1_o.jpg",
        "zoom": 0.4,
        "take": 400
    },
    "nd2": {
        "file": "data/Notre Dame/2_o.jpg",
        "zoom": 0.4,
        "take": 400
    },
    "eg": {
        "file": "data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg",
        "zoom": 0.5,
        "take": 400
    },
    "eg2": {
        "file": "data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg",
        "zoom": 0.7,
        "take": 400
    },
}[dataset]

# Load image data
image = scipy.ndimage.imread(data["file"])[:,:,0:3]
zoom = data["zoom"] if "zoom" in data else 1.0
image = scipy.ndimage.zoom(image, (zoom,zoom,1))
# Convert image to intensity
I = rgb2Y(image)
# Normalize the image to [0,1]
I = I / I.max()

# Find features
if use_harris_laplace:
    points = harris_laplace(I)
else:
    points, R = harris_corners(I, harris_sigma)

print("Total %d features" % len(points))

# Use adaptive non-maximal suppression to take some best sparesly placed features
# By default, take 200 features
take = data["take"] if "take" in data else 200
points = ANMS(points, take)
# Alternatively, apply simple treshold
# points = [(x,y,r,s) for x,y,r,s in points if r > maxima_treshold]

print("Using %d features" % len(points))

# Mark features on the image
image2 = image.copy()
for x,y,_,s in points:
    image2 = cv2.circle(image2, (int(x),int(y)), int(2.5*s), (255,0,0))

# Display response function (only when using basic harris corners mode)
if not use_harris_laplace:
    R2, R2l = rescale(R), np.zeros_like(R)
    for x,y,_,s in points:
        R2l = cv2.circle(R2l, (int(x),int(y)), 0, (255,0,0))
    R2 = combine_channels(R2, np.zeros_like(R2), R2l)
    cv2.imshow('R2', R2)

cv2.imshow('image', cv2_shuffle(image2))

# Wait 20ms for Esc key, repeat.
while cv2.waitKey(20) & 0xff != 27:
    pass
