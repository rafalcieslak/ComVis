import scipy
import numpy as np
import cv2
from common.show import *
from common.image import *

dataset = "eg"

data = {
    "test": {
        "file": "data/test.png",
        "maxima_treshold": 6,
        "disable-anms": True
    },
    "nd": {
        "file": "data/Notre Dame/1_o.jpg",
        "zoom": 0.4,
        "take": 100
    },
    "eg": {
        "file": "data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg",
        "zoom": 0.4,
        "take": 200
    },
}

data = data[dataset]
fname = data["file"]

image = scipy.ndimage.imread(fname)[:,:,0:3]
zoom = data["zoom"] if "zoom" in data else 1.0
image = scipy.ndimage.zoom(image, (zoom,zoom,1))
# Convert image to intensity
I = rgb2Y(image)
# Normalize the image to [0,1]
I = I / I.max()

SIGMA1 = 1.2
SIGMA2 = 4
alpha = 0.05
maxima_treshold = data["maxima_treshold"] if "maxima_treshold" in data else 0.3
take = data["take"] if "take" in data else 100

R = harris_corner_response(I, alpha, SIGMA1, SIGMA2)*(255*255)

points = find_local_maximas_with_values(R)

if "disable-anms" in data:
    # Simple treshold
    points = [(x,y,r) for x,y,r in points if r > maxima_treshold]
else:
    points = ANMS(points, take)

print("Using %d corners" % len(points))

R2, R2l = rescale(R), np.zeros_like(R)

image2 = image.copy()
# Mark features
for x,y,_ in points:
    image2 = cv2.circle(image2, (int(x),int(y)), 4, (255,0,0))
    R2l = cv2.circle(R2l, (int(x),int(y)), 0, (255,0,0))

R2 = combine_channels(R2, np.zeros_like(R2), R2l)
cv2.imshow('image', cv2_shuffle(image2))
cv2.imshow('R2', R2)

while cv2.waitKey(5) & 0xff != 27:
    pass
cv2.destroyAllWindows()
