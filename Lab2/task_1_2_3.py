import scipy
import scipy.io
import scipy.ndimage
import numpy as np
import cv2
import colorsys
from common import *

normalize = True

np.random.seed(42)

# Load input data
q = scipy.io.loadmat("data/compEx1data.mat")['x']
p1 = q[0,0].T
p2 = q[1,0].T

# Load images
img1 = scipy.ndimage.imread(("data/kronan1.JPG"))
img2 = scipy.ndimage.imread(("data/kronan2.JPG"))
zoom_factor = 0.5

# Prepare normalization matrices
if normalize:
    stdev1 = np.std(p1, axis=0)[0:2]
    stdev2 = np.std(p1, axis=0)[0:2]
    means1 = np.mean(p1, axis=0)[0:2]
    means2 = np.mean(p2, axis=0)[0:2]
    scale1 = 1.41/stdev1
    scale2 = 1.41/stdev2
    trans1 = -means1*scale1
    trans2 = -means2*scale2
    T1 = np.asarray([scale1[0], 0, trans1[0],
                     0, scale1[1], trans1[1],
                     0,         0,         1]).reshape(3,3)
    T2 = np.asarray([scale2[0], 0, trans2[0],
                     0, scale2[1], trans2[1],
                     0,         0,         1]).reshape(3,3)
else:
    T1 = np.identity(3)
    T2 = np.identity(3)

print("T1:")
show(T1)
print("T2:")
show(T2)

p1_normalized = np.einsum('xy,zy->zx', T1, p1)
p2_normalized = np.einsum('xy,zy->zx', T2, p2)

print(p1_normalized.mean(axis=0))
print(p2_normalized.mean(axis=0))
print(p1_normalized.std(axis=0))
print(p2_normalized.std(axis=0))

# Compute fundamental matrix
F, choice = calculate_fundamental(p1_normalized, p2_normalized, 10)

print("This is F:")
show(F)

# Ask openCV to compute, for comparison
F2, _ = cv2.findFundamentalMat(p1_normalized[choice][:,0:2], p2_normalized[choice][:,0:2], cv2.FM_8POINT)
print("F according to openCV")
show(F2)

F = T2.T @ F @ T1
F = F/F[2,2]
print("This is F after denormalizing:")
show(F)

print("|F| = ",)
print(np.linalg.det(F))

testpoints = np.random.choice(p1.shape[0], 20, replace=False)
testpoints = choice

# Draw points and epipolar lines
line_width = int(1.5 / zoom_factor)
circle_radius = int(8 / zoom_factor)
circle_width = int(2/ zoom_factor)
rows, cols, _ = img1.shape
for pt1, pt2 in zip(p1[testpoints], p2[testpoints]):
    color = (np.random.randint(0,255)/255.0, 1.0, 1.0)
    color = tuple(np.array(colorsys.hsv_to_rgb(*color))*255.0)
    # Points on image 2, lines on image 1
    cv2.circle(img2,(int(pt2[0]), int(pt2[1])),circle_radius,color,circle_width)
    a,b,c = F @ pt1
    x1,y1 = 0, int(-c/b)
    x2,y2 = cols, int(-(c+a*cols)/b)
    img2 = cv2.line(img2, (x1,y1), (x2,y2), color,line_width)
    # Points on image 1, lines on image 2
    cv2.circle(img1,(int(pt1[0]), int(pt1[1])),circle_radius,color,circle_width)
    a,b,c = F.T @ pt2
    x1,y1 = 0, int(-c/b)
    x2,y2 = cols, int(-(c+a*cols)/b)
    img1 = cv2.line(img1, (x1,y1), (x2,y2), color,line_width)

img1 = zoom_3ch(img1,zoom_factor)
img2 = zoom_3ch(img2,zoom_factor)

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
