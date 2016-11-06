import scipy
import scipy.io
import scipy.ndimage
import numpy as np
import cv2
from common import *

MODE_FUNDAMENTAL = 1
MODE_FUNDAMENTAL_NORMALIZED = 2
MODE_ESSENTIAL = 3

mode = MODE_FUNDAMENTAL_NORMALIZED
pick_n_points = 15

np.random.seed(42)

# Load input data
q = scipy.io.loadmat("data/compEx1data.mat")['x']
p1 = q[0,0].T
p2 = q[1,0].T

# Load images
img1 = scipy.ndimage.imread(("data/kronan1.JPG"))
img2 = scipy.ndimage.imread(("data/kronan2.JPG"))
zoom_factor = 0.5

if mode == MODE_FUNDAMENTAL_NORMALIZED:
    T1, T2 = prepare_normalization_transforms(p1,p2)
    
    print("T1:")
    show(T1)
    print("T2:")
    show(T2)
    
    p1_normalized = np.einsum('xy,zy->zx', T1, p1)
    p2_normalized = np.einsum('xy,zy->zx', T2, p2)
    
    # Compute fundamental matrix
    F, choice = calculate_fundamental(p1_normalized, p2_normalized, pick_n_points)
    print("This is F:")
    show(F)

    # Ask openCV to compute on the same points, for comparison
    F2, _ = cv2.findFundamentalMat(p1_normalized[choice][:,0:2], p2_normalized[choice][:,0:2], cv2.FM_8POINT)
    print("F according to openCV")
    show(F2)

    
    F = T2.T @ F @ T1
    F = F/F[2,2]
    print("This is F after denormalizing:")
    show(F)
    
elif mode == MODE_FUNDAMENTAL:
    # Compute fundamental matrix
    F, choice = calculate_fundamental(p1, p2, pick_n_points)

    print("This is F:")
    show(F)
    
    # Ask openCV to compute on the same points, for comparison
    F2, _ = cv2.findFundamentalMat(p1[choice][:,0:2], p2[choice][:,0:2], cv2.FM_8POINT)
    print("F according to openCV")
    show(F2)

elif mode == MODE_ESSENTIAL:
    pass

testpoints = np.random.choice(p1.shape[0], 20, replace=False)
if pick_n_points is not None:
    testpoints = choice

img1, img2 = draw_points_and_epipolar(img1, img2, p1[testpoints], p2[testpoints], F, zoom_factor)

img1 = zoom_3ch(img1,zoom_factor)
img2 = zoom_3ch(img2,zoom_factor)

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
