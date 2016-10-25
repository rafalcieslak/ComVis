import numpy as np
import cv2
import scipy.ndimage
from common import *

# Input image id
n = 1
# Mark red dots - undistorted pattern
draw_undistorted = True
# Mark green dots - distorted pattern
draw_distorted = True
# Reshape image to remove distortion effect
undistort_image = True
# Open a display window with the result
demo_output = True

data3d = np.loadtxt("data/task34/Model.txt").reshape((-1,2))
data3d = np.pad(data3d, ((0,0),(0,1)), mode='constant', constant_values=0)

# data2d = np.loadtxt(("data/task34/data%d.txt" % n)).reshape(-1,2)

# Load data from callibrations file. Mr. Zhang, for some reason
# unknown to me, decided to use a shitty confusing file format for
# callibrations. The code below cleans up this mess.
with open("data/task34/Calib.txt", "r") as file:
    cal = file.read().split()[0:7]
cal = map(float,cal)
a,c,b,u0,v0,k1,k2 = cal
calibs_mat = np.loadtxt("data/task34/Calib.txt", skiprows=4).reshape((-1,3)).T
R = calibs_mat[:,(n-1)*4:n*4]
R[0:3,0:3] = R[0:3,0:3].T
show(R)
K = np.array([[a,c,u0],[0,b,v0],[0,0,1]])
show(K)

# Apply extrinsic matrix
proj = project(data3d, R)

# Apply distortion
def distortion(coords):
    q1 = np.power(coords,2).sum(axis=1)
    q2 = np.power(q1,2)
    f = k1*q1 + k2*q2 + 1
    return coords * f[:,np.newaxis].repeat(1, axis=1)
dproj = distortion(proj)

# Apply intrisic matrix
proj = project(proj, K)
dproj = project(dproj, K)

# Load the image
image = scipy.ndimage.imread(("data/task34/CalibIm%d.gif" % n))

# Prepare image undistoring transformation
K_inv = np.linalg.inv(K)
show(K_inv)
def undistort_map(coords):
    imgspace_coords = project(coords, K_inv)
    imgspace_coords = distortion(imgspace_coords)
    coords = project(imgspace_coords, K)
    return coords

# Undistort image
if undistort_image:
    image = arbitrary_image_transform(image, undistort_map)

if draw_undistorted:
    for i,j in proj.astype(int):
        cv2.circle(image,(i,j),2,(0,0,255),-1)
if draw_distorted:
    for i,j in dproj.astype(int):
        cv2.circle(image,(i,j),2,(0,255,0),-1)

#image = cv2.bilateralFilter(image,9,75,75)

# Write output to file
cv2.imwrite(("points%d.out.png" % n), image)

# Show output
if demo_output:
    cv2.imshow('image',image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
