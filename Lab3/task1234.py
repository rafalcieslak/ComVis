import scipy
import scipy.io
import scipy.ndimage
import numpy as np
import cv2
import colorsys
from common.show import *
from common.image import *
from common.matrices import *

show_images = True
show_plot = False
write_obj = False

mat = scipy.io.loadmat("data/Sport_cam.mat")
Po1 = mat['pml']
Po2 = mat['pmr']

img1 = scipy.ndimage.imread("data/Sport0.png")/255
img2 = scipy.ndimage.imread("data/Sport1.png")/255
img1 = cv2.bilateralFilter(img1.astype(np.float32),2,1,1)
img2 = cv2.bilateralFilter(img2.astype(np.float32),2,1,1)

# Rectify
imgcenter = np.array([img1.shape[1], img1.shape[0], 1])
H1, H2, P = rectify_stereo(Po1, Po2, imgcenter)

# Prepare bounding box
BB1 = image_bounds_homography(img1, H1)
BB2 = image_bounds_homography(img1, H2)
BB = pointlist_from_homog(np.vstack([BB1,BB2]))
size, offset = image_bounds_factorize(BB)
size = (size[0], size[1], img1.shape[2])

print(size)
print(offset)

img1_rect = img_transform_H(img1, H1, size, offset=offset)
img2_rect = img_transform_H(img2, H2, size, offset=(offset + np.array([0, 0])))

# Ask openCV about disparities.

dispar_min = 50
dispar_max = 160
window = 11

left  = np.uint8(256*img1_rect)
right = np.uint8(256*img2_rect)
sbm = cv2.StereoSGBM_create(dispar_min,dispar_max,window,2000,8000)
disparity = sbm.compute(left,right)/16.0
show(disparity)

dmin, dmax = disparity.min(), disparity.max()
print((dmin,dmax))
drange = dmax - dmin
dnorm = (disparity - dmin)/drange

# Draw a mesh on rectified images
cx,cy = np.meshgrid(np.arange(0, left.shape[1], 40), np.arange(0, left.shape[0], 40))
coords_sparse = np.fliplr(np.stack((cx,cy), axis=2).reshape((-1,2), order='F'))
res_sparse = scipy.ndimage.map_coordinates(disparity, coords_sparse.T, cval=-1)

right_dots = right.copy()
left_dots = left.copy()
circle_radius = 2
circle_width = -1
for (y,x), dis in zip(coords_sparse, res_sparse):
    if dis < dispar_min:
        continue
    color = (np.random.randint(0,255)/255.0, 1.0, 1.0)
    color = tuple(np.array(colorsys.hsv_to_rgb(*color))*255.0)
    right_dots = cv2.circle(right_dots,(int(x-dis), int(y)),circle_radius,color,circle_width)
    left_dots = cv2.circle(left_dots,(int(x), int(y)),circle_radius,color,circle_width)

# Scatter plot 3d points
STEP = 8
cx,cy = np.meshgrid(np.arange(0, left.shape[1], STEP), np.arange(0, left.shape[0], STEP))
coords_dense = np.fliplr(np.stack((cx,cy), axis=2).reshape((-1,2), order='F'))
res_dense = scipy.ndimage.map_coordinates(disparity, coords_dense.T, cval=-1)
baseline = baseline_length(Po1,Po2)
pts = []
K, R, t, c = decompose_P(P)
coords_dense = np.einsum('ba,na->nb', np.linalg.inv(K), pointlist_to_homog(coords_dense))
coords_dense = pointlist_from_homog(coords_dense)
for (y,x), dis in zip(coords_dense, res_dense):
    if dis < dispar_min:
        continue
    z = baseline/dis
    pts.append([10*x,10*y,z])

# =============================

STEP = 1
"""
with open('out.obj', 'w') as f:
    f.write("# output\n")
    for (y,x), dis in zip(coords_megadense, res_megadense):
        z = baseline/dis
        f.write("v %f %f %f\n" % (x,y,z))
"""
vcnt = 1
v = []
vt = []
f = []
shape = left.shape
for y in range(0, shape[0]-1, STEP):
    for x in range(0, shape[1]-1, STEP):
        if x + STEP >= shape[1] or y + STEP >= shape[0]:
            continue
        # Emit 4 vertices
        skip = False
        for dx in [0, STEP]:
            for dy in [0, STEP]:
                disp = disparity[y+dy,x+dx]
                if disp <= dispar_min:
                    skip = True
                z = baseline/disp
                v.append("v %f %f %f" % (-(x+dx)/20, -(y+dy)/20, z*8))
                vt.append("vt %f %f" % ((x+dx)/float(shape[1]), -(y+dy)/float(shape[0])))
        # Emit at most 2 faces
        if not skip:
            f.append("f %d/%d %d/%d %d/%d" % (vcnt, vcnt, vcnt+1, vcnt+1, vcnt+2, vcnt+2))
            f.append("f %d/%d %d/%d %d/%d" % (vcnt+3, vcnt+3, vcnt+2, vcnt+2, vcnt+1, vcnt+1))
        vcnt += 4

with open('out.obj', 'w') as out:
    out.write("# output\n")
    out.write("mtllib out.mtl\n")
    out.write("usemtl data\n")
    out.write("\n".join(v))
    out.write("\n")
    out.write("\n".join(vt))
    out.write("\n")
    out.write("\n".join(f))

# =============================

cv2.imwrite(("left.png"),cv2_shuffle(left))

if show_images:
    cv2.imshow('left',cv2_shuffle(left_dots))
    cv2.imshow('right',cv2_shuffle(right_dots))
    cv2.imshow('dispar',dnorm)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if show_plot:
    plot_points(np.asarray(pts))
