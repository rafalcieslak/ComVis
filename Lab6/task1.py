import scipy
import numpy as np
import cv2
import argparse
import sys
import os.path
from common.show import *
from common.image import *
from common.SIFT import *
from common.matrices import *

parser = argparse.ArgumentParser(description='Lab 6 task 1.')
parser.add_argument('image1', metavar="IMAGE1", type=str, help="Path to image 1.")
parser.add_argument('image2', metavar="IMAGE2", type=str, help="Path to image 2.")
parser.add_argument('--dbg', '-d', action='store_true', help="Debug mode.")
    
try:
    args = parser.parse_args()
except SystemExit:
    sys.exit(0)

TAKE = 800
MATCHES = 500
SCALE = 1.0
RANSAC_TRESHOLD = 5
RANSAC_SAMPLES = 1500
show_epipolar = False
K = np.array([[2759.48, 0.00000, 1520.69],
              [0.00000, 2764.16, 1006.81],
              [0.00000, 0.00000, 1.00000]])

# Open image files
image1 = scipy.ndimage.imread(args.image1)[:,:,0:3]
image2 = scipy.ndimage.imread(args.image2)[:,:,0:3]
image1 = zoom_3ch(image1,SCALE)
image2 = zoom_3ch(image2,SCALE)

def line_point_distance(a,b,c,x,y):
    n = np.abs(a*x + b*y + c)
    d = np.sqrt(a*a + b*b)
    return n/d

def features(image):
    I = rgb2Y(image)
    I = I / I.max()
    points = harris_laplace(I)
    print("Total %d features" % len(points))
    points = ANMS(points, TAKE)
    print("Using %d features" % len(points))
    return np.array(points)[:,[0,1,3]]

def features_cached(image, name):
    filename = os.path.split(name)[1]
    print("  Looking for feature cache for %s" % filename)
    basename = filename.split('.')[0:-1]
    cachefile = '.'.join(basename) + ".feat"
    if(os.path.isfile(cachefile)):
        print("  Loading feature data from cache %s" % cachefile)
        return np.loadtxt(cachefile)
    else:
        print("  Computing feature data for image %s" % filename)
        feat = features(image)
        np.savetxt(cachefile, feat, '%d %d %f')
        return feat

def describe_cached(image, name):
    filename = os.path.split(name)[1]
    print("Looking for descriptor cache for %s" % filename)
    basename = filename.split('.')[0:-1]
    cachefile = '.'.join(basename) + ".descrip"
    if(os.path.isfile(cachefile)):
        print("Loading descriptor data from cache %s" % cachefile)
        return np.loadtxt(cachefile)
    else:
        features = features_cached(image, name)
        print("Computing descriptor data for image %s" % filename)
        descr = describe(image, features)
        print("Done.")
        np.savetxt(cachefile, descr)
        return descr

described1 = describe_cached(image1, args.image1)
described2 = describe_cached(image2, args.image2)

print("Matching")
matches = match2D(described1, described2, MATCHES)
matches = np.array(matches)
# print(matches)

K_inv = np.linalg.inv(K)

print("Using K:")
show(K)

p1 = pointlist_to_homog(matches[:,0:2])
p2 = pointlist_to_homog(matches[:,2:4])

p1_normalized = np.einsum('xy,zy->zx', K_inv, p1)
p2_normalized = np.einsum('xy,zy->zx', K_inv, p2)

print("Running RANSAC search for best E.")
best_inliers = 0
for i in range(0, RANSAC_SAMPLES):
    # Pick 8 indices at random and compute fundamental matrix
    choice = np.random.choice(np.arange(p1.shape[0]), 10, replace=False)
    E = calculate_fundamental(p1_normalized[choice], p2_normalized[choice], essential=True)
    F = K_inv.T @ E @ K_inv
    F = F/F[2,2]

    # Count outliers.
    inliers = []
    for pt1, pt2 in zip(p1, p2):
        a,b,c = F @ pt1
        d1 = line_point_distance(a,b,c, pt2[0],pt2[1])
        a,b,c = F.T @ pt2
        d2 = line_point_distance(a,b,c, pt1[0],pt1[1])
        q = d1 < RANSAC_TRESHOLD and d2 < RANSAC_TRESHOLD
        inliers += [q]
    inliers_mask = np.array(inliers)
    inliers = inliers_mask.sum()
    #print("Inliers: ", inliers)

    if inliers > best_inliers:
        best_inliers = inliers
        best_E_F = (E,F)
        best_mask = inliers_mask

E, F = best_E_F
inliers_mask = best_mask
print("Best E has %d/%d inliers." % (best_inliers, p1.shape[0]))
#   Uncomment this to calculate E again, usign ALL inliers.
#E = calculate_fundamental(p1_normalized[inliers_mask], p2_normalized[inliers_mask], essential=True)
#F = K_inv.T @ E @ K_inv
#F = F/F[2,2]

print("This is the best E:")
show(E)

# Remove outliers from matches set.
matches = matches[inliers_mask]
print("Using %d matches." % matches.shape[0])

p1 = pointlist_to_homog(matches[:,0:2])
p2 = pointlist_to_homog(matches[:,2:4])

if show_epipolar:
    choice = np.random.choice(np.arange(p1.shape[0]), 25, replace=False)
    zoom_factor = 0.3
    img1, img2 = draw_points_and_epipolar(image1.copy(), image2.copy(), p1[choice], p2[choice], F, zoom_factor)
    img1 = zoom_3ch(img1,zoom_factor)
    img2 = zoom_3ch(img2,zoom_factor)
    cv2.imshow('img1',cv2_shuffle(img1))
    cv2.imshow('img2',cv2_shuffle(img2))
    # Wait 20ms for Esc key, repeat.
    while cv2.waitKey(20) & 0xff != 27:
        pass

# Triangulate matches from first two images.
Pi1, Pi2l = get_Ps_from_E(E)
Pi2 = Pi2l[0] # Assume cameras facing same direction. Implicitly assume spatially ordered image sequence.
print("Pi1:")
show(Pi1)
print("Pi2:")
show(Pi2)

P1 = K @ Pi1
P2 = K @ Pi2

results = np.array([triangulate(P1, P2,p1_,p2_) for p1_, p2_ in zip(p1,p2)])
results = [(x/w,y/w,z/w) for x,y,z,w in results]

print("Saving results...")
save_to_ply(results, "out.ply")
