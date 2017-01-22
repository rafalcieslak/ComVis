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
parser.add_argument('image1', metavar="IMG1", type=str, help="Path to image 1.")
parser.add_argument('image2', metavar="IMG2", type=str, help="Path to image 2.")
parser.add_argument('IMAGES', metavar="IMG", nargs="*", default=[], type=str, help="Subsequent images.")
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
P_TEST_POINTS = 20
K = np.array([[2759.48, 0.00000, 1520.69],
              [0.00000, 2764.16, 1006.81],
              [0.00000, 0.00000, 1.00000]])

image_files = [args.image1, args.image2] + args.IMAGES

# Open image files
print("Opening image files...")
images = []
for image_file in image_files:
    image = scipy.ndimage.imread(image_file)[:,:,0:3]
    image = zoom_3ch(image,SCALE)
    images += [image]

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


described = [describe_cached(image, image_file) for image, image_file in zip(images, image_files)]
    
# Start with first two images.
image1 = images[0]
image2 = images[1]
described1 = described[0]
described2 = described[1]

print("Matching first two images.")
matches = match2D(described1, described2, MATCHES)
matches = np.array(matches)

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
print("Pi1:")
show(Pi1)

"""
# Pick some points to test matrices.
testmatches = matches[np.random.randint(0, matches.shape[0], P_TEST_POINTS)]
tm1 = np.vstack([testmatches[:,0], testmatches[:,1], [1]*P_TEST_POINTS]).T
tm2 = np.vstack([testmatches[:,2], testmatches[:,3], [1]*P_TEST_POINTS]).T

print(tm1)

for i in range(0,4):
    totalz1 = totalz2 = 0;
    for tp1, tp2 in zip(tm1, tm2):
        testP1, testP2 = K @ Pi1, K @ Pi2l[i]
        X = triangulate(testP1, testP2, tp1, tp2)
        x1 = testP2 @ X
        x2 = testP1 @ X
        totalz1 += np.sign(x1[2])
        totalz2 += np.sign(x2[2])
    print("DONE")
    print(totalz1)
    print(totalz2)
    if totalz1 > 0 and totalz2 > 0:
        print("Picking P2 variant #%d" % i)
        print("Pi2:")
        show(Pi2l[i])
        P1,P2 = testP1,testP2
        break
    if totalz1 < 0 and totalz2 < 0:
        print("Picking P2 variant -1* #%d" % i)
        testP2 = -testP2
        print("Pi2:")
        show(Pi2l[i])
        P1,P2 = testP1,testP2
        break
    if i == 3:
        print("ERROR: No Ps derived from E are valid!")
"""

# For some reason, the above doesn't work very well. Instead we'll just pick a
# martix that has positive scale on all axes. This may flip the orientation of
# the model, but whaterver, who cares.
for i in range(0,4):
    testP1, testP2 = Pi1, Pi2l[i]
    if (np.sign(testP2[0,0]) == np.sign(testP2[1,1]) and
        np.sign(testP2[1,1]) == np.sign(testP2[2,2])):
        testP2 *= np.sign(testP2[0,0])
        print("Picking P2 variant #%d" % i)
        print("Pi2:")
        show(testP2)
        P1,P2 = K @ testP1, K @ testP2
        break             
    if i == 3:
        print("ERROR: No Ps derived from E are valid!")

P_matrices = {0: P1, 1: P2}
Pi_matrices = {0: testP1, 1: testP2}

print("Initializing pointcould...")
results = np.array([triangulate(P1,P2,p1_,p2_) for p1_, p2_ in zip(p1,p2)])
# Convert results to a pointcloud
pointcloud = []
for i in range(0, matches.shape[0]):
    X = results[i]
    match = matches[i]
    matchlist = [(0, match[0], match[1]), (1, match[2], match[3])]
    pointcloud += [(X, matchlist)]

# TODO: subsequent images.
for i in range(2, len(image_files)):
    print("Now adding image", image_files[i])

    # Start by preparing matches between this and previous image.
    print("Matching with previous image.")
    matches = match2D(described[i-1], described[i], MATCHES)
    matches = np.array(matches)
    p1 = pointlist_to_homog(matches[:,0:2])
    p2 = pointlist_to_homog(matches[:,2:4])
    p1_normalized = np.einsum('xy,zy->zx', K_inv, p1)
    p2_normalized = np.einsum('xy,zy->zx', K_inv, p2)

    # Now search for a good Pi.
    
print("Total points in pointcloud:", len(pointcloud))    
print("Extracting data from pointcloud...")
results = []
for (x,y,z), matchlist in pointcloud:
    if len(matchlist) == 0:
        print("ERROR: Empty matchlist")
        results += [(x,y,z,0,0,0)]
        continue
    avg = np.array([0,0,0])
    for match in matchlist:
        avg += images[match[0]][int(match[2]), int(match[1])]
    r,g,b = avg/len(matchlist)
    results += [(x,y,z,r,g,b)]

print("Saving results...")
save_to_ply(results, "out.ply")
