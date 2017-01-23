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

TAKE = 900
MATCHES = 600
SCALE = 1.0
RANSAC_TRESHOLD = 5
RANSAC_SAMPLES = 4500
show_epipolar = False
make_last_image_green = False
P_TEST_POINTS = 20
CORRESP_2D_3D = 8
K = np.array([[2759.48, 0.00000, 1520.69],
              [0.00000, 2764.16, 1006.81],
              [0.00000, 0.00000, 1.00000]])

print("Note: Using K = ")
show(K)
K_inv = np.linalg.inv(K)

image_files = [args.image1, args.image2] + args.IMAGES

# Open image files
print("Opening image files...")
images = []
for image_file in image_files:
    image = scipy.ndimage.imread(image_file)[:,:,0:3]
    image = zoom_3ch(image,SCALE)
    images += [image]

if make_last_image_green:
    images[-1][:,:] = np.array([0,255,0])
    
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


# Returns (filtered matches, E).
def ransac_for_E(matches):
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

    # Remove outliers from matches set.
    matches = matches[inliers_mask]
    return matches, E

described = [describe_cached(image, image_file) for image, image_file in zip(images, image_files)]
    
# Start with first two images.
image1 = images[0]
image2 = images[1]
described1 = described[0]
described2 = described[1]

print("Matching first two images.")
matches = match2D(described1, described2, MATCHES)
matches = np.array(matches)

# Figure out a good E with RANSAC, and filter out outlier matches.
matches, E = ransac_for_E(matches)
    
print("This is the best E:")
show(E)

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

# For some reason, testing point direction relative to camera does not work
# well. Instead we'll just pick a martix that has positive scale on all
# axes. This may flip the orientation of the model, but whaterver, who cares.
for i in range(0,4):
    testP1, testP2 = Pi1, Pi2l[i]
    if (np.sign(testP2[0,0]) == np.sign(testP2[1,1]) and
        np.sign(testP2[1,1]) == np.sign(testP2[2,2])):
        q = np.sign(testP2[0,0])
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
    matchlist = [(0, int(match[0]), int(match[1])), (1, int(match[2]), int(match[3]))]
    pointcloud += [[X, matchlist]]

for i in range(2, len(image_files)):
    print("Now adding image", image_files[i])

    # Start by preparing matches between this and previous image.
    print("Matching with previous image.")
    matches = match2D(described[i-1], described[i], MATCHES)
    matches = np.array(matches).astype(np.int)

    print("All matches: ", matches.shape[0])

    matches, E = ransac_for_E(matches)
    print("Filtered matches: ", matches.shape[0])
    
    p1 = pointlist_to_homog(matches[:,0:2])
    p2 = pointlist_to_homog(matches[:,2:4])
    p1_normalized = np.einsum('xy,zy->zx', K_inv, p1)
    p2_normalized = np.einsum('xy,zy->zx', K_inv, p2)
    
    # Create a list of correspondences between 2D points on image i and 3D
    # points in the pointcloud, according to matches.
    matched2D = []
    matched3D = []
    new2D = []
    def matchlist_has_entry(matchlist, i, x, y):
        for m in matchlist:
            if m[0] == i and m[1] == x and m[2] == y:
                return True
        return False
    for xp,yp,xc,yc in matches:
        #print("searching for:", (xp,yp,xc,yc))
        gen = ((q,p) for q,p in enumerate(pointcloud) if matchlist_has_entry(p[1], i-1, xp, yp))
        q,p = next(gen, (-1,None))
        if p is None:
            new2D += [(xp,yp,xc,yc)]
        else:
            #print("found:", p)
            matched2D += [(xc, yc)]
            matched3D += [p[0]]
            # Add to matchlist
            pointcloud[q][1] += [(i, xc, yc)]
    matched2D = np.array(matched2D).astype(np.int)
    matched3D = np.array(matched3D)
    new2D = np.array(new2D).astype(np.int)
    print(matched2D.shape)
    print(matched3D.shape)
    print(new2D.shape)
    
    matched2D_h = pointlist_to_homog(matched2D[:,0:2])
    matched3D_h = pointlist_to_homog(matched3D[:,0:3])
    matched2D_normalized = np.einsum('xy,zy->zx', K_inv, matched2D_h)
    matched2D_normalized = pointlist_from_homog(matched2D_normalized)

    best_r = 1000
    best_inliers = 0
    for n in range(0,1000):
        # Choose randomly some 2d-3d correspondences.
        choice = np.random.choice(np.arange(len(matched2D)), CORRESP_2D_3D, replace=False)
        Pinew = estimate_P(matched2D_normalized[choice], matched3D[choice])
    
        r = calculate_residual(matched2D_normalized[choice], matched3D[choice], Pinew)

        # Use this P to reproject all points.
        proj = project(matched3D, Pinew)
        # Calculate distances
        d = np.sqrt(np.power(proj - matched2D_normalized,2).sum(axis=1))
        inliers_mask = d < 0.001
        inliers = inliers_mask.sum()
    
        if inliers > best_inliers:
            best_inliers = inliers
            best_Pi = Pinew
            best_inliers_mask = inliers_mask
            best_choice = choice
    print("Most inliers:", best_inliers)
    show(best_Pi)
    inliers_mask = best_inliers_mask
    
    print("Best P has %d/%d inliers." % (inliers_mask.sum(), matched3D.shape[0]))

    #det = np.linalg.det(best_P[0:3,0:3])
    #print(det)
    #best_P /= det

    best_Pi /= -best_Pi[0,0]
    
    #K, _, _, _ = decompose_P(best_P)
    #best_P = np.linalg.inv(K) @ best_P
    #show(best_P)
    
    # Use this P to reproject all points.
    proj = project(matched3D, best_Pi)
    # Calculate distances
    d = np.sqrt(np.power(proj - matched2D_normalized,2).sum(axis=1))
    inliers_mask = d < 0.001
    inliers = inliers_mask.sum()

    print("inliers2:", inliers)

    Pi_matrices[i] = best_Pi
    P_matrices[i] = K @ best_Pi

    P = K @ best_Pi

    P_prev = P_matrices[i-1]
    # Now, use that P with P_prev to triangulate some new points to the pointcloud.

    p1 = pointlist_to_homog(new2D[:,0:2])
    p2 = pointlist_to_homog(new2D[:,2:4])
    results = np.array([triangulate(P_prev,P,p1_,p2_) for p1_, p2_ in zip(p1,p2)])
    for n in range(0, p1.shape[0]):
        X = results[n]
        match = new2D[n]
        matchlist = [(i-1, int(match[0]), int(match[1])), (i, int(match[2]), int(match[3]))]
        pointcloud += [[X, matchlist]]
    
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

# Append camera positions
k=0
for i in range(0,len(images)):
    P = P_matrices[i]
    K, R, T, C = decompose_P(P)
    print(K)
    print(T)
    results += [(-T[0],T[1],T[2],255,0,0)]
    k+=1

print("Saving results...")
save_to_ply(results, "out.ply")
