import scipy
import scipy.io
import scipy.ndimage
import numpy as np
import cv2
import sys
from common import *

MODE_FUNDAMENTAL = 1
MODE_FUNDAMENTAL_NORMALIZED = 2
MODE_ESSENTIAL = 3

mode = MODE_ESSENTIAL
pick_n_points = 15
triangulation = True

np.random.seed(42)

# Load input data
q = scipy.io.loadmat("data/compEx1data.mat")['x']
p1 = q[0,0].T
p2 = q[1,0].T

# Load images
img1 = scipy.ndimage.imread(("data/kronan1.JPG"))
img2 = scipy.ndimage.imread(("data/kronan2.JPG"))
zoom_factor = 0.5

# Load K matrix
K = scipy.io.loadmat("data/compEx3data.mat")['K']
K_inv = np.linalg.inv(K)

if mode == MODE_FUNDAMENTAL:
    # Compute fundamental matrix
    F, choice = calculate_fundamental(p1, p2, pick_n_points)

    print("This is F:")
    show(F)

    # Ask openCV to compute on the same points, for comparison
    F2, _ = cv2.findFundamentalMat(p1[choice][:,0:2], p2[choice][:,0:2], cv2.FM_8POINT)
    print("F according to openCV")
    show(F2)

elif mode == MODE_FUNDAMENTAL_NORMALIZED:
    T1, T2 = prepare_normalization_transforms(p1,p2)

    print("T1:")
    show(T1)
    print("T2:")
    show(T2)

    p1_normalized = np.einsum('xy,zy->zx', T1, p1)
    p2_normalized = np.einsum('xy,zy->zx', T2, p2)

    # Compute fundamental matrix
    F, choice = calculate_fundamental(p1_normalized, p2_normalized,
                                      pick_n_points)
    print("This is F:")
    show(F)

    # Ask openCV to compute on the same points, for comparison
    F2, _ = cv2.findFundamentalMat(p1_normalized[choice][:,0:2],
                                   p2_normalized[choice][:,0:2], cv2.FM_8POINT)
    print("F according to openCV")
    show(F2)

    F = T2.T @ F @ T1
    F = F/F[2,2]
    print("This is F after denormalizing:")
    show(F)

elif mode == MODE_ESSENTIAL:
    print("K:")
    show(K)

    p1_normalized = np.einsum('xy,zy->zx', K_inv, p1)
    p2_normalized = np.einsum('xy,zy->zx', K_inv, p2)

    # Compute fundamental matrix
    E, choice = calculate_fundamental(p1_normalized, p2_normalized, pick_n_points, essential=True)
    print("This is E:")
    show(E)

    F = K_inv.T @ E @ K_inv
    F = F/F[2,2]
    print("This is F:")
    show(F)

testpoints = np.random.choice(p1.shape[0], 20, replace=False)
if pick_n_points is not None:
    testpoints = choice

img1, img2 = draw_points_and_epipolar(img1, img2, p1[testpoints], p2[testpoints], F, zoom_factor)

img1 = zoom_3ch(img1,zoom_factor)
img2 = zoom_3ch(img2,zoom_factor)

if triangulation:
    P1, P2l = get_Ps_from_F_alternative(F)
    P2 = P2l[0]
    #P2 = P2/P2[2:3]
    print("P1:")
    show(P1)
    print("P2:")
    show(P2)
    # Test triangulation for some point
    n = 0
    X = triangulate(P1,P2,p1[n],p2[n])
    # X = np.array(X)/X[3]
    print("p")
    show(p1[n])
    show(p2[n])
    print("X")
    print( X )
    print("Reproject P1 X")
    r1 = P1 @ np.array([X[0], X[1], X[2], X[3]])
    show(r1/r1[2])
    print("Reproject P2 X")
    r2 = P2 @ np.array([X[0], X[1], X[2], X[3]])
    show(r2/r2[2])
    
    results = np.array([triangulate(P1,P2,p1_,p2_) for p1_, p2_ in zip(p1,p2)])
    print(results[n])
    scale = np.array([1, 0.2, 100.0])
    results = [(x*scale[0]/w,y*scale[1]/w,z*scale[2]/w) for x,y,z,w in results]
    
    #print(p1[:,0:2].shape)
    #compare = cv2.triangulatePoints(P1, P2, p1[:,0:2].T, p2[:,0:2].T).T
    #compare = compare/compare[0,0]
    #compare = [(x/w,y/w,z/w) for x,y,z,w in compare]

    print("Saving results")
    save_to_ply(results, "out.ply")
    #print("Saving compare")
    #save_to_ply(compare, "compare.ply")
    

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
if triangulation:
    plot_points(np.asarray(results))
    
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    
