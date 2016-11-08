import scipy
import scipy.linalg
import numpy as np
import colorsys
import cv2
from plyfile import PlyData, PlyElement

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# A helper printer which I use instead of print for numpy arrays.
# I don't know why, but print clearly ignores numpy output settings
# for some (large?) arrays. show() fixes this.
def show(array):
    print(np.array_str(array,suppress_small=True,precision=5))

def rank_reduce(A):
    U, S, V = np.linalg.svd(A)
    S[-1] = 0
    return U.dot(np.diag(S)).dot(V)

def rank_reduce_essential(A):
    assert(A.shape == (3,3))
    U, S, V = np.linalg.svd(A)
    a, b, c = S
    q = 0.5 * (a+b)
    S = np.diag((q,q,0))
    return U.dot(S).dot(V)

# A helper function for merging three single-channel images into an RGB image
def combine_channels(ch1, ch2, ch3):
    return np.array([ch1.T, ch2.T, ch3.T]).T

def per_channel(func, image):
    assert(image.shape[2] == 3)
    r,g,b = image[:,:,0],image[:,:,1],image[:,:,2]
    return combine_channels(func(r), func(g), func(b))

def zoom_3ch(image, factor):
    return per_channel(lambda x : scipy.ndimage.zoom(x, factor, order=1), image)

def calculate_fundamental(p1, p2, n=None, essential=False):
    if n is None:
        n = p1.shape[0]
    points_combined = np.hstack((p1, p2))
    choice = np.random.choice(p1.shape[0], n, replace=False)
    points = points_combined[choice]
    A = []
    for row in points:
        u, v, _, up, vp, _ = row
        A.append([up*u, up*v, up,
                  vp*u, vp*v, vp,
                     u,    v,  1])
    A = np.asarray(A)
    # Solve Af=0 with least squares, constarining |f|=1
    U, S, V = np.linalg.svd(A)
    F = V[-1,:].reshape(3,3)
    if essential:
        F = rank_reduce_essential(F)
    else:
        F = rank_reduce(F)
    F = F/F[2,2]
    return F, choice


def prepare_normalization_transforms(p1, p2):
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
    return T1, T2

def get_Ps_from_F(F):
    U,S,Vt = np.linalg.svd(F)
    print("factorisation")
    show(F)
    show(S)
    show(U @ np.diag(S) @ Vt)
    u3 = U[2].reshape((3,1))
    
    W = np.asarray([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])
    P1 = np.asarray([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0]])
    P2_1 = np.hstack([U @ W   @ Vt,  u3])
    P2_2 = np.hstack([U @ W   @ Vt, -u3])
    P2_3 = np.hstack([U @ W.T @ Vt,  u3])
    P2_4 = np.hstack([U @ W.T @ Vt, -u3])
    return P1, [P2_1, P2_2, P2_3, P2_4]

def get_Ps_from_F_alternative(F):
    U,S,Vt = np.linalg.svd(F)
    # show(F)
    # show(S)
    # show(U @ np.diag(S) @ Vt)
    u3 = U[2].reshape((3,1))
    
    W = np.asarray([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])
    Z = np.asarray([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 0]])
    P1 = np.asarray([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0]])
    UZUt = U @ Z @ U.T
    UZDVt = U @ Z @ np.diag(S) @ Vt
    t = np.array([UZUt[2,1], UZUt[0,2], UZUt[1,0]])
    M = UZDVt
    # show(UZUt)
    # show(t)
    # show(M)
    P2 = np.hstack([M,t.reshape(3,1)])
    # show(P2)
    return P1, [P2]
    
    
def triangulate(P1, P2, p1, p2):
    x1,y1,_ = p1/p1[2]
    x2,y2,_ = p2/p2[2]
    A1 = x1*P1[2,:] - P1[0,:]
    A2 = y1*P1[2,:] - P1[1,:]
    A3 = x2*P2[2,:] - P2[0,:]
    A4 = y2*P2[2,:] - P2[1,:]
    A = np.vstack([A1,A2,A3,A4])
    # print("A")
    # show(A1)
    # show(A)
    U,S,V = np.linalg.svd(A)
    X = V[-1,:]
    q = A @ X
    # print("A dot X")
    # show(q)
    return X
    

def draw_points_and_epipolar(img1, img2, p1, p2, F, zoom_factor=1.0):
    line_width = int(1.5 / zoom_factor)
    circle_radius = int(8 / zoom_factor)
    circle_width = int(2/ zoom_factor)
    rows, cols, _ = img1.shape
    for pt1, pt2 in zip(p1, p2):
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
    return img1, img2

def save_to_ply(points, filename):
    points = np.array(points, dtype=[('x','f4'), ('y','f4'),('z', 'f4')])
    el = PlyElement.describe(points, 'vertex')
    PlyData([el]).write(filename)

# Creates a simple 3d plot from a set of points. Ensures that a uniform scaling
# is applied to all axes, so that 3d spatial data is not stretched
def plot_points(data3d):
    C = np.array([0,0,0])
    fig = plt.figure()
    sp = fig.add_subplot(111, projection='3d')
    # sp.axis('equal') # Does not work for 3d.
    X,Y,Z = data3d[:,0], data3d[:,1], data3d[:,2]
    """
    X = np.insert(X, 0, C[0])
    Y = np.insert(Y, 0, C[1])
    Z = np.insert(Z, 0, C[2])
    """ 
    # Correction for unit scaling
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()*0.5
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    sp.set_xlim(mid_x - max_range, mid_x + max_range)
    sp.set_ylim(mid_y - max_range, mid_y + max_range)
    sp.set_zlim(mid_z - max_range, mid_z + max_range)
    
    sp.scatter(X[1:],Y[1:],Z[1:], c='b', marker='o', s=2)
    # sp.scatter(C[0], C[1], C[2], c='r', marker='x')
    plt.show()
