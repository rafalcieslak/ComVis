import scipy
import scipy.linalg
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# A helper printer which I use instead of print for numpy arrays.
# I don't know why, but print clearly ignores numpy output settings
# for some (large?) arrays. show() fixes this.
def show(array):
    print(np.array_str(array,suppress_small=True,precision=5))

# Generic dataset normalization around 0 with stdev 1
def normalize_data(data):
    mean = data.mean(axis=0)
    translate = -mean
    data = data + translate
    scale = data.std() * 1.6
    data /= scale
    return translate, scale, data

# Adds a new column eq to 1, effectively transforming a pointset to
# homogenous coordinates
def to_H(data):
    return np.hstack((data,np.ones((data.shape[0],1))))

# Estimates projection matrix based on corresionding 3d points in
# world space and 2d points on image plane
def estimate_P(data2d, data3d):
    n = data2d.shape[0]
    assert(n == data3d.shape[0])
    # To homogenous
    data2dH = np.hstack((data2d,np.ones((n,1))))
    data3dH = np.hstack((data3d,np.ones((n,1))))
    # Prepare matrix for linear system
    A1 = np.zeros((2*n,4))
    A2 = np.zeros((2*n,4))
    A1[0::2] = data3dH
    A2[1::2] = data3dH
    A3 = -1*(A1+A2)*data2d.reshape((2*n,1))
    A = np.hstack((A1, A2, A3))
    # Use A as the matrix of system Ax = 0, solve
    U, S, V = np.linalg.svd(A)
    P = V[-1,:].reshape(3,4)
    return P

# Performs a projection of a point set using the given matrix
def project(data3d, P):
    n = data3d.shape[0]
    # To homogenous
    data3dH = np.hstack((data3d,np.ones((n,1))))
    # Apply P to each 3d point
    proj = np.einsum('ab,cb->ca',P, data3dH)
    # Back from homogenous
    proj = (proj.T / proj[:,-1]).T[:,0:2]
    return proj

# Computes the total residual error of projection between
# corresponding sets of 3d and 2d points with a specified projection
# matrix
def calculate_residual(data2d, data3d, P):
    proj = project(data3d, P)
    # Calculate distances
    d = np.sqrt(np.power(proj - data2d,2).sum(axis=1))
    return d.sum()

# Decomposes P into: K - intrisic matrix, R - rotation matrix, T -
# world origin coordinates in camera space, C - camera origin
# coordinates in world space
def decompose_P(P):
    M = P[:,0:3]
    mMC = P[:,3:4].reshape((3))
    C = -(np.linalg.inv(M)).dot(mMC)
    K, R = scipy.linalg.rq(M)
    
    # Optional: Correct axis orientation
    U = np.diag(np.sign(np.diag(K)))
    K = K.dot(U)
    R = U.dot(R)
    
    T = -R.dot(C)
    return K, R, T, C

# Creates a simple 3d plot from a set of points and one specified
# point to be highlighted. Ensures that a uniform scaling is applied
# to all axes, so that 3d spatial data is not stretched
def plot_points_camera(data3d, C):
    fig = plt.figure()
    sp = fig.add_subplot(111, projection='3d')
    # sp.axis('equal') # Does not work for 3d.
    X,Y,Z = data3d[:,0], data3d[:,1], data3d[:,2]

    X = np.insert(X, 0, C[0])
    Y = np.insert(Y, 0, C[1])
    Z = np.insert(Z, 0, C[2])
    
    # Correction for unit scaling
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()*0.5
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    sp.set_xlim(mid_x - max_range, mid_x + max_range)
    sp.set_ylim(mid_y - max_range, mid_y + max_range)
    sp.set_zlim(mid_z - max_range, mid_z + max_range)
    
    sp.scatter(X[1:],Y[1:],Z[1:], c='b', marker='o')
    sp.scatter(C[0], C[1], C[2], c='r', marker='x')
    plt.show()

# Applies an arbitrary image transformation. The function which is
# passed as the second argument will be called with a long list of
# coordinates. It must return an array of identical size, but it may
# modify the coordinates of each point, to mark that its value shall
# be sampled from specified location. For example, a function which
# adds n to the second column of its input will transform the image so
# that it shifts n pixels downwards. What is cool about this
# implemntation is that the called function may batch-process indices,
# which allows to create an arbitrary image warp transformation without
# processing pixels in a loop.
def arbitrary_image_transform(image, function):
    cx,cy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    coords = np.stack((cx,cy), axis=2).reshape((-1,2), order='F')
    coords2 = np.fliplr(function(np.fliplr(coords)))
    assert coords.shape == coords2.shape, ("Original coords shape %s is not equal to modified coords shape %s." % (coords.shape, coords2.shape))
    coordsB = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=0)
    coordsG = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=1)
    coordsR = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=2)
    ptsB = scipy.ndimage.map_coordinates(image, coordsB.T, order=3)
    ptsG = scipy.ndimage.map_coordinates(image, coordsG.T, order=3)
    ptsR = scipy.ndimage.map_coordinates(image, coordsR.T, order=3)
    pts = np.vstack((ptsB, ptsG, ptsR)).T
    pts = pts.reshape(image.transpose(1,0,2).shape, order='F').transpose((1,0,2))
    # A copy is needed due to a bug in opencv which causes it to
    # incorrectly track the data layout of numpy arrays which are
    # temporarily in an optimized layout
    return pts.copy()
