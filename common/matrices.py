import scipy
import scipy.linalg
import numpy as np
from .show import *

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

def rectify_stereo(Po1, Po2, imgcenter):
    K1, R1, t1, c1 = decompose_P(Po1)
    K2, R2, t2, c2 = decompose_P(Po2)
    
    v1 = c2 - c1;
    v2 = np.cross(R1[2,:].T, v1)
    v3 = np.cross(v1,v2)
    
    R = np.vstack([v1.T/np.linalg.norm(v1),
                   v2.T/np.linalg.norm(v2),
                   v3.T/np.linalg.norm(v3)])
    Kn1 = K1
    Kn2 = K2
    Pn1 = Kn1 @ np.hstack([R, (-R @ c1).reshape(3,1)])
    Pn2 = Kn2 @ np.hstack([R, (-R @ c2).reshape(3,1)])
    
    # Compute homographies translating from Po to Pn
    H1 = Po1[0:3,0:3] @ np.linalg.inv(Pn1[0:3,0:3])
    H2 = Po2[0:3,0:3] @ np.linalg.inv(Pn2[0:3,0:3])
    
    ctrH1 = H1 @ imgcenter
    ctrH2 = H2 @ imgcenter
    displ1 = (ctrH1/ctrH1[2] - imgcenter)[0:2]
    displ2 = (ctrH2/ctrH2[2] - imgcenter)[0:2]
    
    # Now, correct Hs.
    Kn1[0,2] += displ1[0]
    Kn1[1,2] += displ1[1]
    Kn2[0,2] += displ2[0]
    Kn2[1,2] += displ2[1]
    Pn1 = Kn1 @ np.hstack([R, (-R @ c1).reshape(3,1)])
    Pn2 = Kn2 @ np.hstack([R, (-R @ c2).reshape(3,1)])
    H1 = Po1[0:3,0:3] @ np.linalg.inv(Pn1[0:3,0:3])
    H2 = Po2[0:3,0:3] @ np.linalg.inv(Pn2[0:3,0:3])
    
    return H1,H2, Pn1

def baseline_length(P1, P2):
    K1, R1, t1, c1 = decompose_P(P1)
    K2, R2, t2, c2 = decompose_P(P2)
    return np.sqrt(np.dot(c1 - c2, c1 - c2))

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

# Make sure both p1 and p2 contain exactly 8 points.
def calculate_fundamental(p1, p2, essential=False):
    points = np.hstack((p1, p2))
    # choice = np.random.choice(p1.shape[0], n, replace=False)
    # points = points_combined[choice]
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
    return F

# As hinted by http://users.cecs.anu.edu.au/~hartley/Papers/CVPR99-tutorial/tut_4up.pdf
# around page 11
def get_Ps_from_F_alternative(F):
    U,S,Vt = np.linalg.svd(F)
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
    P2 = np.hstack([M,t.reshape(3,1)])
    return P1, [P2]

def get_Ps_from_E(E):
    U,S,Vt = np.linalg.svd(E)
    # This was U[2], but that would be wrong, right?
    u3 = U[:,2].reshape((3,1))
    
    W = np.asarray([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])
    P1 = np.asarray([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0]]).astype(np.float)
    P2_1 = np.hstack([U @ W   @ Vt,  u3])
    P2_2 = np.hstack([U @ W   @ Vt, -u3])
    P2_3 = np.hstack([U @ W.T @ Vt,  u3])
    P2_4 = np.hstack([U @ W.T @ Vt, -u3])
    return P1, [P2_1, P2_2, P2_3, P2_4]

def triangulate(P1, P2, p1, p2):
    x1,y1,_ = p1/p1[2]
    x2,y2,_ = p2/p2[2]
    A1 = x1*P1[2,:] - P1[0,:]
    A2 = y1*P1[2,:] - P1[1,:]
    A3 = x2*P2[2,:] - P2[0,:]
    A4 = y2*P2[2,:] - P2[1,:]
    A = np.vstack([A1,A2,A3,A4])
    U,S,V = np.linalg.svd(A)
    X = V[-1,:]
    X /= X[3]
    return X[0:3]


# Estimates projection matrix based on corresionding 3d points in
# world space and 2d points on image plane
def estimate_P(data2d, data3d):
    #show(data2d)
    #show(data3d)
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
    
    #T = -R.dot(C)
    T = R.dot(C)
    return K, R, T, C
