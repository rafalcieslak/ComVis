import scipy
import scipy.linalg
import numpy as np

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
