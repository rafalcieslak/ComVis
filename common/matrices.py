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
