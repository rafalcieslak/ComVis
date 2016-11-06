import scipy
import scipy.linalg
import numpy as np

# A helper printer which I use instead of print for numpy arrays.
# I don't know why, but print clearly ignores numpy output settings
# for some (large?) arrays. show() fixes this.
def show(array):
    print(np.array_str(array,suppress_small=True,precision=5))

def rank_reduce(A):
    U, S, V = np.linalg.svd(A)
    S[-1] = 0
    return U.dot(np.diag(S)).dot(V)

# A helper function for merging three single-channel images into an RGB image
def combine_channels(ch1, ch2, ch3):
    return np.array([ch1.T, ch2.T, ch3.T]).T

def per_channel(func, image):
    assert(image.shape[2] == 3)
    r,g,b = image[:,:,0],image[:,:,1],image[:,:,2]
    return combine_channels(func(r), func(g), func(b))

def zoom_3ch(image, factor):
    return per_channel(lambda x : scipy.ndimage.zoom(x, factor, order=1), image)

def calculate_fundamental(p1, p2, n=None):
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
    F = rank_reduce(F)
    F = F/F[2,2]
    return F, choice



