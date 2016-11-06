import scipy
import scipy.linalg
import numpy as np
import colorsys
import cv2

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
