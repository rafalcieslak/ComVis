import scipy
import scipy.ndimage
import numpy as np
import cv2
from .show import show

# A helper function for merging three single-channel images into an RGB image
def combine_channels(ch1, ch2, ch3):
    return np.array([ch1.T, ch2.T, ch3.T]).T

def image_to_3ch(image):
    return combine_channels(image,image,image)

def per_channel(func, image):
    assert(image.shape[2] == 3)
    r,g,b = image[:,:,0],image[:,:,1],image[:,:,2]
    return combine_channels(func(r), func(g), func(b))

def zoom_3ch(image, factor):
    return per_channel(lambda x : scipy.ndimage.zoom(x, factor, order=1), image)


def rgb2Y(img):
    return np.dot(img, np.array([ 0.2989,  0.5866,  0.1145]))

# Functions for converting color space on a three-channel image between RGB and YCbCr.
# Coefficients follow JPEG specification (https://www.w3.org/Graphics/JPEG/jfif3.pdf).
def split_to_YCbCr(img):
    Y  = np.dot(img, np.array([ 0.2989,  0.5866,  0.1145]))
    Cb = np.dot(img, np.array([-0.1687, -0.3313,  0.5000]))
    Cr = np.dot(img, np.array([ 0.5000, -0.4184, -0.0816]))
    return Y,Cb,Cr
def merge_from_YCbCr(Y, Cb, Cr):
    img = combine_channels(Y, Cb, Cr)
    r = np.dot(img, np.array([1.0,  0.0000,  1.4022]))
    g = np.dot(img, np.array([1.0, -0.3456, -0.7145]))
    b = np.dot(img, np.array([1.0,  1.7710,  0.0000]))
    return np.maximum(np.minimum(1.0,combine_channels(r,g,b)), 0.0)

# Shuffle dimentions to match cv2 color layout
def cv2_shuffle(image):
    return np.array([image.T[2], image.T[1], image.T[0]]).T

def remove_alpha(image):
    if image.shape[2] == 4:
        return image[:,:,0:3]
    else:
        return image

def flip_xy(pointlist):
    print("Flippin'")
    print(pointlist.shape)
    pointlist2 = np.zeros_like(pointlist)
    pointlist2[:,0] = pointlist[:,1]
    pointlist2[:,1] = pointlist[:,0]
    if pointlist.shape[1] > 2:
        pointlist2[:,2] = pointlist[:,2]
    return pointlist2

# (n,2) -> (n,3)
def pointlist_to_homog(points):
    return np.hstack([points, np.ones((points.shape[0], 1))])

# (n,3), -> (n,2)
def pointlist_from_homog(points):
    points[:,0] = points[:,0]/points[:,2]
    points[:,1] = points[:,1]/points[:,2]
    return points[:,0:2]


class HomographyApplier:
    def __init__(self, H, offset=np.array([0,0])):
        self.H = H
        self.offset = offset
    def __call__(self, data):
        data = data + self.offset
        dataH = pointlist_to_homog(data)
        dataH = np.einsum('ba,na->nb', self.H, dataH)
        data = pointlist_from_homog(dataH)
        return data

# Applies an arbitrary image transformation. The function which is
# passed as the second argument will be called with a long list of
# coordinates. It must return an array of identical size, but it may
# modify the coordinates of each point, to mark that its value shall
# be sampled from specified location. For example, a function which
# adds n to the second column of its input will transform the image so
# that it shifts n pixels downwards. What is cool about this
# implementation is that the called function may batch-process indices,
# which allows to create an arbitrary image warp transformation without
# processing pixels in a loop.
def img_transform(source_image, function, target_shape=None, constant=[0,0,0], order=3, mode='constant'):
    if target_shape is None:
        target_shape = source_image.shape
    cx,cy = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]))
    coords = np.stack((cx,cy), axis=2).reshape((-1,2), order='F')
    coords2 = np.fliplr(function(np.fliplr(coords)))
    assert coords.shape == coords2.shape, ("Original coords shape %s is not equal to modified coords shape %s." % (coords.shape, coords2.shape))
    coordsB = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=0)
    coordsG = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=1)
    coordsR = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=2)
    ptsB = scipy.ndimage.map_coordinates(source_image, coordsB.T, order=order, mode=mode, cval=constant[2])
    ptsG = scipy.ndimage.map_coordinates(source_image, coordsG.T, order=order, mode=mode, cval=constant[1])
    ptsR = scipy.ndimage.map_coordinates(source_image, coordsR.T, order=order, mode=mode, cval=constant[0])
    pts = np.vstack((ptsB, ptsG, ptsR)).T
    tshape = (target_shape[1], target_shape[0], target_shape[2])
    pts = pts.reshape(tshape, order='F').transpose((1,0,2))
    # A copy is needed due to a bug in opencv which causes it to
    # incorrectly track the data layout of numpy arrays which are
    # temporarily in an optimized layout
    return pts.copy()


def img_transform_1ch(source_image, function, target_shape=None, constant=0, order=3, mode='constant'):
    if target_shape is None:
        target_shape = source_image.shape
    cx,cy = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]))
    coords = np.stack((cx,cy), axis=2).reshape((-1,2), order='F')
    coords2 = np.fliplr(function(np.fliplr(coords)))
    assert coords.shape == coords2.shape, ("Original coords shape %s is not equal to modified coords shape %s." % (coords.shape, coords2.shape))
    coordsX = np.pad(coords2, ((0,0),(0,0)), mode='constant', constant_values=0)
    # print(coordsX.T.shape)
    pts = scipy.ndimage.map_coordinates(source_image, coordsX.T, order=order, mode=mode, cval=constant)
    pts = pts.T
    tshape = (target_shape[1], target_shape[0])
    pts = pts.reshape(tshape, order='F').transpose((1,0))
    # A copy is needed due to a bug in opencv which causes it to
    # incorrectly track the data layout of numpy arrays which are
    # temporarily in an optimized layout
    return pts.copy()

# Similar to img_transform, but the argument function shall return not coordinates, but values.
def img_gen(function, target_shape=None):
    if target_shape is None:
        target_shape = source_image.shape
    cx,cy = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]))
    coords = np.stack((cx,cy), axis=2).reshape((-1,2), order='F')
    pts = function(np.fliplr(coords))
    assert coords.shape[0] == pts.shape[0], ("Original coords shape %s is not mathching samples shape %s." % (coords.shape, pts.shape))
    tshape = (target_shape[1], target_shape[0], target_shape[2])
    pts = pts.reshape(tshape, order='F').transpose((1,0,2))
    return pts.copy()

def img_transform_H(source, H, target_shape=None, constant=[0,0,0], order=3, offset=np.array([0,0])):
    if target_shape is None:
        target_shape = source.shape
    return img_transform(source, HomographyApplier(H, offset), target_shape, constant, order)


def find_homography(points1, points2):
    A = []
    for (x1, y1, _), (x2, y2, _) in zip(points1, points2):
        A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
    A = np.asarray(A)

    U,S,V = np.linalg.svd(A)
    H = V[-1,:].reshape(3,3)
    H /= H[2,2]
    return H

def image_bounds_pointlist(image):
    h,w,_ = image.shape
    return np.array([[0,0],[0,h],[w,0],[w,h]])

def image_bounds_homography(image, H):
    return np.einsum('ba,na->nb', np.linalg.inv(H), pointlist_to_homog(image_bounds_pointlist(image)))

# Returns: size (as tuple), base offset (as np.array)
def image_bounds_factorize(BB):
    xmin, ymin = BB.min(axis=0)
    xmax, ymax = BB.max(axis=0)
    xsize = int(xmax - xmin)
    ysize = int(ymax - ymin)
    offset = np.array([xmin, ymin])
    return (ysize,xsize), offset

def img_gen_mask_smooth(img):
    def one_func(coords):
        size = np.array([img.shape[1], img.shape[0]])
        q = 1.0 - np.abs(coords/(size/2) - 1.0)
        q = q.min(axis=1)
        # q = np.power(q,2)
        return combine_channels(q,q,q)
    i = img_gen(one_func, img.shape)
    # i = scipy.ndimage.gaussian_filter(i, 10)
    #low = i.min()
    #s = 1.0 - low
    #return (i-low)/s
    return i

def img_gen_mask_ones(img):
    def one_func(coords):
        return np.ones((coords.shape[0], 3))
    return img_gen(one_func, img.shape)

# Normalizes values into range [0,1]
def rescale(img):
    min, max = img.min(), img.max()
    return (img-min)/(max-min)

# Sigma1 is used for derivative calculation.
# Sigma2 corresponds to the gaussian window stdev
def harris_corner_response(I, alpha=0.05, sigma1=None, sigma2=4):
    if sigma1 is None:
        sigma1 = 0.7 * sigma2
    Ix = scipy.ndimage.gaussian_filter(I, (0, sigma1), order=(0,1))
    Iy = scipy.ndimage.gaussian_filter(I, (sigma1, 0), order=(1,0))
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    gIxx = scipy.ndimage.gaussian_filter(Ixx, sigma2)
    gIxy = scipy.ndimage.gaussian_filter(Ixy, sigma2)
    gIyy = scipy.ndimage.gaussian_filter(Iyy, sigma2)
    det = gIxx * gIyy - gIxy * gIxy
    tr = gIxx + gIyy
    return det*(256*256) - alpha * tr

# For a 1-channel 2d image, returns a list of tuples x,y,v, where x and y are
# coordinates of a local maxima, and v is the value at the point
def find_local_maximas_with_values(image, neighbourhood_size=5, flat_treshold=0.003):
    # Find local maximas
    i_max = scipy.ndimage.maximum_filter(image, neighbourhood_size)
    maxima = (image == i_max)
    # Clean flat regions
    i_min = scipy.ndimage.minimum_filter(image, neighbourhood_size)
    d = ((i_max - i_min) > flat_treshold)
    maxima[d == 0] = False
    # Get coordinates of the maximas
    labeled_R, n_obj = scipy.ndimage.label(maxima)
    slices = scipy.ndimage.find_objects(labeled_R)
    # print("Found %d corners" % n_obj)

    # Sample the response function at these points
    points = []
    for dy,dx in slices:
        x = (dx.start + dx.stop - 1)/2
        y = (dy.start + dy.stop - 1)/2
        points.append((x, y, image[y,x]))
    return points

def harris_corners(I, sigma):
    R = harris_corner_response(I, sigma2=sigma)
    # Scale response function to stay sigma-invariant. Obviously, the scaling
    # factor should be a power of sigma, but I have no idea why 5th power
    # works. Quick calculation suggests that either 2 or 4 should be the right
    # choice, but it's easy to verify that 5 actually works much beter and is
    # apparently the perfect choice.
    R = R * np.power(sigma, 5)
    points = find_local_maximas_with_values(R)
    # Append sigma to all points
    return [(x,y,r,sigma) for x,y,r in points], R

# Adaptive non-maximal suppresion.
# Input: List of tuples x,y,v where x and y are coordinates and v is the value at point
def ANMS(pointlist, N):
    pointlist = np.asarray(pointlist)
    radii = []
    for x,y,v,s in pointlist:
        # Find the distance to the nearest point with a higher value
        # First, get indices where v is larger than ours
        # print(pointlist[:,2])
        q = (pointlist[:,2] > v).nonzero()
        p = pointlist[q,:][0,:,0:2]
        if p.shape[0] == 0:
            r = 999999999999; # Global maximum
        else:
            diffs = p - np.array([x,y])
            r = (diffs**2).sum(axis=1).min()
        radii.append(r)
    N = min(len(radii)-1, N)
    args = np.argpartition(-np.asarray(radii), N)[:N]
    cutoff = np.asarray(radii)[args[-1]]
    print("Cutoff radius for ANMS: %f" % np.sqrt(cutoff))
    return pointlist[args]

# Harris-laplace scale-invariant corner detector
def harris_laplace(I):
    sigma_begin = 1.5
    sigma_step = 1.2
    sigma_n = 13
    sigmas = np.power(sigma_step, np.arange(sigma_n)) * sigma_begin

    # Run harris corners detector for all sigma values, collect results
    points, allpoints, Rs = [], [], []
    for s in sigmas:
        p, R = harris_corners(I, s)
        allpoints += p
        points.append(p)
        Rs.append(R)
    assert(len(points) == sigma_n)

    print("Before selecting scale-space local maximas: %d points" % len(allpoints))

    filtered_points = []
    # Now, filter the points to only include those that are local maximas in
    # scale space as well. For convenience, add zeroed arrays to end end
    # beginning, so there are no corner cases when comparing scales
    Rs = [np.zeros_like(Rs[0])] + Rs + [np.zeros_like(Rs[-1])]
    # This entire loop might be rewritten numpy-style, but it is tricky to run
    # the maxima filter only in the third dimension. Thus, for simplicity, this
    # loop is explicit. It's not costing much anyway, since usually there is not
    # much (< 20000) candidate points anyway.
    for i in range(0,sigma_n):
        sigma = sigmas[i]
        Ra, R, Rb = Rs[i], Rs[i+1], Rs[i+2]
        for p in points[i]:
            x,y,v,_ = p
            # If neighbouring scales have a smaller value at (x,y)...
            if v > Ra[y,x] and v > Rb[y,x]:
                # ... then this point is a local maxima in scale-space as well.
                filtered_points.append(p)

    print("After selecting scale-space local maximas: %d points" % len(filtered_points))

    return filtered_points


def decimate(image, target_size):
    decimate_factors = (target_size[0] / image.shape[0],
                        target_size[1] / image.shape[1])
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


