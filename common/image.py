import scipy
import scipy.ndimage
import numpy as np
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
