import argparse
import scipy
import numpy as np
import cv2
import sys
import colorsys
from common.show import *
from common.image import *

step_by_step_debug = False

parser = argparse.ArgumentParser(description='Generate a list of features from a image file.')
parser.add_argument('image_path', metavar="IMAGE", type=str, help="Path to source image file.")
parser.add_argument('feat_path', metavar="FEATURES", type=str, help="Path to features file.")
parser.add_argument('output_path', metavar="OUTPUT", type=str,
                    help="Path to otuput file. Descriptors will be stored in this file, one per line, matching the input features file.")

try:
    args = parser.parse_args()
except SystemExit:
    sys.exit(0)
    
# Open image file
image = scipy.ndimage.imread(args.image_path)[:,:,0:3]/255.0

# Convert image to intensity
I = rgb2Y(image)
# Normalize the image to [0,1]
I = I / I.max()

features = np.loadtxt(args.feat_path)

# Add padding around I
PATCH_R = 20
PAD_SIZE = PATCH_R*2 + 1
I = np.pad(I, ((PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE)), mode='edge')
features += np.array([PAD_SIZE, PAD_SIZE, 0])

# Y is inverted, because the Y axis direction in complex space is different than in image space
gradient_kernel = [-1.0, 0.0, 1.0]
dY =  scipy.ndimage.convolve1d(I, gradient_kernel, axis=0)
dX = -scipy.ndimage.convolve1d(I, gradient_kernel, axis=1)
cgrad = dX + 1j * dY

gradangle = np.angle(cgrad, deg=True)

def image_hsv_2_rgb(image):
    # Note: This could be done faster, but I don't care - this is just for debug display.
    n,m,c = image.shape
    assert(c == 3)
    for y in range(0,n):
        for x in range(0,m):
            h,s,v = image[y,x]
            r,g,b = colorsys.hsv_to_rgb(h,s,v)
            image[y,x] = (r,g,b)
    return image

def cgrad_display(cgrad):
    angle = np.angle(cgrad, deg=True)
    hue = (angle/360 + 2.0)%1.0
    lightness = np.abs(cgrad)
    lightness = lightness/lightness.max()
    hsv = combine_channels(hue, np.ones_like(hue) * 0.7, lightness)
    rgb = image_hsv_2_rgb(hsv)
    return rgb

kernel_cache = {}
class KernelGen:
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, data):
        data -= PATCH_R
        dist = (data**2).sum(axis=1)
        # Dist is already squared!
        return np.exp(-(dist)/(2 * (self.sigma**2)))

def get_gaussian_kernel(sigma):
    if sigma in kernel_cache:
        return kernel_cache[sigma]
    print("No kernel %f in cache, generating" % sigma)
    kernel = img_gen(KernelGen(sigma), (PATCH_R*2+1,PATCH_R*2+1,1))[:,:,0]
    kernel_cache[sigma] = kernel
    return kernel

# Iterate over features
descrips = []
for i,f in enumerate(features):
    x, y, scale = f
    x, y = int(x), int(y)
    if step_by_step_debug:
        print("Feature %d: %d %d (%f)" % (i, x, y, scale))
    sl = (slice(y-PATCH_R, y+PATCH_R+1), slice(x-PATCH_R,x+PATCH_R+1))
    patch = I[sl]
    patch_cgrad = cgrad[sl]
    patch_gradangle = gradangle[sl]
    kernel = get_gaussian_kernel(scale * 1.5)
    w = kernel * np.abs(patch_cgrad)
    
    hist, bins = np.histogram(patch_gradangle, 37, (-185, 185), weights=w, density=False)
    hist[0] += hist[-1]
    hist = hist[0:36]
    hist_s = np.argsort(hist)

    dominant_angle = bins[hist_s][-1] + 5;
    print(dominant_angle)
    
    if step_by_step_debug:
        show(hist[hist_s])
        show(bins[hist_s] + 5)
        cv2.imshow('patch', image_to_3ch(scipy.ndimage.zoom(patch, 10.0, order=0)))
        cv2.imshow('kernel', image_to_3ch(scipy.ndimage.zoom(kernel, 10.0, order=0)))
        cv2.imshow('grads', scipy.ndimage.zoom(cgrad_display(patch_cgrad), (10,10,1), order=0))
        patch_cgrad_mult = patch_cgrad * kernel
        cv2.imshow('grads_mult', scipy.ndimage.zoom(cgrad_display(patch_cgrad_mult), (10,10,1), order=0))
        # patch_dY = dY[sl]/2 + 0.5
        # patch_dX = dX[sl]/2 + 0.5
        # cv2.imshow('dY', image_to_3ch(scipy.ndimage.zoom(patch_dY, 10.0, order=0)))
        # cv2.imshow('dX', image_to_3ch(scipy.ndimage.zoom(patch_dX, 10.0, order=0)))
        while cv2.waitKey(20) & 0xff != 27:
            pass
