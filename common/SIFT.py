import scipy
import scipy.ndimage
import numpy as np
import cv2
import sys
import colorsys
from .image import *
from .show import show

PATCH_R = 20
PAD_SIZE = PATCH_R*2 + 1
    
step_by_step_debug = False

kernel_cache = {}
class KernelGen:
    def __init__(self, sigma, R):
        self.sigma = sigma
        self.R = R
    def __call__(self, data):
        data = data - self.R
        dist = (data**2).sum(axis=1)
        # Dist is already squared!
        return np.exp(-(dist)/(2 * (self.sigma**2)))

def get_gaussian_kernel(sigma):
    if sigma in kernel_cache:
        return kernel_cache[sigma]
    #print("No kernel %f in cache, generating..." % sigma)
    kernel = img_gen(KernelGen(sigma, PATCH_R), (PATCH_R*2+1,PATCH_R*2+1,1))[:,:,0]
    kernel_cache[sigma] = kernel
    return kernel


class ExtractRotate:
    def __init__(self, pos, angle, scale, R):
        self.pos = np.array([pos[0], pos[1]])
        self.R = R
        th = np.deg2rad(-angle)
        self.M = np.array([np.cos(th), -np.sin(th), np.sin(th), np.cos(th)]).reshape((2,2)) * scale
    def __call__(self, data):
        off = data - self.R
        off = np.einsum("ab,xb->xa", self.M, off)
        res = self.pos + off
        return res

def extract_feature(img, pos, angle=0, scale=1, R=PATCH_R):
    x, y = pos
    if np.abs(angle) < 1 and scale==1:
        sl = (slice(y-PATCH_R, y+PATCH_R+1), slice(x-R,x+R+1))
        return img[sl]
    return img_transform_1ch(img, ExtractRotate(pos, angle, scale, R), target_shape=(R*2 + 1, R*2 + 1))


def describe(image, features):    
    # Convert image to intensity
    I = rgb2Y(image)
    # Normalize the image to [0,1]
    I = I / I.max()

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

    kernel16 = img_gen(KernelGen(8/3, 7.5), (16,16,1))[:,:,0]
    
    BIN_N = 36
    BIN_SIZE = 360/BIN_N
    DOMINANT_ANGLE_PEAK_TRESHOLD = 0.9

    # Iterate over features
    descrips = []
    for i,f in enumerate(features):
        x, y, scale = f
        x, y = int(x), int(y)
        if step_by_step_debug:
            print("Feature %d: %d %d (%f)" % (i, x, y, scale))

        patch = extract_feature(I, [x,y])
        patch_cgrad = extract_feature(cgrad, [x,y])
        patch_gradangle = extract_feature(gradangle, [x,y])

        kernel = get_gaussian_kernel(scale * 1.5)
        w = kernel * np.abs(patch_cgrad)

        # Compute histogram
        hist, bins = np.histogram(patch_gradangle, BIN_N + 1, (-180 - BIN_SIZE/2, 180 + BIN_SIZE/2), weights=w, density=False)
        hist[0] += hist[-1]
        hist = hist[0:BIN_N]
        hist_s = np.argsort(hist)
        bins += BIN_SIZE/2

        bins = bins[hist_s][::-1]
        hist = hist[hist_s][::-1]

        last_v = hist[0]
        dominant_angles = []
        for i, v in enumerate(hist):
            if v < DOMINANT_ANGLE_PEAK_TRESHOLD * last_v:
                break
            dominant_angles += [bins[i]]
            last_v = v

        for angle in dominant_angles:
            rotated_patch = extract_feature(I, [x,y], angle, scale * 3 / 64, R=64)
            # patch16 = extract_feature(I, [x,y], angle, scale * 3 / 8, R=8)
            patch16 = decimate(rotated_patch, (16,16))

            dY16 =  scipy.ndimage.convolve1d(patch16, gradient_kernel, axis=0)[:16,:16]
            dX16 = -scipy.ndimage.convolve1d(patch16, gradient_kernel, axis=1)[:16,:16]
            cgrad16 = dX16 + 1j * dY16
            angle16 = np.angle(cgrad16, deg=True)

            # Smoothing weights
            cgrad16 = cgrad16 * kernel16
            magn16 = np.abs(cgrad16)

            d = []
            for Qy in [0,1,2,3]:
                for Qx in [0,1,2,3]:
                    sy = slice(4*Qy, 4*Qy + 4)
                    sx = slice(4*Qx, 4*Qx + 4)
                    Qmagn16 = magn16[sy,sx]
                    Qangle16 = angle16[sy,sx]
                    Qhist, _ = np.histogram(Qangle16, 9, (-180 - 45, 180 + 45), weights=Qmagn16, density=False)
                    Qhist[0] += Qhist[-1]
                    Qhist = Qhist[:8]
                    d += list(Qhist)

            assert(len(d) == 128)
            d = np.asarray(d).reshape(16,8)
            d = d/d.max();
            d[d > 0.2] = 0.2
            d = d/d.max();
            descrips += [(x,y,scale, angle, d)]

            """
            if step_by_step_debug:
                print("Dominant angle: %d" % angle)
                show(hist)
                show(bins)
                cv2.imshow('patch', image_to_3ch(scipy.ndimage.zoom(patch, 10.0, order=0)))
                cv2.imshow('kernel', image_to_3ch(scipy.ndimage.zoom(kernel, 10.0, order=0)))
                cv2.imshow('grads', scipy.ndimage.zoom(cgrad_display(patch_cgrad), (10,10,1), order=0))
                patch_cgrad_mult = patch_cgrad * kernel
                cv2.imshow('grads_mult', scipy.ndimage.zoom(cgrad_display(patch_cgrad_mult), (10,10,1), order=0))
                cv2.imshow('rotated_patch', image_to_3ch(scipy.ndimage.zoom(rotated_patch, 3.0, order=0)))
                cv2.imshow('patch16', image_to_3ch(scipy.ndimage.zoom(patch16, 12.0, order=0)))
                cv2.imshow('grad16', scipy.ndimage.zoom(cgrad_display(cgrad16), (10,10,1), order=0))
                cv2.imshow('kernel16', image_to_3ch(scipy.ndimage.zoom(kernel16, (10,10), order=0)))
                show(d)

                while cv2.waitKey(20) & 0xff != 27:
                    pass
            """
    # Manually flatten descrip list
    d2 = []
    for x,y,scale,angle,DESCRIPTOR in descrips:
        l = np.asarray([x-PAD_SIZE,y-PAD_SIZE,scale,angle])
        d2 += [np.hstack([l,DESCRIPTOR.ravel()])]
    d2 = np.asarray(d2)

    print(d2.shape)
    return d2


def match2D(featuresA, featuresB, N=100):
    descripA, descripB = featuresA[:,4:], featuresB[:,4:]
    metaA, metaB = featuresA[:,:4], featuresB[:,:4]

    matches = []
    for i,descrip in enumerate(descripA):
        diff = np.abs(descripB - descrip).sum(axis=1)
        indices = np.argpartition(diff, 2)[0:2]
        values = diff[indices]
        # Make sure value[0] is the smallest
        if values[1] < values[0]:
            indices[0], indices[1] = indices[1], indices[0]
            values[0], values[1] = values[1], values[0]
        # x1, y1, _, _ = metaA[i]
        # x21, y21, _, _ = metaA[indices[0]]
        # x22, y22, _, _ = metaA[indices[1]]
        # d0, d1 = dist(x1,y1,x21,y21), dist(x1,y1,x22,y22)
        confidence = values[0]/values[1]
        # confidence = d0/d1
        match = (i, indices[0], confidence)
        matches += [match]

    matches = sorted(matches, key=lambda x: x[2], reverse=False)
        
    # Take some best matches.
    matches = matches[0:N]

    result = []
    for i,j,confidence in matches:
        fA, fB = metaA[i], metaB[j]
        xA, yA, scaleA, _ = fA
        xB, yB, scaleB, _ = fB
        result += [(xA, yA, xB, yB)]
    return result
