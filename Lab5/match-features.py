import argparse
import scipy
import numpy as np
import cv2
import sys
import colorsys
from common.show import *
from common.image import *

display_matches = False

parser = argparse.ArgumentParser(description='Matches desribed features from two images.')
parser.add_argument('imageA', metavar="IMAGE_A", type=str, help="Path to source image A.")
parser.add_argument('imageB', metavar="IMAGE_B", type=str, help="Path to source image B.")
parser.add_argument('featA', metavar="FEATURES_A", type=str, help="Path to features A.")
parser.add_argument('featB', metavar="FEATURES_B", type=str, help="Path to features B.")
parser.add_argument('output_path', metavar="OUTPUT", type=str, help="Path to otuput file.")
parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode.")
parser.add_argument('-z', '--zoom', type=float, default=1.0, help="Scale the image output for debug mode.")

try:
    args = parser.parse_args()
except SystemExit:
    sys.exit(0)

if args.debug:
    display_matches = True
zoom = args.zoom

print("zoom: %f" % zoom)

# Open image files
imageA = scipy.ndimage.imread(args.imageA)[:,:,0:3]/255.0
imageB = scipy.ndimage.imread(args.imageB)[:,:,0:3]/255.0

imageAz = scipy.ndimage.zoom(imageA, (zoom,zoom,1))
imageBz = scipy.ndimage.zoom(imageB, (zoom,zoom,1))

# Convert image to intensity
IA, IB = rgb2Y(imageA), rgb2Y(imageB)
# Normalize the image to [0,1]
IA, IB = IA/IA.max(), IB/IB.max()

featuresA, featuresB = np.loadtxt(args.featA), np.loadtxt(args.featB)

descripA, descripB = featuresA[:,4:], featuresB[:,4:]
metaA, metaB = featuresA[:,:4], featuresB[:,:4]

def dist(x1,y1,x2,y2):
    dx, dy = x1-x2, y1-y2
    return np.sqrt(dx**2 + dy**2)

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

# Temporarily, take just 100 best matches.
matches = matches[0:100]

for i,j,confidence in matches:
    fA, fB = metaA[i], metaB[j]
    xA, yA, scaleA, _ = fA
    xB, yB, scaleB, _ = fB
    print("Match: %d %d | %d %d: NNR %f" % (xA, yA, xB, yB, confidence))
    
    if display_matches:
        imageA2, imageB2 = imageAz.copy(), imageBz.copy()
        cv2.circle(imageA2, (int(xA*zoom),int(yA*zoom)), int(2.5*scaleA*zoom), (255,0,0), 2)
        cv2.circle(imageB2, (int(xB*zoom),int(yB*zoom)), int(2.5*scaleB*zoom), (255,0,0), 2)
    
        cv2.imshow('imageA', cv2_shuffle(imageA2))
        cv2.imshow('imageB', cv2_shuffle(imageB2))
        
        # Wait 20ms for Esc key, repeat.
        while cv2.waitKey(20) & 0xff != 27:
            pass
