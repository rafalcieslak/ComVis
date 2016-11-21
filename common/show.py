import scipy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# A helper printer which I use instead of print for numpy arrays.
# I don't know why, but print clearly ignores numpy output settings
# for some (large?) arrays. show() fixes this.
def show(array):
    print(np.array_str(array,suppress_small=True,precision=5))


# Verbose aliases for gamma transformation
def gamma_decode(img, gamma=2.2):
    return np.power(img, gamma) 

def gamma_encode(img, gamma=2.2):
    return np.power(img, 1.0/gamma)
    
# A hepler function for displaying images within the notebook.
# It may display multiple images side by side, optionally apply gamma transform, and zoom the image.
def show_image(imglist, c='gray', vmin=0, vmax=1, zoom=1, needs_encoding=False):
    if type(imglist) is not list:
       imglist = [imglist]
    n = len(imglist)
    first_img = imglist[0]
    dpi = 77 # pyplot default?
    plt.figure(figsize=(first_img.shape[0]*zoom*n/dpi,first_img.shape[0]*zoom*n/dpi))
    for i in range(0,n):
        img = imglist[i]
        if needs_encoding:
            img = gamma_encode(img)
        plt.subplot(1,n,i + 1)
        plt.tight_layout()    
        plt.axis('off')
        if len(img.shape) == 2:
           img = np.repeat(img[:,:,np.newaxis],3,2)
        plt.imshow(img, interpolation='nearest')


# Creates a simple 3d plot from a set of points. Ensures that a uniform scaling
# is applied to all axes, so that 3d spatial data is not stretched
def plot_points(data3d):
    C = np.array([0,0,0])
    fig = plt.figure()
    sp = fig.add_subplot(111, projection='3d')
    # sp.axis('equal') # Does not work for 3d.
    X,Y,Z = data3d[:,0], data3d[:,1], data3d[:,2]
    """
    X = np.insert(X, 0, C[0])
    Y = np.insert(Y, 0, C[1])
    Z = np.insert(Z, 0, C[2])
    """ 
    # Correction for unit scaling
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()*0.5
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    sp.set_xlim(mid_x - max_range, mid_x + max_range)
    sp.set_ylim(mid_y - max_range, mid_y + max_range)
    sp.set_zlim(mid_z - max_range, mid_z + max_range)
    
    sp.scatter(X[1:],Y[1:],Z[1:], c='b', marker='o', s=2)
    # sp.scatter(C[0], C[1], C[2], c='r', marker='x')
    plt.show()
