import numpy as np
from common import *

np.set_printoptions(precision=5, suppress=True, linewidth=200)

data2d_A_norm = np.loadtxt("data/task12/pts2d-norm-pic_a.txt").reshape((-1,2))
data2d_A      = np.loadtxt("data/task12/pts2d-pic_a.txt")     .reshape((-1,2))
data3d_norm   = np.loadtxt("data/task12/pts3d-norm.txt")      .reshape((-1,3))
data3d        = np.loadtxt("data/task12/pts3d.txt")           .reshape((-1,3))

### Dataset choice:
data2d, data3d = data2d_A_norm, data3d_norm
normalize = False
### or
# data2d, data3d = data2d_A, data3d
# normalize = True

if normalize:
    # Normalize data
    data2d_translate, data2d_scale, data2d = normalize_data(data2d)
    data3d_translate, data3d_scale, data3d = normalize_data(data3d)
else:
    # Already normalized
    data2d_translate = np.asarray([0.0, 0.0])
    data2d_scale = 1.0
    data2d = data2d_A_norm
    data3d_translate = np.asarray([0.0, 0.0, 0.0])
    data3d_scale = 1.0
    data3d = data3d_norm
    
P = estimate_P(data2d, data3d)
print("Projection matrix:")
print(P)

r = calculate_residual(data2d_A_norm, data3d_norm, P)
print("Total residual:")
print(r)

K, R, T, C = decompose_P(P)
print("K:")
print(K)
print("R:")
print(R)
print("T:")
print(T)
print("C:")
print(C)

# Interactive 3d plot
plot_points_camera(data3d_norm * data3d_scale - data3d_translate, C * data3d_scale - data3d_translate)
