import numpy as np
import scipy.cluster as cluster

from scipy.spatial.distance import pdist

from hexrd.xrd import rotations as rot
from hexrd.xrd import symmetry as sym
from hexrd     import matrixutil as mutil

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import colors

d2r = np.pi/180.

qsym = sym.quatOfLaueGroup('d6h')

crds   = np.loadtxt('hfr_5M.dx-positions.dat')
compl  = np.loadtxt('compl_5M_1248.txt')

scrd = crds[compl > 0.51, :]

def rodDistance(x, y):
    nrms   = np.sqrt(np.sum(np.vstack([x, y])**2, axis=1))
    angles = 2*np.arctan(nrms)
    axes   = mutil.unitVector(np.vstack([x, y]).T)
    quats  = rot.quatOfAngleAxis(angles, mutil.unitVector(np.vstack([x, y]).T))
    return rot.misorientation(quats[:, 0].reshape(4, 1), quats[:, 1].reshape(4, 1), (qsym, ))[0]

cl = cluster.hierarchy.fclusterdata(scrd, 5*d2r, criterion='distance', metric=rodDistance)

nblobs = len(np.unique(cl))

cmap = cm.ScalarMappable(norm=norm, cmap='spectral')
norm = cm.colors.Normalize(vmin=1, vmax=nblobs)
rgba = np.array([cmap.to_rgba(i+1) for i in range(nblobs)])
hexcolors = [colors.rgb2hex(rgba[i, :]) for i in range(nblobs)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(nblobs):
    xs = scrd[cl == i, 0]
    ys = scrd[cl == i, 1]
    zs = scrd[cl == i, 2]
    ax.scatter(xs, ys, zs, c=hexcolors[i], marker='o', s=48)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
