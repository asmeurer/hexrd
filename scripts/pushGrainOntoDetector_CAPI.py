import sys, os, time
import numpy as np

import matplotlib
from   matplotlib import pyplot as plt

import distortion as dFuncs
# import fitting
import transforms as xf
import transforms_CAPI as xfcapi

# Module Data
d2r = np.pi/180.
r2d = 180./np.pi

bVec_ref    = xfcapi.bVec_ref
eta_ref     = xfcapi.eta_ref

idxFile = './ruby_4537-8_log.txt'

gtable  = np.loadtxt(idxFile, delimiter='\t')
idx     = gtable[:, 0] >= 0
hkls    = gtable[idx, 2:5]
ijo     = gtable[idx, -3:]
xyo_det = np.vstack([0.2*(ijo[:, 1] - 1024), 0.2*(1024 - ijo[:, 0]), d2r*ijo[:, 2]]).T
hkls    = gtable[idx, 2:5]

# input parameters
wavelength = 0.153588                     # Angstroms (80.725keV)

bMat = np.array( [ [  2.10048731e-01,   0.00000000e+00,   0.00000000e+00],
                   [  1.21271692e-01,   2.42543383e-01,   0.00000000e+00],
                   [  0.00000000e+00,   0.00000000e+00,   7.69486476e-02] ] )


rMat_d = xfcapi.makeDetectorRotMat( np.array([ 0.0011546340766314521,
                                               -0.0040527538387122993,
                                               -0.0026221336905160211]) ) 
tVec_d = np.array( [ [   -1.44904 ],
                     [   -3.235616],
                     [-1050.74026 ] ] )

chi    = -0.0011591608938627839
tVec_s = np.array([ [-0.15354144],
                    [ 0.        ],
                    [-0.23294777] ] )

rMat_c = xfcapi.makeRotMatOfExpMap(np.array( [ [ 0.66931818],
                                               [-0.98578066],
                                               [ 0.73593251] ] ) )
tVec_c = np.array( [ [ 0.07547626],
                     [ 0.08827523],
                     [-0.02131205] ] )

# ######################################################################
# Calculate pixel coordinates
#
start = time.clock()                      # time this

pvec  = 204.8 * np.linspace(-1, 1, 2048)
dcrds = np.meshgrid(pvec, pvec)
XY    = np.ascontiguousarray(np.vstack([dcrds[0].flatten(), dcrds[1].flatten()]).T)
dangs = xfcapi.detectorXYToGvec(XY, rMat_d, xfcapi.makeOscillRotMat(np.array([chi, 0.])), 
                                tVec_d.flatten(), tVec_s.flatten(), tVec_c.flatten(), 
                                beamVec=bVec_ref.flatten(),etaVec=np.array([1.0,0.0,0.0]))

tTh_d = dangs[0][0]
eta_d = dangs[0][1]

elapsed = (time.clock() - start)

print "Generation of pixel coords took %f seconds" % (elapsed)
#
#
# ######################################################################



# ######################################################################
# Generate angles for an orientation, find pixels on detector
#
start = time.clock()                      # time this

# oscillation angle arrays
oangs0, oangs1 = xfcapi.oscillAnglesOfHKLs(hkls, chi, rMat_c, bMat, wavelength, 
                                           beamVec=bVec_ref, etaVec=eta_ref)

angList       = np.vstack([oangs0, oangs1])
angList[:, 1] = xfcapi.mapAngle(angList[:, 1])
angList[:, 2] = xfcapi.mapAngle(angList[:, 2])

omeMin = d2r * -60.; omeMax = d2r *  60.
omeMask = xf.validateAngleRanges(angList[:, 2], np.array([omeMin]), np.array([omeMax]))

allAngs_m = angList[omeMask, :]

allAngs_omeSort = allAngs_m[np.argsort(allAngs_m[:, 2]), :]

# duplicate HKLs
allHKLs   = np.vstack([hkls, hkls])
allHKLs_m = allHKLs[omeMask, :]

nRefl    = sum(omeMask)
hkl_xy   = np.empty((nRefl, 2))
tThTol   = d2r * 0.15
etaTol   = d2r * 1.00
omeTol   = d2r * 1.00
ij_lists = []
for i in range(nRefl):
    rMat_s = xfcapi.makeOscillRotMat( np.array([ chi, allAngs_m[i, 2] ]) )
    hkl_xy[i, :] = xfcapi.gvecToDetectorXY(np.dot(bMat, allHKLs_m[i, :].reshape(3, 1)).T,
                                           rMat_d, rMat_s, rMat_c, 
                                           tVec_d, tVec_s, tVec_c, 
                                           beamVec=bVec_ref).flatten()
    tTh_in = np.logical_and(tTh_d >= allAngs_omeSort[i, 0] - tThTol,
                            tTh_d <= allAngs_omeSort[i, 0] + tThTol)
    tTh_w = np.where(tTh_in)
    eta_in = xfcapi.angularDifference(allAngs_omeSort[i, 1]*np.ones_like(eta_d[tTh_w]), eta_d[tTh_w]) <= etaTol
    ij_lists.append(tTh_w[0][np.where(eta_in)])
    pass

elapsed = (time.clock() - start)

print "Generation of IJ windows took %f seconds" % (elapsed)
#
#
# ######################################################################

for i in range(nRefl):
    tmpXY = XY[ij_lists[i], :]
    plt.plot(tmpXY[:, 0], tmpXY[:, 1], 'r.')
plt.plot(hkl_xy[:, 0], hkl_xy[:, 1], 'b*')

# omega angles to frame number
omeStart = -60. 
omeDel   =   0.25
nFrames  = 480

omeRange_l = omeStart + omeDel * np.arange(0, nFrames)

frame_number = lambda x : np.where(omeRange_l - x <= 0)[0][-1]

frameNumber = np.zeros(nRefl)
for i in range(nRefl):
    frameNumber[i] = frame_number(allAngs_omeSort[i, 2] * r2d)
