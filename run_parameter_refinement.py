"""
Created on Tue 18th July 2017

@author: mirfan
"""

import numpy as np
import healpy as hp
import parameter_refinement as paramref

def blackbody(freq, temp, CST):
    ''' calculate black body intensity from temp and frequency (Hz) '''
    bb = ((2. * CST["plancks"] * freq**3.)/CST["light"]**2.) * \
        (1./(np.exp((CST["plancks"]* freq)/(CST["kbolt"] * temp)) - 1))
    return bb
    
CST = { "C": 2.99792458e+08, "H": 6.6260755e-34, "HBAR": 1.0545726691251021e-34, \
            "K": 1.3806580e-23, "TCMB": 2.7255 }
NSIDE = 2048
NPIX = 12*NSIDE**2
X_GHZ = np.array([353., 545., 857., 3000.])
CIBOFF = np.array([0.1256, 0.3310, 0.6436, 0.6570])

#colour corrected total flux maps
flux = np.zeros((4, NPIX))
mask = np.zeros((4, NPIX))

for ival in range(0, 4):
    flux[ival, :] = hp.fitsfunc.read_map('totalDustFlux_CMBpsCIBnoise_cc.fits', \
                        nest=False, field=ival)
    mask[ival, :] = hp.fitsfunc.read_map('totalMask.fits', nest=False, field=ival)

TEMP = hp.fitsfunc.read_map('quadtreeDustTemp.fits', nest=False, field=0)
BETA = hp.fitsfunc.read_map('quadtreeDustBeta.fits', nest=False, field=0)

#remove offset
flux[0,:] = flux[0,:] - CIBOFF[0]
flux[1,:] = flux[1,:] - CIBOFF[1]
flux[2,:] = flux[2,:] - CIBOFF[2]
flux[3,:] = flux[3,:] - CIBOFF[3]

cc = np.zeros((4, NPIX))
fbb, tbb = pym.get_bands([353, 545, 857], btt=1e-4, locbands="bands/")
cc[0:3,:], _ = pym.colorcor(np.array([BETA]), np.array([TEMP]), X_GHZ[0:3], fbb, tbb)
cc[3,:] = iris_cc(np.array(TEMP), np.array(BETA))

# #colour correct
for ii in range (0, 4):
    flux[ii, :] = flux[ii, :] * cc[ii,:]

OPT_INIT = hp.fitsfunc.read_map('quadtreeDustTau.fits')

mask[0, :][np.where(flux[0, :] < 0.)[0]] = 0.
mask[1, :][np.where(flux[1, :] < 0.)[0]] = 0.
mask[2, :][np.where(flux[2, :] < 0.)[0]] = 0.
mask[3, :][np.where(flux[3, :] < 0.)[0]] = 0.

beta_tot, temp_tot = paramref.run_mbb_estimation(flux, X_GHZ, TEMP, BETA, OPT_INIT, outer_loop=2, \
    mask=MASK, kmad=2., n_iter=500, nscale=5, bin_size=32, alpha=0.5, epsilon=1e-3, \
        verb=1, tol=1e-5)

con = np.array([blackbody(freqs[val]*10**9, myTemp, CST) * (freqs[val]/353.)**myBeta \
        * 10**(20.) / cc[val, :] for val in range(0,4)])
        
con = con[2:4]
data = data[2:4]
noise = noise[2:4]

npix = 
myTau = np.zeros((NPIX))
myNoise = np.zeros((NPIX))
for ii in range(NPIX):
    if ii % 0.5e6 == 0:
        print '%r out of %r' %(ii, NPIX)
    myTau[ii] = (1./np.sum(con[:,ii]**2 / noise[:, ii]**2)) * (np.sum((con[:,ii] *data[:,ii])/ noise[:, ii]**2))
    myNoise[ii] = (1./np.sum(con[:,ii]**2 / noise[:, ii]**2)) * (np.sum((con[:,ii] * noise[:,ii]) / noise[:, ii]**2))

tau_tot = SPH_WT_filtering_loc(myTau, nscale=2, lmax=4000, kmad=3, perscale=1, L1_W=None, \
            bin_min=32, verb=1, sims=myNoise, Ref=myTau)
            
hp.fitsfunc.write_map('refinedDustBeta.fits', beta_tot, nest=False)
hp.fitsfunc.write_map('refinedDustTemp.fits', temp_tot, nest=False)
hp.fitsfunc.write_map('refinedDustTau.fits', tau_tot, nest=False)
