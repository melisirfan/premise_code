# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:26:14 2017

@author: jbobin
"""
from copy import deepcopy
import numpy as np
import pyfits as pyf
import planck as plk
import healpy as hpy

def colorcor(beta, temp, freq, freqb, transb):
    """colour correction calculation """

    n_freq = len(freq)
    nex = len(temp)

    ccfact = np.zeros((n_freq, len(beta)))
    mbb = np.zeros((n_freq, len(beta)))

    for rval in range(n_freq):
        nef = len(freqb[rval])
        nu0 = np.array(freq[rval])
        qres = plk.PLK(n_freq, nex, nef).ccfact(freqb[rval].astype('float64'), \
                transb[rval].astype('float64'), temp, beta, nu0)
        ccfact[rval, :] = qres[0, :]
        mbb[rval, :] = qres[1, :]

    return ccfact, mbb

def diff_mbb(beta, temp, freq):
    """ differential of the modified black body"""

    n_freq = len(freq)
    nex = len(temp)
    nef = 0

    mbb = plk.PLK(n_freq, nex, nef).d_mbb(freq, temp, beta)

    return mbb

def get_bands(freq, btt=1e-4, locbands=None):
    """ read the planck instrumental bands """

    n_freq = len(freq)

    allfreq = []
    allbands = []

    # loading all bandpasses

    for rval in range(n_freq):

        hdu = pyf.open(locbands+"freq_avg"+str(np.int(freq[rval]))+"_red.fits")
        freq_b = hdu[0].data
        hdu.close()
        if freq[rval] < 90:
            freq_b = freq_b*1e9

        hdu = pyf.open(locbands+"bandpass_avg"+str(np.int(freq[rval]))+"_red.fits")
        trans_b = hdu[0].data
        hdu.close()

        posfreq = np.where((freq_b > btt) & (trans_b > btt*np.max(trans_b)))[0]

        allfreq.append(freq_b[posfreq])
        allbands.append(trans_b[posfreq])

    return allfreq, allbands

def applyh_pinv(hch, xin, epsi=1e-3):
    """ Apply Pseudo-Inverse of H """

    nhch = np.shape(hch)
    n_freq = nhch[0]
    nex = nhch[2]
    nef = 0
    eps = np.array([epsi]).astype('double')

    return plk.PLK(n_freq, nex, nef).applyH_Pinv(hch, xin, eps)

def lip_c(hch):
    """ Compute the Lipschitz Ct """

    nhch = np.shape(hch)
    n_freq = nhch[0]
    nex = nhch[2]
    nef = 0

    return plk.PLK(n_freq, nex, nef).LipC(hch)

def spline2(size, lval, lcval):
    """ Cubic spline """

    res = np.linspace(0, size, size+1)

    res = 2.0 * lval * res / (lcval * size)
    tab = (3.0/2.0) * 1.0 /12.0 * ((abs(res-2))**3 - 4.0 * (abs(res-1))**3 + \
            6 * (abs(res))**3 - 4.0 * (abs(res+1))**3 + (abs(res+2))**3)

    return tab

def compute_h(size, lcval):
    """ Calculate hval """

    tab1 = spline2(size, 2. * lcval, 1)
    tab2 = spline2(size, lcval, 1)
    hval = tab1/(tab2 + 0.000001)
    hval[np.int(size/(2. * lcval)):size] = 0.

    return hval

def apply_ht(hch, xin):
    """ Apply Pseudo-Inverse of H """

    nhch = np.shape(hch)
    n_freq = nhch[0]
    nex = nhch[2]
    nef = 0

    return plk.PLK(n_freq, nex, nef).applyHt(hch, xin)

def wttrans(mapval, nscale=4, lmax=128):
    """ Wavelet transform on the sphere """

    ech = 1

    taille = np.size(mapval)

    alm = almtrans(mapval, lmax=lmax)
    alm_temp = deepcopy(alm)

    lscale = deepcopy(mapval)
    nside = hpy.get_nside(mapval)

    wtr = np.zeros((taille, nscale))

    for jval in range(nscale-1):

        hch = compute_h(lmax, ech)

        alm_temp = alm_product(alm, hch)

        mval = almrec(alm_temp, nside=nside)

        hscale = lscale - mval
        lscale = mval

        wtr[:, jval] = hscale

        ech = 2 * ech

    wtr[:, nscale-1] = lscale

    return wtr

def get_all_faces(imag, nested=False):
    """ Extract 12 healpix faces """

    npix = np.shape(imag)[0]
    nside = hpy.npix2nside(npix)
    taille_face = npix/12
    cote = np.int(np.sqrt(taille_face))
    cubeface = np.zeros((12, cote, cote))

    if nested != True:
        newim = hpy.reorder(imag, r2n=True)

    else:
        newim = imag

    index = np.zeros((cote, cote))
    index = np.array([hpy.xyf2pix(nside, eex, yii, 0, True) for eex in range(nside) \
            for yii in range(nside)])
    for face in range(12):
        cubeface[face, :, :] = np.resize(newim[index + taille_face * face], (cote, cote))

    return cubeface

def put_all_faces(cubeface, nested=False):
    """ Return 12 healpix faces to the shpere """

    npix = np.size(cubeface)
    nside = hpy.npix2nside(npix)
    taille_face = npix/12
    cote = np.int(np.sqrt(taille_face))
    imag = np.zeros((npix))

    index = np.zeros((cote, cote))
    index = np.array([hpy.xyf2pix(nside, eex, yii, 0, True) for eex in range(nside) \
            for yii in range(nside)])
    for face in range(12):
        imag[index + taille_face * face] = np.resize(cubeface[face, :, :], (cote, cote))

    if nested != True:
        newim = hpy.reorder(imag, n2r=True)

    else:
        newim = imag

    return newim

def tab2alm(tab):
    """ Transform table of ALM values to healpix ALMs """

    lmax = np.int(np.shape(tab)[0]) - 1
    taille = np.int(lmax * (lmax + 3)/2) + 1
    alm = np.zeros((taille, ), dtype=complex)

    ell, mmm = hpy.sphtfunc.Alm.getlm(lmax, np.array(range(0, taille)))

    alm = np.array([np.complex(tab[ell[rval], mmm[rval], 0], tab[ell[rval], mmm[rval], 1]) \
            for rval in range(len(ell))])

    return alm

def almrec(tab, nside=512):
    """ Transform ALM values back to map """

    alm = tab2alm(tab)

    mapdat = hpy.alm2map(alm, nside, verbose=False)

    return mapdat

def alm_product(tab, filt):
    """ Multiply ALM values """

    length = np.size(filt)
    lmax = np.shape(tab)[0]

    if lmax > length:
        print "Filter length is too small"

    for rval in range(lmax):
        tab[rval, :, :] = filt[rval] * tab[rval, :, :]

    return tab

def almtrans(mapval, lmax=None):
    """ Transform map to a table of ALM values """

    if lmax is None:
        lmax = 3.* hpy.get_nside(mapval)
        print("lmax = ", lmax)

    alm = hpy.sphtfunc.map2alm(mapval, lmax=lmax)

    tab = alm2tab(alm, lmax)

    return tab

def alm2tab(alm, lmax):
    """ Tabulate the L and M values of ALMs """

    taille = np.size(alm)
    tab = np.zeros((lmax+1, lmax+1, 2))

    ell, mmm = hpy.sphtfunc.Alm.getlm(lmax, np.array(range(0, taille)))
    tab[ell, mmm, 0] = np.real(alm)
    tab[ell, mmm, 1] = np.imag(alm)

    return tab
