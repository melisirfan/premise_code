# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:46:32 2017

@author: jbobin
"""

import scipy.linalg as scl
import numpy as np
import wavelet_tools as pys

def m_filtering_one_scale(data, nuisance, kmad=3, min_bin=16, overlap=0.5):
    """ function to filter data using a mixture of the GNILC method and sparsity """

    npix = np.shape(data)[1]
    nch = np.shape(data)[0]

    bin_size = npix

    # Filtering
    bin_size = np.int(min_bin)

    wind = np.zeros((bin_size, bin_size))

    asig = np.log(2.)*(4./bin_size)**2

    for rr1 in range(bin_size):
        for rr2 in range(bin_size):
            wind[rr1, rr2] = np.exp(-asig * (np.double(bin_size/2.- rr1)**2. + \
                                np.double(bin_size/2.- rr2)**2.))

    #covariance matrix for noise coefficients
    wrr = nuisance.reshape((nch, npix**2))
    rnh = 1./(npix**2) * np.dot(wrr, wrr.T)
    irnh = scl.inv(scl.sqrtm(rnh))

    xout = np.zeros((nch, npix, npix))

    nbin = np.int(npix/bin_size)
    nbino = np.int(np.double(nbin)/overlap)

    mask = np.zeros((npix, npix))

    for rxx in range(nbino-1):
        for ryy in range(nbino-1):

            xstart = np.int(rxx * bin_size * overlap)
            xend = xstart + bin_size
            ystart = np.int(ryy * bin_size * overlap)
            yend = ystart + bin_size

            xdat = data[:, xstart:xend, ystart:yend].reshape((nch, bin_size**2))

            # Whitening the data
            xrr = np.dot(irnh, xdat)
            nus = 1./(bin_size**2) * np.dot(xrr, xrr.T)
            inus = scl.inv(nus)

            nus_u, nus_r, _ = np.linalg.svd(nus)
            nus_r = np.real(nus_r)

            # Marcenko-Pastur distribution
            lamb = 1. + np.sqrt(np.double(nch)/bin_size**2)

            cind = np.where(nus_r > lamb)[0]
            n_elem = np.min([len(cind), nch])

            if n_elem > 1:

                pmat = nus_u[:, 0:n_elem].reshape((nch, n_elem))
                fmix = np.dot(rnh, pmat)

                if n_elem == 1:
                    pfm = 1/np.dot(fmix.reshape((1, nch)), np.dot(inus, fmix.reshape((nch, 1)))) \
                            * np.dot(fmix.T, inus)
                else:
                    pfm = np.dot(np.linalg.inv(np.dot(fmix.T, np.dot(inus, fmix))), \
                            np.dot(fmix.T, inus))

                sig = np.dot(pfm, xdat)
                noi = np.dot(pfm, wrr)

                for qval in range(n_elem):
                    sig[qval, :] = sig[qval, :] * (abs(sig[qval, :] - np.median(sig[qval, :])) > \
                                    kmad * np.std(noi[qval, :]))

                res = np.dot(fmix, sig).reshape((nch, bin_size, bin_size))

            for zval in range(nch):
                xout[zval, xstart:xend, ystart:yend] = xout[zval, xstart:xend, ystart:yend] \
                                                        + wind * res[zval, :, :]

        mask[xstart:xend, ystart:yend] = mask[xstart:xend, ystart:yend] + wind

    for qval in range(nch):
        xout[qval, :, :] = xout[qval, :, :] / mask

    return xout

def multi_filtering(data, nuisance, jval=5, bins=32, overlap=0.5, kmad=0):
    """ processes data to be filtered """

    nch = np.shape(data)[0]
    npix = np.shape(data)[1]

    mwn = pys.forward(nuisance, jscale=jval)
    mwd = pys.forward(data, jscale=jval)

    min_bin = bins

    for jind in range(jval):

        xout = m_filtering_one_scale(mwd[:, :, :, jind], mwn[:, :, :, jind], kmad=kmad, \
                min_bin=min_bin, overlap=overlap)

        min_bin = np.min([np.min([npix, np.shape(data)[2]]), 2. * min_bin])

        noi = mwn[:, :, :, jind].reshape(nch, npix**2)
        sig = (mwd[:, :, :, jind] - xout).reshape(nch, npix**2)

        nus = 1./npix**2 * np.dot(noi, noi.T)
        inus2 = scl.sqrtm(scl.inv(nus))

        sig = np.dot(inus2, sig)
        prob = np.sqrt(1./nch * np.sum(sig*sig, axis=0))
        ind = np.where(prob < 2.7)[0]

        if len(ind) > 0:
            for qval in range(nch):
                temp = (mwd[qval, :, :, jind] - xout[qval, :]).reshape(npix**2)
                temp[ind] = 0.
                xout[qval, :, :] += temp.reshape(npix, npix)

        mwd[:, :, :, jind] = xout

    return pys.backward(mwd)
    