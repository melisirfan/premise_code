# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:11:33 2016

@author: jbobin
"""

import os
from copy import deepcopy as dp
import sparse2d as sp2
import numpy as np
import pyfits as pyf

LOCEXEC = 'exec/'

def wt_multi_trans(imag, nscale=4, wt_type=0):
    """ multi-wavelet transform """

    nameimag = 'xx_imag.fits'
    hdu = pyf.PrimaryHDU(imag)
    hdulist = pyf.HDUList([hdu])
    hdulist.writeto(nameimag)

    #Undecimated wavelet transform
    if wt_type == 0:
        opt = ' -n'+str(nscale)

    if wt_type == 1:
        opt = '-t24 -L'+ ' -n' + str(nscale)

    if wt_type == 2:
        opt = '-t25 -L'+ ' -n' + str(nscale)

    com = LOCEXEC+'/./mr_multi_transform ' + opt + ' ' + nameimag + ' ' +  'default'
    os.system(com)

    data_transf = pyf.open("default.fits")
    dat = data_transf[0].data
    data_transf.close()

    com2 = "rm "+ nameimag
    os.system(com2)

    return dat

def wt_multi_recons(dat, nscale=4, wt_type=0):
    """ multi-wavelet reconstruction """

    file = pyf.open('default.fits', mode='update')
    file[0].data = dat
    file.flush()
    file.close()

    # Undecimated wavelet transform
    if wt_type == 0:
        opt = ' -n' + str(nscale)

    if wt_type == 1:
        opt = '-t24 -L' + ' -n' + str(nscale)

    if wt_type == 2:
        opt = '-t25 -L' + ' -n' + str(nscale)

    com = LOCEXEC+'/./mr_multi_transform -r ' + opt + ' default.fits output'
    os.system(com)
    data_transf = pyf.open("output.fits")
    xout = data_transf[0].data
    data_transf.close()

    com2 = "rm output.fits"
    os.system(com2)
    com2 = "rm default.fits"
    os.system(com2)

    return xout

def get_pos_band(dat):
    """ coefficient positions """

    nelem = np.shape(dat)[0]
    pos = np.zeros((3, 1))

    go_on = 1
    counter = 3

    while go_on:

        nnx = np.int(dat[counter, 0])
        counter += 1
        nny = np.int(dat[counter, 0])
        counter += 1
        scale = np.int(dat[counter, 0])
        counter += 1

        ind = np.zeros((3, 1))
        ind[0, 0] = counter
        ind[1, 0] = counter + nnx * nny
        ind[2, 0] = scale

        pos = np.concatenate((pos, ind), axis=1)

        counter = counter + nnx * nny

        if counter > nelem - 1:
            go_on = 0

    pos = pos[:, 1::]

    return pos

def get_mwt_band(dat, pos, band):
    """ wavelet bands """

    numz = np.shape(dat)[1]

    ind = np.where(pos[2, :] == band)[0]

    lind = len(ind)
    nupx = np.int(dat[np.int(pos[0, ind[0]]) - 3, 0])
    nupy = np.int(dat[np.int(pos[0, ind[0]]) - 2, 0])

    coef = np.zeros((nupx, nupy, lind, numz))

    for zval in range(0, numz):
        for rval in range(0, lind):
            sta = np.int(pos[0, ind[rval]])
            enn = np.int(pos[1, ind[rval]])
            coef[:, :, rval, zval] = dat[sta:enn, zval].reshape((nupx, nupy))

    return coef

def put_mwt_band(dat, coef, pos, band):
    """ return wavelet bands """

    numz = np.shape(dat)[1]
    cout = dp(dat)

    ind = np.where(pos[2, :] == band)[0]

    lind = len(ind)
    nupx = np.int(dat[np.int(pos[0, ind[0]]) - 3, 0])
    nupy = np.int(dat[np.int(pos[0, ind[0]]) - 2, 0])

    for zval in range(0, numz):
        for rval in range(0, lind):
            sta = np.int(pos[0, ind[rval]])
            enn = np.int(pos[1, ind[rval]])
            cout[sta:enn, zval] = coef[:, :, rval, zval].reshape((nupx * nupy))

    return cout

def mad(xin=0):
    """ mean absolute deviation """

    zed = np.median(abs(xin - np.median(xin)))/0.6735

    return zed

def wt_decompose(data, jscale=1):
    """ perform wavelet decomposition """

    nch = np.shape(data)[0]
    nxcol = np.shape(data)[1]
    nycol = np.shape(data)[2]

    mat_wt = np.zeros((nch, nxcol * nycol, jscale))
    mat_c = np.zeros((nch, nxcol * nycol))

    dat = wt_multi_trans(data, nscale=jscale, wt_type=0)

    pos = get_pos_band(dat)

    for rval in range(jscale-1):

        coef = get_mwt_band(dat, pos, rval)

        for qval in range(nch):

            mat_wt[qval, :, rval] = coef[:, :, 0, qval].reshape((nxcol * nycol))

    coef = get_mwt_band(dat, pos, jscale-1)

    for qval in range(nch):

        mat_c[qval, :] = coef[:, :, 0, qval].reshape((nxcol * nycol))

    return mat_c, mat_wt, dat

def wt_recompose(data, mat_c, mat_wt, dat, jscale=1):
    """ perform wavelet recomposition """

    nch = np.shape(data)[0]
    nxcol = np.shape(data)[1]
    nycol = np.shape(data)[2]

    coef = np.zeros((nxcol, nycol, 1, nch))

    pos = get_pos_band(dat)

    for rval in range(jscale-1):

        for qval in range(nch):

            coef[:, :, 0, qval] = mat_wt[qval, :, rval].reshape((nxcol, nycol))

        dat = put_mwt_band(dat, coef, pos, rval)

    for qval in range(nch):

        coef[:, :, 0, qval] = mat_c[qval, :].reshape((nxcol, nycol))

    dat = put_mwt_band(dat, coef, pos, jscale-1)

    xout = wt_multi_recons(dat, nscale=jscale, wt_type=0)

    return xout

def forward(data, consts=[0.0625, 0.25, 0.375, 0.25, 0.0625], jscale=1):
    """ multichannel starlet decomposition """

    nsz = np.shape(data)
    csz = np.size(consts)

    trans = sp2.Starlet2D(nsz[1], nsz[2], nsz[0], jscale, csz).forward_omp(np.real(data), \
                np.array(consts))

    return trans

def backward(data, consts=[0.0625, 0.25, 0.375, 0.25, 0.0625]):
    """ multichannel starlet recomposition """

    nsz = np.shape(data)
    csz = np.size(consts)

    rec = sp2.Starlet2D(nsz[1], nsz[2], nsz[0], nsz[3]-1, csz).backward_omp(np.real(data))

    return rec
