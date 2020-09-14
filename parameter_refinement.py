# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:05:57 2017

Code that provides samples for the single MBB model for different values of the input parameters

@author: jbobin
"""

from copy import deepcopy as dp
from multiprocessing import Queue, Process
from wavelet_tools import wt_decompose, wt_recompose, mad
import premise_utilities as pym
import numpy as np
import healpy as hp

def mbb_estimation_2p(raw, queue, freq=None, beta_init=None, temp_init=None, \
    mask=None, pinv=0, nit=100, tol=1e-6, verb=1, alpha=0.1, epsi=1):
    """gradient descent for temp and beta params, 2 out of the 3 MBB params """

    numex = np.shape(raw)
    n_ch = numex[0]
    n_rows = numex[1]
    copyraw = dp(raw)

    # Main loop

    beta = dp(beta_init)
    beta_old = dp(beta_init)

    temp = dp(temp_init)
    temp_old = dp(temp_init)

    itnum = 1

    go_on = 1

    rho = alpha

    npix = np.sqrt(n_rows).astype('int')
    data = np.zeros((2, npix, npix))
    data[0, :, :] = temp_init.reshape((npix, npix))
    data[1, :, :] = beta_init.reshape((npix, npix))

    # WT Filter

    while go_on:

        ztemp = pym.diff_mbb(beta, temp, freq)/1e9

        model = dp(ztemp[:, 0, :])
        d_mbb = dp(ztemp[:, 1:3, :])
        amp = dp(copyraw/model[0, :])

        for rval in range(n_ch):
            d_mbb[rval, 0, :] = d_mbb[rval, 0, :]*amp
            d_mbb[rval, 1, :] = d_mbb[rval, 1, :]*amp
            model[rval, :] = model[rval, :]*amp

        residual = (model - copyraw)
        mask = mask.reshape(n_ch, n_rows)

        if mask is not None:
            residual = residual * mask

        # Updating beta and temp for fixed OD
        if pinv:
            delta = pym.applyh_pinv(d_mbb, residual, epsi=epsi)

        else:
            delta = pym.apply_ht(d_mbb, residual)
            lip = pym.lip_c(d_mbb)
            rho = alpha/np.max(lip)

        temp = temp - rho*delta[0, :]
        beta = beta - rho*delta[1, :]

        # Check the convergence

        diff = np.max([np.sum(abs(beta-beta_old))/np.sum(1e-12+abs(beta_old)), \
                np.sum(abs(temp-temp_old))/(1e-12+np.sum(temp_old))])

        temp_old = dp(temp)
        beta_old = dp(beta)

        itnum += 1

        if verb:

            print('It. #', itnum, ' / ', nit, ' - diff = ', diff, ' / ', tol)

        if (itnum > nit) | (diff < tol):
            go_on = 0

    queue.put([beta, temp])

def sph_wt_filtering_loc(data, nscale=3, lmax=128, kmad=3, l_one='1', sims=None, bin_min=0, \
    remove_first=0, l_one_w=None, verb=1, ref=None):
    """ wavelet thresholding on the sphere """

    npix = hp.get_nside(data)

    mat_wt = pym.wttrans(data, nscale=nscale, lmax=lmax)

    bins = dp(bin_min)

    wei = dp(mat_wt)
    wei[:] = 1.

    if l_one_w != None:
        wei = dp(l_one_w)

    if sims != None:
        mat_wt_s = pym.wttrans(sims, nscale=nscale, lmax=lmax)

    for qval in range(nscale-1):  # Each scale
        if verb:
            print('Scale %r out of %r') %(qval, nscale-1)

        temp_w = mat_wt[:, qval]
        mws = pym.get_all_faces(temp_w)

        if sims != None:
            mwn = pym.get_all_faces(mat_wt_s[:, qval])

        for rval in range(12):  # Each face

            temp = mws[rval, :, :]

            if remove_first:
                if qval == 0:
                    temp[:, :] = 0

            if (qval == 0) or ((qval > 0)):

                if bin_min > 0:

                    stdmap = np.zeros((npix, npix))

                    nbinx = np.floor(npix/bins).astype('int')
                    nbiny = np.floor(npix/bins).astype('int')

                    for nex in range(nbinx):
                        for nyi in range(nbiny):

                            if sims == None:
                                sigma = mad(temp[nex*bins:(nex+1)*bins, nyi*bins:(nyi+1)*bins])
                            else:
                                p_im = mwn[rval, :, :]
                                sigma = np.std(p_im[nex*bins:(nex+1)*bins, nyi*bins:(nyi+1)*bins])

                            stdmap[nex*bins:(nex+1)*bins, nyi*bins:(nyi+1)*bins] = sigma

                    thrd = kmad*stdmap

                else:

                    if sims != None:
                        thrd = kmad*np.std(mwn[rval, :, :])
                    else:
                        thrd = kmad*mad(temp)

            if l_one == '1':

                temp = (temp - thrd * np.sign(temp)) * (abs(temp) - thrd > 0)

            else:

                temp = temp * (abs(temp) > thrd)

            mws[rval, :, :] = temp

        mat_wt[:, qval] = pym.put_all_faces(mws)

        bin_min = 2. * bin_min

    if ref is not None:
        mat_wt0 = pym.wttrans(ref, nscale=nscale, lmax=lmax)
        mat_wt[:, nscale-1] = mat_wt0[:, nscale-1]

    xout = np.sum(mat_wt, axis=1)

    return xout

################ MAIN ESTIMATION PROCEDURE

def run_mbb_estimation(raw, freq, temp_init, beta_init, opt_init, outer_loop=2, mask=None, kmad=3,\
    n_iter=500, nscale=3, bin_size=32, pinv=1, alpha=0.5, epsilon=0.1, \
        verb=0, tol=1e-6):
    """estimates pixel by pixels temperature and spectral index values """

    specind = dp(beta_init)
    temperature = dp(temp_init)

    d_ata0 = np.zeros((2, len(temp_init)))
    d_ata0[0, :] = temp_init
    d_ata0[1, :] = beta_init

    for outer in range(outer_loop):
        if verb:
            print('Outer Loop %r out of %r') %(outer, outer_loop)

        #get 12 facecs
        faces_b = pym.get_all_faces(specind)
        faces_t = pym.get_all_faces(temperature)
        n_ch = np.shape(raw)[0]
        npix = np.shape(raw)[1]
        nside = int(np.sqrt(npix/12.))
        faces_x = np.zeros((n_ch, 12, nside, nside))
        faces_m = np.zeros((n_ch, 12, nside, nside))

        for nch in range(0, n_ch):
            faces_x[nch, :, :, :] = pym.get_all_faces(raw[nch, :])
            faces_m[nch, :, :, :] = pym.get_all_faces(mask[nch, :])

        final_b_cube = np.zeros((12, nside, nside))
        final_t_cube = np.zeros((12, nside, nside))
        
        optAll = pym.get_all_faces(opt_init) * 10.**(20)

        for face in range(0, 12):
            if verb:
                print('Face %r') %face

            beta_face = faces_b[face, :, :]
            temp_face = faces_t[face, :, :]
            x_face = faces_x[:, face, :, :]
            m_face = faces_m[:, face, :, :]

            xnu = x_face.reshape((n_ch, nside**2))
            masking = m_face.reshape((n_ch, nside**2))

            d_ata = np.zeros((2, nside, nside))
            d_ata[0, :, :] = temp_face
            d_ata[1, :, :] = beta_face

            mcou, mwav, cpw = wt_decompose(d_ata, jscale=nscale)
            # keep only course scale
            mwav[:, :, :] = 0.
            xout = wt_recompose(d_ata, mcou, mwav, cpw, jscale=nscale)
            t_est = xout[0, :, :].astype('double')
            b_est = xout[1, :, :].astype('double')
            
            opt = optAll[face, :, :].reshape(nside**2) * blackbody(353 *1e9, te.reshape(nside**2))

            queue = Queue()

            pro = Process(target=mbb_estimation_2p, args=(xnu, queue), \
                    kwargs={'epsi':epsilon, 'freq':freq, 'opt_depth_init':opt, \
                        'pinv':pinv, 'beta_init':b_est.reshape((nside**2)), \
                            'temp_init':t_est.reshape((nside**2)), 'mask':masking, \
                                'nit':n_iter, 'tol':tol, 'verb':verb, 'alpha':alpha})
            pro.start()

            result = queue.get()
            beta1 = result[0]
            temp1 = result[1]

            print np.mean(beta1)
            print np.mean(temp1)

            pro.join()

            del pro, queue, result

            final_b_cube[face, :, :] = beta1.reshape(nside, nside)
            final_t_cube[face, :, :] = temp1.reshape(nside, nside)

        if verb:
            print'Applying sparsity to full shpere'

        final_b = pym.put_all_faces(final_b_cube)
        final_t = pym.put_all_faces(final_t_cube)

        if nside > 4000./3.:
            lmax = int(4000)
        else:
            lmax = int(nside * 3)

        if verb:
            print 'Temp first...'
        temperature = sph_wt_filtering_loc(final_t, lmax=lmax, nscale=3, kmad=kmad, \
                     l_one_w=None, bin_min=bin_size, verb=1, ref=d_ata0[0, :])

        if verb:
            print '...then beta'
        specind = sph_wt_filtering_loc(final_b, lmax=lmax, nscale=3, kmad=kmad, \
                    l_one_w=None, bin_min=bin_size, verb=1, ref=d_ata0[1, :])

    return specind, temperature
 