"""
Created on Mon 12th Dec 2016

@author: mirfan
"""

import numpy as np
import healpy as hp
from iris_cc import iris_cc
import thermaldust_filtering as mfil
from lmfit import Minimizer, Parameters
from quadtree_chi import QuadTreeChi
from quadtree128 import QuadTree128
from wavelet_tools import wt_decompose
import premise_utilities as pym
import scipy.linalg as scl

class HfiModelFitting(object):
    """ Fits a modified blackbody model to the planck HFI and IRIS data """

    def __init__(self):
        """ define constants and setup arrays """

        self.freqs = np.array([353., 545., 857., 3000.])
        self.size = 2048
        self.redwn = 4
        self.plancks = 6.626*10.**(-34.)
        self.light = 2.99792458 * 10.**(8.)
        self.kbolt = 1.381*10.**(-23.)
        self.locbands = "bands/"
        self.ciboff = np.array([0.1256, 0.3310, 0.6436, 0.6570])

        self.redchi = np.zeros((self.redwn-1, self.size, self.size))
        self.show_tau = np.zeros((12, self.size, self.size))
        self.show_temp = np.zeros((12, self.size, self.size))
        self.show_beta = np.zeros((12, self.size, self.size))
        self.show_patches = np.zeros((12, self.size, self.size))

    def resid_one_beta(self, params, freq, data, **kwargs):
        """ The single MBB model with one beta """

        noise = kwargs['noise']
        tau = params['tau']
        temp = params['temp']
        beta = params['beta']

        x_ghz = np.array(freq)
        freq = np.array(freq) * 1.e9
        blackbody_fir = ((2. * self.plancks * freq**3.)/self.light**2.) * \
                        (1./(np.exp((self.plancks* freq)/(self.kbolt * temp)) - 1))

        #turn from intensity into MJy/Str
        model_mjystr = tau * blackbody_fir * (x_ghz/353.)**beta * 10.**26. * 10.**-6.

        #data not colour corrected so need to apply inverse cc to the model used for fitting
        model_corr = np.zeros(len(freq))
        lowind = np.where(x_ghz < 1000.)
        highind = np.where(x_ghz > 1000.)

        fbb, tbb = pym.get_bands([353, 545, 857], btt=1e-4, locbands=self.locbands)
        ccfact, _ = pym.colorcor(np.array([beta]), np.array([temp]), x_ghz[lowind], fbb, tbb)
        model_corr[lowind] = model_mjystr[lowind] / ccfact.T
        model_corr[highind] = model_mjystr[highind] / iris_cc(np.array(temp), np.array(beta))

        weights = 1./noise**2

        return weights * (model_corr - data)

    def  dust_model(self, freq, tau, temp, beta):
        """ The single MBB model """

        x_ghz = np.array(freq)
        freq = np.array(freq) * 1.e9
        blackbody = ((2. * self.plancks * freq**3.)/self.light**2.) * \
                        (1./(np.exp((self.plancks * freq)/(self.kbolt * temp)) - 1))

        #turn from intensity into MJy/Str
        model_mjystr = tau * blackbody * (x_ghz/353)**beta * 10.**26. * 10.**-6.

        #data not colour corrected so need to apply inverse cc to the model used for fitting
        model_corr = np.zeros(len(freq))
        lowind = np.where(x_ghz < 1000.)
        highind = np.where(x_ghz > 1000.)

        fbb, tbb = pym.get_bands([353, 545, 857], btt=1e-4, locbands=self.locbands)
        ccfact, _ = pym.colorcor(np.array([beta]), np.array([temp]), x_ghz[lowind], fbb, tbb)
        model_corr[lowind] = model_mjystr[lowind] / ccfact.T
        model_corr[highind] = model_mjystr[highind] / iris_cc(np.array(temp), np.array(beta))

        return model_corr

    def chicov(self, wns, rnus, size):
        """ get the chi squared using the covariance matrix """
        whitten = np.zeros(size*size)
        for jump in range(0, size*size):
            whitten[jump] = np.dot(wns[:, jump], np.dot(scl.inv(rnus), wns[:, jump]))
        whitten = whitten.reshape((size, size))

        return whitten

    def loop_over_patches(self, best_patches, flux, contamins, faceval, rnus):
        """ fit mean signal in patch over full range of freqs """

        params = Parameters()
        params.add('tau', value=9.6e-7, min=1e-9, max=1e-3)
        params.add('temp', value=21, min=10, max=30)
        params.add('beta', value=1.55, min=1.0, max=2.2)

        lengthp = np.shape(best_patches)[0]
        allsigflux = np.zeros((len(self.freqs), lengthp))
        allcontamins = np.zeros((len(self.freqs), lengthp))

        for val in range(0, lengthp):
            poss = best_patches[val]
            self.show_patches[poss[0]:poss[2], poss[1]:poss[3]] = np.random.rand(1)[0]

            for freqvals in range(0, len(self.freqs)):
                covalsflux = flux[freqvals, poss[0]:poss[2], poss[1]:poss[3]]
                contamsflux = contamins[freqvals, poss[0]:poss[2], poss[1]:poss[3]]
                nonzeropos = np.where(np.ravel(covalsflux) != 0.)
                allsigflux[freqvals, val] = np.median(np.ravel(covalsflux)[nonzeropos])
                allcontamins[freqvals, val] = np.median(np.ravel(contamsflux)[nonzeropos])

            # fitting model
            yval = allsigflux[:, val]
            nus = allcontamins[:, val]
            kwsdict = {'noise':nus}
            resultpre = Minimizer(self.resid_one_beta, params, fcn_args=(self.freqs, yval), \
                        fcn_kws=kwsdict)
            result = resultpre.minimize()
            val_dic = result.params

            self.show_tau[faceval, poss[0]:poss[2], poss[1]:poss[3]] = np.array(val_dic["tau"])
            self.show_beta[faceval, poss[0]:poss[2], poss[1]:poss[3]] = np.array(val_dic["beta"])
            self.show_temp[faceval, poss[0]:poss[2], poss[1]:poss[3]] = np.array(val_dic["temp"])

            #calculate redchi
            lenp = int(poss[2] - poss[0])
            simval = self.dust_model(self.freqs, val_dic["tau"], val_dic["temp"], val_dic["beta"])
            resid = flux[:, poss[0]:poss[2], poss[1]:poss[3]] - \
                    simval.reshape(len(self.freqs), 1, 1)
            _, wtres, _ = wt_decompose(resid, jscale=self.redwn)
            for wnum in range(0, self.redwn-1):
                self.redchi[wnum, poss[0]: poss[2], poss[1]:poss[3]] = \
                                    self.chicov(wtres[:, :, wnum], rnus[wnum, :, :], lenp)

        return self.show_tau, self.show_beta, self.show_temp, self.show_patches, self.redchi

    def run_fit(self):
        """ run the fitting routine of data """

        fluxfull = np.zeros((len(self.freqs), 12*self.size**2))
        maskfull = np.zeros((len(self.freqs)-1, 12*self.size**2))
        contamfull = np.zeros((len(self.freqs), 12*self.size**2))

        for fvals in range(0, len(self.freqs)):
            fluxfull[fvals, :] = hp.fitsfunc.read_map('totalDustFlux_CMBpsCIBnoise_noCC.fits', \
                                    nest=False, field=fvals)
            contamfull[fvals, :] = hp.fitsfunc.read_map('totalNoiseCIB.fits', nest=False, \
                                        field=fvals)
            if fvals < 3:
                maskfull[fvals, :] = hp.fitsfunc.read_map('totalMask.fits', nest=False, field=fvals)

        #get 12 faces
        fluxcube = np.array([pym.get_all_faces(fluxfull[x, :], nested=False) \
                    for x in range(len(self.freqs))])
        contaminscube = np.array([pym.get_all_faces(contamfull[x, :], nested=False) \
                        for x in range(len(self.freqs))])
        maskcube = np.array([pym.get_all_faces(maskfull[x, :], nested=False) \
                    for x in range(len(self.freqs)-1)])

        for faceval in range(0, 12):
            print 'Face %r out of %r' %(faceval, 12)

            fluxfilt = np.zeros((len(self.freqs), self.size, self.size))

            #thresholding on 2048X2048 takes too long - further divide the face into quarters
            for quart in range(0, 4):
                print 'Quarter number %r' %(quart+1)

                inda = [0, self.size/2, 0, self.size/2]
                indb = [0, 0, self.size/2, self.size/2]
                jump = self.size/2

                flu = fluxcube[:, faceval, inda[quart]:inda[quart] + jump, \
                        indb[quart]:indb[quart] + jump]
                con = contaminscube[:, faceval, inda[quart]:inda[quart] + jump, \
                        indb[quart]:indb[quart] + jump]
                mas = maskcube[:, faceval, inda[quart]:inda[quart] + jump, \
                        indb[quart]:indb[quart] + jump]

                #for the chi squared
                _, wtn, _ = wt_decompose(con - self.ciboff.reshape(len(self.freqs), 1, 1), \
                                jscale=self.redwn)
                rnus = np.array([(1./self.size**2)*np.dot(wtn[:, :, wnum], wtn[:, :, wnum].T) \
                        for wnum in range(self.redwn)])

                print 'Filtering'
                fluxback, _ = mfil.multi_filtering(flu, con, jval=2, bins=8, kmad=2.4)

                #remove CIB offset for IRIS
                flu[3, :, :] = fluxback[3, :, :] - self.ciboff[3]

                #apply mask and remove CIB offset for HFI maps
                for val in range(0, 3):
                    flu[val, :, :] = (fluxback[val, :, :] - self.ciboff[val]) * mas[val, :, :]

                fluxfilt[:, inda[quart]:inda[quart] + jump, indb[quart]:indb[quart] + jump] = flu

            patches = []

            mask = maskcube[:, faceval, :, :]
            contamins = contaminscube[:, faceval, :, :]

            # split the data into 128X128 areas to calculate the reduced chi squared
            firstgo = QuadTree128(fluxfilt, patches, [0, 0, self.size, self.size])
            best_patches = firstgo.split_four()

            self.show_tau, self.show_beta, self.show_temp, self.show_patches, self.redchi = \
                    self.loop_over_patches(best_patches, fluxfilt, contamins, faceval, rnus)

            print 'Finished calcualting chi squared'

            #mask so quadtree knows not to divide into patches smaller than we have signal
            totmask = mask[0, :, :] *  mask[1, :, :] *  mask[2, :, :]

            new_img = QuadTreeChi(self.redchi * totmask, patches, [0, 0, self.size, self.size])
            best_patches = new_img.split_four()

            self.show_tau, self.show_beta, self.show_temp, self.show_patches, self.redchi = \
                        self.loop_over_patches(best_patches, fluxfilt, contamins, faceval, rnus)

        finaltemp = pym.put_all_faces(self.show_temp, nested=False)
        finaltau = pym.put_all_faces(self.show_tau, nested=False)
        finalbeta = pym.put_all_faces(self.show_beta, nested=False)
        finalpatches = pym.put_all_faces(self.show_patches, nested=False)

        #write out results
        hp.fitsfunc.write_map('quadtreeDustTemp.fits', finaltemp, nest=False)
        hp.fitsfunc.write_map('quadtreeDustTau.fits', finaltau, nest=False)
        hp.fitsfunc.write_map('quadtreeDustBeta.fits', finalbeta, nest=False)
        hp.fitsfunc.write_map('quadtreeDustPatches.fits', finalpatches, nest=False)

        return None

RUNNING = HfiModelFitting()
RUNNING.run_fit()
