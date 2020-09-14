"""
Created on Wed 4th Jan 2017

@author: mirfan
"""
import numpy as np

class QuadTreeChi(object):
    """ quadtree --> splits image until signal is gaussian """
    def __init__(self, motherim, patches, rect):
        self.motherim = motherim
        self.children = [None, None, None, None]
        self.patches = patches
        self.rect = rect

    def too_empty(self, tocut):
        """ see if splitting this into 4 will leave you with an empty quater """
        lenp = np.shape(tocut)[0]
        sig1 = np.sum(tocut[0:lenp/2, 0: lenp/2])
        sig2 = np.sum(tocut[lenp/2:lenp, 0: lenp/2])
        sig3 = np.sum(tocut[0:lenp/2, lenp/2: lenp])
        sig4 = np.sum(tocut[lenp/2:lenp, lenp/2: lenp])

        if sig1 * sig2 * sig3 * sig4 == 0:
            fine = False
        else:
            fine = True

        return fine

    def which_wave(self, tocut):
        """ criterion for splitting """
        good_to_go = self.too_empty(tocut)

        signal = np.array(tocut).flatten()
        biginds = np.where(np.array(signal) > 2.0)

        criterion = len(signal) > 128.**2 or (len(signal[biginds])/float(len(signal)) > \
                        0.10 and good_to_go == True)

        return criterion

    def test_chi(self, rects):
        """ See if the chi squared in  """
        lenp = rects[2] - rects[0]
        if lenp > 64:
            cut_up = self.motherim[2, rects[0]: rects[2], rects[1]: rects[3]]
        elif lenp == 64:
            cut_up = self.motherim[1, rects[0]: rects[2], rects[1]: rects[3]]
        else:
            cut_up = self.motherim[0, rects[0]: rects[2], rects[1]: rects[3]]

        criterion = self.which_wave(cut_up)

        if criterion:
            ans = 0
        else:
            ans = 1

        return ans

    def getinstance(self, motherim, patches, rect):
        """ snapshot of parameter values """
        return QuadTreeChi(self.motherim, self.patches, rect)

    def split_four(self):
        """ splits a 2D symmetric image into four """
        corna, cornb, upcoa, upcob = self.rect
        hval = (upcoa - corna)/2
        rects = []
        rects.append((corna, cornb, corna + hval, cornb + hval))
        rects.append((corna, cornb + hval, corna + hval, upcob))
        rects.append((corna + hval, cornb + hval, upcoa, upcob))
        rects.append((corna + hval, cornb, upcoa, cornb + hval))

        for enn in range(len(rects)):

            gtest = self.test_chi(rects[enn])

            if gtest == 0 and hval > 8:
                self.children[enn] = self.getinstance(self.motherim, self.patches, rects[enn])
                self.children[enn].split_four()
            else:
                self.patches.append(rects[enn])

        return self.patches
