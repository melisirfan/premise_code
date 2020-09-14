"""
Created on Wed 28th Aug 2017

@author: mirfan
"""

class QuadTree128(object):
    """ quadtree --> splits image until signal is gaussian """
    def __init__(self, motherim, patches, rect):
        self.motherim = motherim
        self.children = [None, None, None, None]
        self.patches = patches
        self.rect = rect

    def test_length(self, rects):
        """ See if the chi squared in """
        lenp = rects[2] - rects[0]

        if lenp > 128:
            ans = 0
        else:
            ans = 1

        return ans

    def getinstance(self, motherim, patches, rect):
        """ snapshot of parameter values """
        return QuadTree128(self.motherim, self.patches, rect)

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

            gtest = self.test_length(rects[enn])

            if gtest == 0 and hval > 127:
                self.children[enn] = self.getinstance(self.motherim, self.patches, rects[enn])
                self.children[enn].split_four()
            else:
                self.patches.append(rects[enn])

        return self.patches
