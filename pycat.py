# -*- coding: utf-8 -*-

import unittest
import sys
import pdb
#  pdb.set_trace();


import logging
logger = logging.getLogger(__name__)

import argparse

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from pygco import cut_simple, cut_from_graph
import sklearn
from sklearn import mixture


sys.path.append("./extern/py3DSeedEditor/")
import py3DSeedEditor


class Model:
    """ Model for image intensity. Last dimension represent feature vector. 
    m = Model()
    m.train(cla, clb)
    X = numpy.random.random([2,3,4])
    # we have data 2x3 with fature vector with 4 fatures
    m.likelihood(X,0)
    """
    def __init__ (self, nObjects=2):
        self.mdl =  {}
        pass

    def train(self, clx, cl):
        """ Train clas number cl with data clx """
        #mdl1 = sklearn.mixture.GMM(covariance_type='full')
        self.mdl[cl] = sklearn.mixture.GMM(cvtype='full')
	if len(clx.shape) == 1:
            # je to jen jednorozměrný vektor, tak je potřeba to převést na 2d matici
            clx = clx.reshape(-1,1)
        self.mdl[cl].fit(clx)
        #pdb.set_trace();

    def likelihood(self, x, cl, onedimfv = True):
        """
        X = numpy.random.random([2,3,4])
        # we have data 2x3 with fature vector with 4 fatures
        m.likelihood(X,0)
        """

        sha = x.shape
        if onedimfv:
            xr = x.reshape(-1, 1)
        else:
            xr = x.reshape(-1, sha[-1])

        pdb.set_trace();
        px = self.mdl[cl].score(xr)

#todo ošetřit více dimenzionální fv
        px = px.reshape(sha)
        return px

         


class ImageGraphCut:
    def __init__(self):
        self.tdata = {}


    def set_hard_seeds(self, tdata1, tdata2, seeds):
        tdata1[seeds==1] = np.max(tdata1) + 1
        tdata2[seeds==2] = np.max(tdata2) + 1
        tdata1[seeds==2] = 0
        tdata2[seeds==1] = 0


        

    def set_data(self, data, voxels1, voxels2):
        mdl = Model ()
        mdl.train(voxels1, 1)
        mdl.train(voxels2, 2)
        #pdb.set_trace();
        #tdata = {}
        tdata1 = mdl.likelihood(data, 1)
        tdata2 = mdl.likelihood(data, 2)

# as we convert to int, we need to multipy to get sensible values

        unariesalt = (10 * np.dstack([tdata1.reshape(-1,1), tdata2.reshape(-1,1)]).copy("C")).astype(np.int32)

# create potts pairwise
        pairwise = -10 * np.eye(2, dtype=np.int32)
# use the gerneral graph algorithm
# first, we construct the grid graph
        inds = np.arange(data.size).reshape(data.shape)
        edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
        edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
        edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
        edges = np.vstack([edgx, edgy, edgz]).astype(np.int32)

# we flatten the unaries
        #result_graph = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
        result_graph = cut_from_graph(edges, unariesalt.reshape(-1,2), pairwise)

        
        result_labeling = result_graph.reshape(data.shape)

        return result_labeling

def generate_data(shp=[16,16,16]):
    """ Generating random data with cubic object inside"""

    x = np.ones(shp)
# inserting box
    x[4:-4, 6:-2, 1:-6] = -1
    x_noisy = x + np.random.normal(0, 0.6, size=x.shape)
    return x_noisy

def example_3d():

    x = np.ones((10, 10, 10))
    x[:, 5:, :] = -1
    x_noisy = x + np.random.normal(0, 0.8, size=x.shape)
    x_thresh = x_noisy > .0

# create unaries
    unaries = x_noisy
# as we convert to int, we need to multipy to get sensible values
    unariesalt = (10 * np.dstack([unaries.reshape(-1,1), -unaries.reshape(-1,1)]).copy("C")).astype(np.int32)
    # unariesa = (10 * unaries).astype(np.int32)
    # unariesb = (-10* unaries).astype(np.int32)
    # unariescol = np.concatenate([unariesa.reshape(-1,1), unariesb.reshape(-1,1)], axis=1).astype(np.int32)
# create potts pairwise
    pairwise = -10 * np.eye(2, dtype=np.int32)
# use the gerneral graph algorithm
# first, we construct the grid graph
    inds = np.arange(x.size).reshape(x.shape)
    edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
    edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
    edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
    edges = np.vstack([edgx, edgy, edgz]).astype(np.int32)

    pdb.set_trace()
# we flatten the unaries
    #result_graph = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
    result_graph = cut_from_graph(edges, unariesalt.reshape(-1,2), pairwise)

    
    result_labeling = result_graph.reshape(x.shape)

#show results for 3th slice
    plt.subplot(311, title="original")
    plt.imshow(x[:,:,3], interpolation='nearest')
    plt.subplot(312, title="noisy version")
    plt.imshow(x_noisy[:,:,3], interpolation='nearest')
    plt.subplot(313, title="cut_from_graph")

    plt.imshow(result_labeling[:,:,3], interpolation='nearest')
    plt.show()

def example_binary():
# generate trivial data
    x = np.ones((10, 10))
    x[:, 5:] = -1
    x_noisy = x + np.random.normal(0, 0.8, size=x.shape)
    x_thresh = x_noisy > .0

# create unaries
    unaries = x_noisy
# as we convert to int, we need to multipy to get sensible values
    unaries = (10 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
# create potts pairwise
    pairwise = -10 * np.eye(2, dtype=np.int32)

# do simple cut
    result = cut_simple(unaries, pairwise)

# use the gerneral graph algorithm
# first, we construct the grid graph
    inds = np.arange(x.size).reshape(x.shape)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert]).astype(np.int32)

# we flatten the unaries
    result_graph = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
    
    pdb.set_trace()


# plot results
    plt.subplot(231, title="original")
    plt.imshow(x, interpolation='nearest')
    plt.subplot(232, title="noisy version")
    plt.imshow(x_noisy, interpolation='nearest')
    plt.subplot(233, title="rounded to integers")
    plt.imshow(unaries[:, :, 0], interpolation='nearest')
    plt.subplot(234, title="thresholding result")
    plt.imshow(x_thresh, interpolation='nearest')
    plt.subplot(235, title="cut_simple")
    plt.imshow(result, interpolation='nearest')
    plt.subplot(236, title="cut_from_graph")
    plt.imshow(result_graph.reshape(x.shape), interpolation='nearest')

    plt.show()
                                                                                                        

# --------------------------main------------------------------
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
# při vývoji si necháme vypisovat všechny hlášky
    #logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
#   output configureation
    #logging.basicConfig(format='%(asctime)s %(message)s')
    logging.basicConfig(format='%(message)s')

    formatter = logging.Formatter("%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)

    logger.addHandler(ch)


    # input parser
    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('-f','--filename',  
            #default = '../jatra/main/step.mat',
            default = '3d',
            help='*.mat file with variables "data", "segmentation" and "threshod"')
    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-e3', '--example3d', action='store_true',
            help='run with 3D example data')
    parser.add_argument('-t', '--tests', action='store_true', 
            help='run unittest')
    parser.add_argument('-o', '--outputfile', type=str,
        default='output.mat',help='output file name')
    args = parser.parse_args()


    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.tests:
        # hack for use argparse and unittest in one module
        sys.argv[1:]=[]
        unittest.main()

    if args.example3d:
        data = generate_data()
    elif args.filename == 'lena':
        from scipy import misc
        data = misc.lena()
    elif args.filename == '3d':
        from scipy import misc
        data = generate_data()
    else:
    #   load all 
        mat = scipy.io.loadmat(args.filename)
        logger.debug( mat.keys())

        # load specific variable
        dataraw = scipy.io.loadmat(args.filename, variable_names=['data'])
        data = dataraw['data']

        #logger.debug(matthreshold['threshold'][0][0])


        # zastavení chodu programu pro potřeby debugu, 
        # ovládá se klávesou's','c',... 
        # zakomentovat
        #pdb.set_trace();

        # zde by byl prostor pro ruční (interaktivní) zvolení prahu z klávesnice 
        #tě ebo jinak

    pyed = py3DSeedEditor.py3DSeedEditor(data)
    output = pyed.show()

    scipy.io.savemat(args.outputfile,{'data':output})
    pyed.get_seed_val(1)

    voxels1 = pyed.get_seed_val(0)
    voxels2 = pyed.get_seed_val(1)

    #pdb.set_trace();
    logger.debug(len(voxels1))
    logger.debug(len(voxels2))

    igc = ImageGraphCut()
    res_segm = igc.set_data(data, voxels1, voxels2)

    pyed = py3DSeedEditor.py3DSeedEditor(res_segm)
    output = pyed.show()
    # model test
#    mdl = Model()
#    mdl.train(voxels1, voxels2)
#    #pdb.set_trace();
#    ndata = mdl.likelihood(data, 0)
#    pyed = py3DSeedEditor.py3DSeedEditor(ndata)
#    output = pyed.show()
#    
#    



#example_3d()

#example_binary()
