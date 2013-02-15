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
import sklearn.mixture

import scipy.ndimage

sys.path.append("./extern/py3DSeedEditor/")
import py3DSeedEditor
# version comparison
from pkg_resources import parse_version

if parse_version(sklearn.__version__) > parse_version('0.10'):
    #new versions
    defaultmodelparams =  {'type':'gmmsame','params':{'covariance_type':'full'}}
else:
    defaultmodelparams =  {'type':'gmmsame','params':{'cvtype':'full'}}

#defaultmodelparams =  {'type':'gmmsame','params':{'covariance_type':'full'}}

class Model:
    """ Model for image intensity. Last dimension represent feature vector. 
    m = Model()
    m.train(cla, clb)
    X = numpy.random.random([2,3,4])
    # we have data 2x3 with fature vector with 4 fatures
    m.likelihood(X,0)
    """
    def __init__ (self, nObjects=2,  modelparams = defaultmodelparams):

        self.mdl =  {}
        self.modelparams = modelparams
        pass

    def train(self, clx, cl ):
        """ Train clas number cl with data clx """

        if self.modelparams['type'] == 'gmmsame':
            gmmparams = self.modelparams['params']
            #mdl1 = sklearn.mixture.GMM(covariance_type='full')
            self.mdl[cl] = sklearn.mixture.GMM(**gmmparams)
            if len(clx.shape) == 1:
                # je to jen jednorozměrný vektor, tak je potřeba to převést na 2d matici
                clx = clx.reshape(-1,1)
            self.mdl[cl].fit(clx)
        else:
            raise NameError("Unknown model type")

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

        px = self.mdl[cl].score(xr)

#todo ošetřit více dimenzionální fv
        px = px.reshape(sha)
        return px

         


class ImageGraphCut:
    """
    Interactive Graph Cut

    ImageGraphCut(data, zoom, modelparams)
    scale

    Example:

    igc = ImageGraphCut(data)
    igc.interactivity()
    igc.make_gc()
    igc.show_segmentation()
    logger.debug(igc.segmentation.shape)
    """
    def __init__(self, img, zoom = 1, modelparams = defaultmodelparams, gcparams = {'pairwiseAlpha':10}):
        self.img = img
        self.tdata = {}
        self.segmentation = []
        self.imgshape = img.shape
        if np.isscalar(zoom):
            zoom = [zoom]*3
        self.zoom = np.array(zoom)
        self.modelparams = modelparams
        self.gcparams = gcparams

        self.img_input_resize()
        self.seeds = np.zeros(self.img.shape, dtype=np.int8)
        self.editor_mouse_button_map = {1:2,2:3, 3:1}
        self.crinfo = None


    def _seed_input_resize(self, seeds_orig_shape):
        """
        function sets seeds. They are resized to proper shape
        """
        #self.seeds  = scipy.ndimage.zoom(seeds_orig_shape.astype(np.float16) , self.zoom, prefilter=False, mode= 'nearest', order = 1)
        self.seeeds = scipy.ndimage.zoom(seeds_orig_shape , self.zoom, prefilter=False, mode= 'nearest', order = 1)
        self.seeds = self.seeds.astype(np.int8)



        
    def img_input_resize(self):
        #pdb.set_trace();

        self.img = scipy.ndimage.zoom(self.img, self.zoom, prefilter=False, mode= 'nearest', order = 1)
        self.img_orig_shape = False

    def img_output_resize(self):
        if not  self.img_orig_shape:
            #self.working_segmentation = self.segmentation
            self.segmentation = scipy.ndimage.zoom(self.segmentation, 1/self.zoom, order = 1)
        self.img_orig_shape = True

    def get_orig_shape_segmentation(self):
        if not  self.img_orig_shape:
            #self.working_segmentation = self.segmentation
            return  scipy.ndimage.zoom(self.segmentation, 1/self.zoom, order = 1)

    def get_orig_shape_cropped_seeds(self, margin = [1,1,1]):
        """
        Function return seeds in original scale. Image is cropped with 
        respect of segmentation.
        """
        if not  self.img_orig_shape:

            crinfo = self._crinfo_from_specific_data(self.segmentation, margin)
            crseeds = self._crop(self.segmentation, crinfo)
            orig_scale_seeds = scipy.ndimage.zoom(crseeds, 1/self.zoom, order = 1)
            return orig_scale_seeds

        else:
            Exception("Image was not resized")

        
    def get_orig_scale_cropped_segmentation(self, margin=[1,1,1]):
        """
        Make automatic segmentation nonzero crop. Margin adds some space 
        around nonzero values
        """
        # TODO crinfo má původ velikosti a je jen přenásobeno. Patrně tam 
        # budou necelá čísla a nebude to úplně sedět. Asi by bylo lepší to 
        # nejprve přezvětšit a pak až oříznout
        if not  self.img_orig_shape:

            crinfo = self._crinfo_from_specific_data(self.segmentation, margin)
            crdata = self._crop(self.segmentation, crinfo)
            #self.working_segmentation = self.segmentation
            orig_scale_data = scipy.ndimage.zoom(crdata, 1/self.zoom, order = 1)
            
            #import pdb; pdb.set_trace()
            orig_scale_crinfo = [
                    [int(round(crinfo[0][0]/self.zoom[0])),
                     int(round(crinfo[0][0]/self.zoom[0]))+orig_scale_data.shape[0]],
                    [int(round(crinfo[1][0]/self.zoom[1])),
                     int(round(crinfo[1][0]/self.zoom[1]))+orig_scale_data.shape[1]],
                    [int(round(crinfo[2][0]/self.zoom[2])),
                     int(round(crinfo[2][0]/self.zoom[2]))+orig_scale_data.shape[2]],
                    ]
            #print crinfo
            #print orig_scale_crinfo
            return orig_scale_data, orig_scale_crinfo
        else:
            Exception("Image was not resized")

    def _crop(self, data, crinfo):
        """
        Crop data with crinfo
        """
        data = data[crinfo[0][0]:crinfo[0][1], crinfo[1][0]:crinfo[1][1], crinfo[2][0]:crinfo[2][1]]
        return data


    def _crinfo_from_specific_data (self, data, margin):
# hledáme automatický ořez, nonzero dá indexy
        nzi = np.nonzero(data)

        x1 = np.min(nzi[0]) - margin[0]
        x2 = np.max(nzi[0]) + margin[0] + 1
        y1 = np.min(nzi[1]) - margin[0]
        y2 = np.max(nzi[1]) + margin[0] + 1
        z1 = np.min(nzi[2]) - margin[0]
        z2 = np.max(nzi[2]) + margin[0] + 1 

# ošetření mezí polí
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if z1 < 0:
            z1 = 0

        if x2 > data.shape[0]:
            x2 = data.shape[0]-1
        if y2 > data.shape[1]:
            y2 = data.shape[1]-1
        if z2 > data.shape[2]:
            z2 = data.shape[2]-1

# ořez
        crinfo = [[x1, x2],[y1,y2],[z1,z2]]
        #dataout = self._crop(data,crinfo)
        #dataout = data[x1:x2, y1:y2, z1:z2]
        return crinfo


    def interactivity(self):
        """
        Interactive seed setting with 3d seed editor
        """

        pyed = py3DSeedEditor.py3DSeedEditor(self.img, 
                mouse_button_map = self.editor_mouse_button_map )
        pyed.show()

        #scipy.io.savemat(args.outputfile,{'data':output})
        #pyed.get_seed_val(1)


        # control list of non zero values in seeds
        #nzero_seeds_prev = ([],[])
        nzero_seeds = pyed.seeds.nonzero()

        pocitadlo = 0
        opakovat = True

        while opakovat:#(nzero_seeds_prev != nzero_seeds).all():
            pocitadlo = pocitadlo + 1

            self.voxels1 = pyed.get_seed_val(1)
            self.voxels2 = pyed.get_seed_val(2)
            self.seeds = pyed.seeds
            self.make_gc()

# new pyeditor is created, seeds must be setted
            pyed = py3DSeedEditor.py3DSeedEditor(self.img, seeds = self.seeds, 
                    contour = self.segmentation,
                    mouse_button_map = self.editor_mouse_button_map 
                    )

            #pyed.seeds = self.seeds

            pyed.show()

            opakovat = not np.array_equal(pyed.seeds.nonzero() , nzero_seeds)


            #opakovat =True #not all(opakovat)
            nzero_seeds = pyed.seeds.nonzero()



# iterative seed selection
        

    def noninteractivity(self, seeds):
        """
        Function for noninteractive seed setting
        """
        self.seeds = seeds
        self.voxels1 = self.img[seeds==1]
        self.voxels2 = self.img[seeds==2]
        self.make_gc()
        self.img_output_resize()

    def make_gc(self):
        #pdb.set_trace();

        
        res_segm = self.set_data(self.img, self.voxels1, self.voxels2, seeds = self.seeds)

        self.segmentation = res_segm

    def show_segmentation(self, img = None):

        if img == None:
            img = self.segmentation

        pyed = py3DSeedEditor.py3DSeedEditor(img)
        pyed.show()

    def set_hard_hard_constraints(self, tdata1, tdata2, seeds):
        tdata1[seeds==2] = np.max(tdata1) + 1
        tdata2[seeds==1] = np.max(tdata2) + 1
        tdata1[seeds==1] = 0
        tdata2[seeds==2] = 0

        return tdata1, tdata2



        

    def set_data(self, data, voxels1, voxels2, seeds = False, hard_constraints = True):
        """
        Setting of data.
        You need set seeds if you want use hard_constraints.
        
        """
        mdl = Model ( modelparams = self.modelparams )
        mdl.train(voxels1, 1)
        mdl.train(voxels2, 2)
        #pdb.set_trace();
        #tdata = {}
# as we convert to int, we need to multipy to get sensible values

# There is a need to have small vaues for good fit
# R(obj) = -ln( Pr (Ip | O) )
# R(bck) = -ln( Pr (Ip | B) )
# Boykov2001a 
# ln is computed in likelihood 
        tdata1 = (-(mdl.likelihood(data, 1))) * 10
        tdata2 = (-(mdl.likelihood(data, 2))) * 10

        #pyed = py3DSeedEditor.py3DSeedEditor(tdata1)
        #pyed = py3DSeedEditor.py3DSeedEditor(seeds)
        #pyed.show()
        #pdb.set_trace();

        if hard_constraints: 
            #pdb.set_trace();
            if (type(seeds)=='bool'):
                raise Exception ('Seeds variable  not set','There is need set seed if you want use hard constraints')
            tdata1, tdata2 = self.set_hard_hard_constraints(tdata1, tdata2, seeds)
            


        unariesalt = (1 * np.dstack([tdata1.reshape(-1,1), tdata2.reshape(-1,1)]).copy("C")).astype(np.int32)

# create potts pairwise
        #pairwiseAlpha = -10
        pairwise = -self.gcparams['pairwiseAlpha'] * np.eye(2, dtype=np.int32)
# use the gerneral graph algorithm
# first, we construct the grid graph
        inds = np.arange(data.size).reshape(data.shape)
        edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
        edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
        edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
        edges = np.vstack([edgx, edgy, edgz]).astype(np.int32)

# edges - seznam indexu hran, kteres spolu sousedi

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
                                                                                                        
class Tests(unittest.TestCase):
    def setUp(self):
        pass

    def test_segmentation(self):
        data_shp = [16,16,16]
        data = generate_data(data_shp)
        seeds = np.zeros(data_shp)
# setting background seeds
        seeds[:,0,0] = 1
        seeds[6,8:-5,2] = 2
    #x[4:-4, 6:-2, 1:-6] = -1

        igc = ImageGraphCut(data)
        #igc.interactivity()
# instead of interacitivity just set seeeds
        igc.noninteractivity(seeds)

# instead of showing just test results
        #igc.show_segmentation()
        segmentation = igc.segmentation
        # Testin some pixels for result
        self.assertTrue(segmentation[0, 0, -1] == 0)
        self.assertTrue(segmentation[7, 9, 3] == 1)
        self.assertTrue(np.sum(segmentation) > 10)
        #pdb.set_trace()
        #self.assertTrue(True)


        #logger.debug(igc.segmentation.shape)




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

    igc = ImageGraphCut(data)#, zoom=0.8)
    igc.interactivity()
    #data, crinfo = igc.get_orig_shape_cropped_segmentation()
    #igc.show_segmentation(data)
    logger.debug(igc.segmentation.shape)

   # pyed = py3DSeedEditor.py3DSeedEditor(data)
   # output = pyed.show()

   # #scipy.io.savemat(args.outputfile,{'data':output})
   # #pyed.get_seed_val(1)

   # voxels1 = pyed.get_seed_val(0)
   # voxels2 = pyed.get_seed_val(1)

   # #pdb.set_trace();
   # logger.debug(len(voxels1))
   # logger.debug(len(voxels2))

   # igc = ImageGraphCut(data)
   # 
   # res_segm = igc.set_data(data, voxels1, voxels2, seeds = pyed.seeds)

   # pyed = py3DSeedEditor.py3DSeedEditor(res_segm)
   # output = pyed.show()
   # 
   # 
    
    
    
    
    
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
