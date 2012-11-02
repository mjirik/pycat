import matplotlib.pyplot as plt
import numpy as np
from pygco import cut_simple, cut_from_graph

import pdb

def generate_data(shp=[16,16,16]):
    """ Generating data """

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
            default = 'lena',
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

    pyed = py3DSeedEditor(data)
    output = pyed.show()

    scipy.io.savemat(args.outputfile,{'data':output})
    pyed.get_seed_val(1)


example_3d()

#example_binary()
