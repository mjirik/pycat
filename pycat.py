import matplotlib.pyplot as plt
import numpy as np
from pygco import cut_simple, cut_from_graph

import pdb

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
                                                                                                        


example_3d()

#example_binary()
