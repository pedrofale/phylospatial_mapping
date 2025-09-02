import numpy as np
import ot
from ot.gromov import gromov_wasserstein, fused_gromov_wasserstein
from scipy.stats import pearsonr

def fgw(single_cell_data, spatial_data, tree_distance_matrix, space_distance_matrix, alpha=0.5):
    C1 = tree_distance_matrix   
    C2 = space_distance_matrix
    M = ot.dist(single_cell_data, spatial_data)
    w1 = ot.unif(C1.shape[0])
    w2 = ot.unif(C2.shape[0])

    # Computing FGW and GW
    T_fgw = fused_gromov_wasserstein(M, C1, C2, w1, w2, loss_fun='square_loss', alpha=alpha)
    return T_fgw

def gw(tree_distance_matrix, space_distance_matrix):
    C1 = tree_distance_matrix
    C2 = space_distance_matrix
    w1 = ot.unif(C1.shape[0])
    w2 = ot.unif(C2.shape[0])
    T_gw = gromov_wasserstein(C1, C2, w1, w2, loss_fun='square_loss')
    return T_gw

def optimal_transport(single_cell_data, spatial_data):
    M = ot.dist(single_cell_data, spatial_data)
    T_ot = ot.emd([], [], M)
    return T_ot

def pairwise_correlations(single_cell_data, spatial_data):
    n_cells = single_cell_data.shape[0]
    n_spots = spatial_data.shape[0]

    # Preallocate correlation matrix
    pairwise_correlations = np.zeros((n_cells, n_spots))

    for i in range(n_cells):
        for j in range(n_spots):
            # Compute Pearson correlation between cell i and spot j across genes
            corr, _ = pearsonr(single_cell_data[i], spatial_data[j])
            pairwise_correlations[i, j] = corr

    return pairwise_correlations