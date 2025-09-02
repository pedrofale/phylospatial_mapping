import numpy as np
import matplotlib.pyplot as plt
import warnings
import cassiopeia as cas
import squidpy as sq
from scipy.spatial.distance import pdist


## Some utility functions for tree plotting
def plot_selection(tree, adata, node, color):

    subtree = tree.leaves_in_subtree(node)

    adata.obs['Selection'] = np.nan
    adata.obs.loc[tree.leaves, 'Selection'] = "False"
    adata.obs.loc[subtree, 'Selection'] = "True"

    with warnings.catch_warnings():

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
        cas.pl.plot_matplotlib(
            tree,
            add_root=True,
            indel_priors=None,
            clade_colors={node: color},
            ax=ax1
        )

        adata.uns['Selection_colors'] = ['#838383', color]
        adata.obs['library_id'] = 'puck1'

        sq.pl.spatial_scatter(adata, size=1, library_id='puck1', shape=None, na_color='lightgray', ax=ax2, figsize=(5,5), legend_loc=None)
        sq.pl.spatial_scatter(adata, color=['Selection'],
                            groups = ['True'],
                            library_id='puck1',
                            size=20, ax=ax2, shape=None,
                                scalebar_dx=1.0,
                            scalebar_units='um',
                            scalebar_kwargs={"scale_loc": "bottom", "location": "upper right"})

        sq.pl.spatial_scatter(adata, color=['Selection'], library_id='puck1',
                            groups = ['False'],
                            size=5, ax=ax2, shape=None)


        fig.tight_layout()
        plt.show()

def compute_spatial_distance_matrix(spatial_coordinates):
    return pdist(spatial_coordinates, metric='euclidean')