import numpy as np
import matplotlib.pyplot as plt
import warnings
import cassiopeia as cas
import squidpy as sq
import anndata
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import pycea as py

# Try relative import first (for package imports), fall back to absolute import (for direct imports)
try:
    from . import expression_simulator, space_simulator, visium_simulator
except ImportError:
    # For direct imports (e.g., via importlib), import from same directory
    import sys
    import os
    
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    import expression_simulator
    import space_simulator
    import visium_simulator


## Some utility functions for tree plotting
def plot_selection(tree, adata, nodes, colors, orient=90):
    adata.obs['clade'] = np.nan
    for node in nodes:
        subtree = tree.leaves_in_subtree(node)
        adata.obs.loc[subtree, 'clade'] = f'clade{node}'

    with warnings.catch_warnings():

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
        cas.pl.plot_matplotlib(
            tree,
            add_root=False,
            indel_priors=None,
            clade_colors=dict(zip(nodes, colors)),
            extend_branches=False,
            orient=orient,
            ax=ax1
        )

        adata.uns['clade_colors'] = colors
        adata.obs['library_id'] = 'puck1'

        # sq.pl.spatial_scatter(adata, size=1, library_id='puck1', shape=None, na_color='lightgray', ax=ax2, figsize=(5,5), legend_loc=None)
        sq.pl.spatial_scatter(adata, color=['clade'],
                            library_id='puck1',
                            size=20, ax=ax2, shape=None,
                                scalebar_dx=1.0,
                            scalebar_units='um',
                            scalebar_kwargs={"scale_loc": "bottom", "location": "upper right"})


        fig.tight_layout()
        plt.show()

def compute_spatial_distance_matrix(spatial_coordinates):
    return pdist(spatial_coordinates, metric='euclidean')

def simulate_spatial_locations(tree_simulator, mode):
    # TODO: use rejection sampling to ensure the data matches the mode
    n_samples = 100

    modes = {
        'main_clades': {'2': .1, '3': .1},
        'subclades': {'4': .1, '5': .1, '6': .1, '7': .1},
        'main_and_subclades': {'4': .1, '5': .1, '3': .1},
        'unstructured': {'1': 1.},
    }

    simulated_tree = tree_simulator.simulate_tree()

    clade_leaves = {}
    for clade, scale in modes[mode].items():
        clade_leaves[clade] = simulated_tree.leaves_in_subtree(clade)

    for clade in modes[mode]:
        branch_lengths = {}
        for p, c in simulated_tree.breadth_first_traverse_edges(clade):
            branch_lengths = simulated_tree.get_branch_length(p,c) * modes[mode][clade]
            simulated_tree.set_branch_length(p, c, branch_lengths)

    spatial_simulator = space_simulator.BrownianSpatialDataSimulator(2, 1.)
    spatial_simulator.overlay_data(simulated_tree, clades=list(clade_leaves.values()), rates=list(modes[mode].values()), brownian_motion=True, lambda_brownian=1.)

    # Reset the tree branches
    for clade in modes[mode]:
        branch_lengths = {}
        for p, c in simulated_tree.breadth_first_traverse_edges(clade):
            branch_lengths = simulated_tree.get_branch_length(p,c) * 1./modes[mode][clade]
            simulated_tree.set_branch_length(p, c, branch_lengths)

    simulated_adata = anndata.AnnData(obs=pd.DataFrame(index=simulated_tree.leaves))
    simulated_adata.obs['library_id'] = 'simulated_puck'
    simulated_adata.obsm['spatial'] = simulated_tree.cell_meta[['spatial_0', 'spatial_1']].to_numpy()        

    return simulated_tree


def simulate_expression(simulated_tree, expression_mode, n_traits, n_genes, obs_model='normal', sigma=0.1, lib_size=10000):
    expression_modes = {
        'tree': {'alpha': 1., 'rates': [.1]*4},
        'mix': {'alpha': 0.2, 'rates': [.1]*4},
        'external': {'alpha': 0., 'rates': [.1]*4},
    }
    # Each row is a gene program (signature), each column is a gene
    # 1 indicates the gene is active in that program, 0 otherwise
    trait_signatures = np.zeros((n_traits, n_genes))
    # Make non-overlapping gene programs: each gene belongs to only one program
    genes = np.arange(n_genes)
    split_genes = np.array_split(genes, n_traits)
    for k, gene_indices in enumerate(split_genes):
        trait_signatures[k, gene_indices] = 1
    trait_signatures = pd.DataFrame(trait_signatures, index=["T" + str(i) for i in range(n_traits)], columns=["G" + str(i) for i in range(n_genes)])

    # Brownian Motion model on the tree

    # Create two blocks of highly correlated genes (block diagonal covariance)
    block_size = n_traits // 2
    block1 = np.ones((block_size, block_size)) * 0.8 + np.eye(block_size) * 0.2 
    block2 = np.ones((n_traits - block_size, n_traits - block_size)) * 0.8 + np.eye(n_traits - block_size) * 0.2
    trait_covariances = np.zeros((n_traits, n_traits))
    trait_covariances[:block_size, :block_size] = block1
    trait_covariances[block_size:, block_size:] = block2
    trait_covariances = trait_covariances / 1.

    trait_covariances = pd.DataFrame(trait_covariances, index=["T" + str(i) for i in range(n_traits)], columns=["T" + str(i) for i in range(n_traits)])
    spatial_activation = visium_simulator.prior(200, visium_simulator.circ_equation, decay_factor=.0005, radius=50, center_x=100, center_y=100)
    spatial_gene_program = np.zeros((n_genes,))
    spatial_gene_program[np.random.choice(n_genes, size=5)] = 1. # activate these genes

    subclades = []
    for subclade in ['4', '5', '6', '7']:
        subclade_leaves = simulated_tree.leaves_in_subtree(subclade)
        subclades.append(subclade_leaves)
    rates = expression_modes[expression_mode]['rates']
    
    ex_simulator = expression_simulator.ExpressionSimulator(trait_covariances, trait_signatures, spatial_activation, spatial_gene_program)
    ex_simulator.overlay_data(simulated_tree, alpha=expression_modes[expression_mode]['alpha'], clades=subclades, rates=rates) # 0: fully external factors, 1: fully tree
    expression = simulated_tree.cell_meta[[f'G{i}' for i in range(ex_simulator.n_genes)]].loc[ex_simulator.leaf_names]#.to_numpy()
    trait_activations = simulated_tree.cell_meta[[f'T{i}' for i in range(ex_simulator.n_traits)]].loc[ex_simulator.leaf_names]
    spatial_activations = simulated_tree.cell_meta[ex_simulator.spatial_activation_name].loc[ex_simulator.leaf_names]

    # Simulate single-cell gene expression data
    if obs_model == 'normal':
        ss_transcriptomes = np.random.normal(expression, sigma)
    elif obs_model == 'poisson':
        ss_cell_library_sizes = np.random.gamma(10., 1. * 100, size=ex_simulator.n_leaves) * 0 + lib_size
        ss_gene_sizes = np.random.poisson(100, size=ex_simulator.n_genes) * 0 + 1
        ss_transcriptomes = np.random.poisson(np.exp(expression.values)/np.sum(np.exp(expression.values), axis=1)[:, None] * ss_cell_library_sizes[:, None] * ss_gene_sizes[None, :]) # np.random.normal(expression, .1)#        
    else:
        raise ValueError(f"Invalid observation model: {obs_model}")

    ss_simulated_adata = anndata.AnnData(pd.DataFrame(ss_transcriptomes, index=expression.index, columns=expression.columns))
    ss_simulated_adata.raw = ss_simulated_adata.copy()
    
    ss_simulated_adata.layers['expression'] = expression.values
    ss_simulated_adata.uns['trait_signatures'] = ex_simulator.trait_signatures
    ss_simulated_adata.obsm['trait_activations'] = trait_activations
    for trait in ex_simulator.trait_names:
        ss_simulated_adata.obs[trait] = trait_activations[trait]        
    ss_simulated_adata.obs['spatial_activations'] = spatial_activations

    ss_simulated_adata.obs['library_id'] = 'simulated_puck'            
    ss_simulated_adata.obsm['spatial'] = simulated_tree.cell_meta[['spatial_0', 'spatial_1']].loc[expression.index].to_numpy()

    # Show clade in UMAP
    ss_simulated_adata.uns['clade_level2_colors'] = ['#3182bd', '#9ecae1', '#fc9272', '#de2d26']
    
    # Top clades
    top_clades = simulated_tree.children(simulated_tree.root)
    for i, clade in enumerate(top_clades):
        leaves = simulated_tree.leaves_in_subtree(clade)
        ss_simulated_adata.obs.loc[leaves, 'clade_level0'] = 'clade' + str(clade)
        for j, subclade in enumerate(simulated_tree.children(clade)):
            subclade_leaves = simulated_tree.leaves_in_subtree(subclade)
            ss_simulated_adata.obs.loc[subclade_leaves, 'clade_level1'] = 'clade' + str(subclade)
            for k, subsubclade in enumerate(simulated_tree.children(subclade)):
                subsubclade_leaves = simulated_tree.leaves_in_subtree(subsubclade)
                ss_simulated_adata.obs.loc[subsubclade_leaves, 'clade_level2'] = 'clade' + str(subsubclade)

    return ss_simulated_adata


def simulate_visium(ss_simulated_adata, obs_model='normal', sigma=0.1, lib_size=10000):
    n_genes = ss_simulated_adata.shape[1]
    n_traits = ss_simulated_adata.obsm['trait_activations'].shape[1]

    x = ss_simulated_adata.obsm['spatial'][:, 0]
    y = ss_simulated_adata.obsm['spatial'][:, 1]
    gx, gy = visium_simulator.map_points_to_grid_lowerleft(
        x, y,
        xmin=0.0, ymin=0.0, 
        dx=.05, dy=0.05
    )

    cell_spot_locations = np.c_[gx, gy]
    spots = np.unique(cell_spot_locations, axis=0)
    # Make a vector of cell to spot names indicating which spot each cell belongs to
    # For each cell, find the index of its spot in the unique spots array, then assign the corresponding spot name
    cells_to_spots = np.array([
        f'spot_{np.where((spots == loc).all(axis=1))[0][0]}' for loc in cell_spot_locations
    ])
    ss_simulated_adata.obs['spot'] = cells_to_spots

    # Simulate Visium data
    n_spots = spots.shape[0]
    spot_expression = np.zeros((n_spots, n_genes))
    spot_trait_activations = np.zeros((n_spots, n_traits))
    spot_spatial_activations = np.zeros((n_spots, 1))
    spot_names = [f'spot_{i}' for i in range(n_spots)]
    cells_in_spots = []
    clades_fractions = []
    clade_level2_assignments = []
    clade_level1_assignments = []
    for spot in range(n_spots):
        spot_cells = np.where(cells_to_spots == f'spot_{spot}')[0]
        spot_expression[spot] = np.mean(ss_simulated_adata.layers['expression'][spot_cells], axis=0) # all transcripts
        spot_trait_activations[spot] = np.mean(ss_simulated_adata.obsm['trait_activations'].iloc[spot_cells], axis=0) # all traits
        spot_spatial_activations[spot] = np.mean(ss_simulated_adata.obs['spatial_activations'].iloc[spot_cells], axis=0) # all spatial activations
        cells_in_spots.append(len(spot_cells))
        # Ensure clade_counts contains all possible clade_level2 values, fill missing with 0
        clade_counts = ss_simulated_adata.obs.iloc[spot_cells]['clade_level2'].value_counts()
        clade_counts = clade_counts.reindex(ss_simulated_adata.obs['clade_level2'].unique(), fill_value=0)
        clades_fractions.append(clade_counts/len(spot_cells))
        clade_level2_assignments.append(ss_simulated_adata.obs.iloc[spot_cells]['clade_level2'].value_counts().idxmax())
        clade_level1_assignments.append(ss_simulated_adata.obs.iloc[spot_cells]['clade_level1'].value_counts().idxmax())
    if obs_model == 'normal':
        spatial_transcriptomes = np.random.normal(spot_expression, sigma)
    elif obs_model == 'poisson':
        spatial_spot_library_sizes = np.random.poisson(lib_size, size=n_spots) * 0 + lib_size
        spatial_spot_gene_sizes = np.random.poisson(100, size=n_genes) * 0 + 1
        spatial_transcriptomes = np.random.poisson(np.exp(spot_expression)/np.sum(np.exp(spot_expression), axis=1)[:, None] * spatial_spot_library_sizes[:, None] * spatial_spot_gene_sizes[None, :]) # np.random.normal(spot_expression, .01)#
    else:
        raise ValueError(f"Invalid observation model: {obs_model}")

    spatial_simulated_adata = anndata.AnnData(pd.DataFrame(spatial_transcriptomes, index=spot_names, columns=ss_simulated_adata.var_names))
    spatial_simulated_adata.obsm['spatial'] = spots
    spatial_simulated_adata.obsm['clade_level2_fractions'] = pd.DataFrame(clades_fractions, index=spot_names, columns=ss_simulated_adata.obs['clade_level2'].unique()).loc[spatial_simulated_adata.obs.index]
    spatial_simulated_adata.obs['clade_level2'] = clade_level2_assignments
    spatial_simulated_adata.obs['clade_level1'] = clade_level1_assignments
    spatial_simulated_adata.obs['cells_in_spots'] = cells_in_spots

    spot_trait_activations = pd.DataFrame(spot_trait_activations, index=spot_names, columns=ss_simulated_adata.obsm['trait_activations'].columns)
    for trait in spot_trait_activations.columns:
        spatial_simulated_adata.obs[trait] = spot_trait_activations[trait]

    spatial_simulated_adata.obs["spatial_activations"] = spot_spatial_activations

    # Sort spots per clade
    spatial_simulated_adata = spatial_simulated_adata[spatial_simulated_adata.obs['clade_level2'].sort_values().index]
    return spatial_simulated_adata

def simulate_data(spatial_mode, expression_mode, n_cells=512, n_genes=10, n_traits=5, obs_model='normal', sigma=0.1, ss_lib_size=10000, spatial_lib_size=10000, seed=42):
    np.random.seed(seed)
    tree_simulator = cas.sim.CompleteBinarySimulator(num_cells=n_cells)

    # Simulate spatial locations
    simulated_tree = simulate_spatial_locations(tree_simulator, spatial_mode)
    
    # Simulate expression
    ss_simulated_adata = simulate_expression(simulated_tree, expression_mode, n_traits, n_genes, obs_model=obs_model, sigma=sigma, lib_size=ss_lib_size)

    # Simulate spatial gene expression data
    spatial_simulated_adata = simulate_visium(ss_simulated_adata, obs_model=obs_model, sigma=sigma, lib_size=spatial_lib_size)

    # Get distance matrices
    tree_distance_matrix = cas.data.compute_phylogenetic_weight_matrix(simulated_tree)
    tree_distance_matrix = tree_distance_matrix / tree_distance_matrix.max()
    spatial_distance_matrix = squareform(pdist(spatial_simulated_adata.obsm['spatial'], metric='euclidean'))
    spatial_distance_matrix = spatial_distance_matrix / spatial_distance_matrix.max()
    spatial_distance_matrix = pd.DataFrame(spatial_distance_matrix, index=spatial_simulated_adata.obs.index, columns=spatial_simulated_adata.obs.index)

    # Get true couplings
    true_couplings = np.zeros((ss_simulated_adata.shape[0], spatial_simulated_adata.shape[0]))
    for i, cell in enumerate(ss_simulated_adata.obs.index):
        spot = ss_simulated_adata.obs.loc[cell, 'spot']
        spot_idx = np.where(spatial_simulated_adata.obs.index == spot)[0][0]
        true_couplings[i, spot_idx] = 1

    return ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix, true_couplings, simulated_tree


def compute_inferred_clade_fractions(ss_simulated_adata, coupling, spatial_simulated_adata, clade_column='clade_level2'):
    """
    Compute a (clades x spots) matrix containing the normalized average coupling weight of each clade in each spot,
    using the true cell-clade assignments.
    
    The normalized matrix (fractions per spot summing to 1) is stored in 
    `spatial_simulated_adata.obsm['inferred_clade_level2_fractions']`.
    """
    # Compute per-spot clade fractions from the given coupling matrix
    # coupling: (cells, spots) matrix, likely inferred cell-to-spot assignment (soft/hard assignments)

    # Map clade for each cell
    cell_clades = ss_simulated_adata.obs[clade_column]
    # Get list of unique clades and spots (in order)
    clades = pd.Categorical(cell_clades).categories
    spots = spatial_simulated_adata.obs.index

    # For each cell, get the vector over spots (coupling), and clade assignment
    n_clades = len(clades)
    n_spots = len(spots)
    inferred_clade_fractions = np.zeros((n_spots, n_clades))  # spots x clades

    # Convert cell_clades to categorical codes
    clade_codes = pd.Categorical(cell_clades, categories=clades).codes

    # Sum the weights assigned from cells-of-clade c to each spot s, for each c,s
    # coupling: cells x spots
    for clade_idx, clade in enumerate(clades):
        # mask cells belonging to this clade
        cells_in_clade = (clade_codes == clade_idx)
        # sum their couplings to each spot
        clade_coupling = coupling[cells_in_clade, :]
        if clade_coupling.ndim == 1:
            clade_coupling = clade_coupling[np.newaxis, :]
        # sum across cells in this clade
        inferred_clade_fractions[:, clade_idx] = clade_coupling.sum(axis=0)

    # Normalize per spot (rows sum to 1)
    sum_per_spot = inferred_clade_fractions.sum(axis=1, keepdims=True)
    sum_per_spot[sum_per_spot == 0] = 1  # avoid division by zero
    inferred_clade_fractions = inferred_clade_fractions / sum_per_spot

    # Save to .obsm using correct order (spots by clades)
    inferred_clade_fractions_df = pd.DataFrame(
        inferred_clade_fractions,
        index=spots,
        columns=[f"{clade}" for clade in clades]
    )
    key = f'inferred_{clade_column}_fractions'
    spatial_simulated_adata.obsm[key] = inferred_clade_fractions_df


def compute_clades_pearson_corr(true_fractions, inferred_fractions, return_all=False):
    """
    Compute the mean Pearson correlation between true and estimated clade fractions across spots.

    Parameters
    ----------
    true_fractions : array-like
        True clade fractions.
    inferred_fractions : array-like
        Inferred clade fractions.

    Returns
    -------
    float
        Mean Pearson correlation across clades.
    """
    per_clade_corrs = []

    for clade in true_fractions.columns:
        r, _ = pearsonr(true_fractions[clade], inferred_fractions[clade])
        per_clade_corrs.append(r)

    mean_pearson_corr = np.mean(per_clade_corrs)
    if return_all:
        return mean_pearson_corr, per_clade_corrs
    else:
        return mean_pearson_corr


def dc_square(D2, w):
    # double-center a square matrix with weights w (sum=1)
    r = D2 @ w            # [n]
    c = D2.T @ w          # same as r if D2 symmetric
    mu = np.dot(w, r)
    return D2 - r[:,None] - c[None,:] + mu

def structural_concordance(C_T, C_S, gamma, eps=1e-12):
    # 1) normalize mapping
    mass = np.sum(gamma) + eps
    G = gamma / mass
    # 2) induced marginals
    a = np.sum(G, axis=1)   # [n], sum=1
    b = np.sum(G, axis=0)   # [m], sum=1
    # 3) double-center squared distances
    CT2 = C_T * C_T
    CS2 = C_S * C_S
    CTc = dc_square(CT2, a)
    CSc = dc_square(CS2, b)
    # 4) transported inner product
    Num = np.trace(CTc @ G @ CSc.T @ G.T)
    # 5) weighted norms
    sigT2 = np.sum((a[:,None]*a[None,:]) * (CTc**2))
    sigS2 = np.sum((b[:,None]*b[None,:]) * (CSc**2))
    denom = np.sqrt(np.maximum(sigT2, eps) * np.maximum(sigS2, eps))
    corr = np.where(denom > 0, Num / denom, 0.0)
    # 6) map to [0,1]
    score = 0.5 * (corr + 1.0)
    return score

def compute_rmse(true_scores, inferred_scores):
    return np.sqrt(np.mean((true_scores - inferred_scores)**2))


def annotate_clades(tdata_slice, max_depth=4, min_cells=30):
    lcas = py.tl.clades(tdata_slice, depth=4, depth_key="time", update=False, copy=True)
    lcas.set_index("node", inplace=True)
    # Create dict mapping nodes to clades, relabeling clades with <100 cells as 'NA'
    clade_counts = tdata_slice.obs["clade"].value_counts()
    node_to_clade = {}
    for node, clade in lcas["clade"].items():
        clade_str = str(clade)
        if clade_counts.get(clade_str, 0) < min_cells:
            node_to_clade[node] = "NA"
        else:
            node_to_clade[node] = clade_str

    # Find all unique non-NA clade labels and map them to new indices starting from 0
    non_na_clades = sorted({clade for clade in node_to_clade.values() if clade != "NA"})
    clade_reindex = {old: str(idx) for idx, old in enumerate(non_na_clades)}

    # Now, build reindexed node_to_clade
    node_to_clade_reindexed = {}
    for node, clade in node_to_clade.items():
        if clade == "NA":
            node_to_clade_reindexed[node] = "NA"
        else:
            node_to_clade_reindexed[node] = clade_reindex[clade]

    relabeled_lca = py.tl.clades(tdata_slice, clades=node_to_clade_reindexed, depth_key="time", update=False, copy=True, key_added=f"clade_depth{max_depth}")
    NA_nodes = relabeled_lca.loc[relabeled_lca[f"clade_depth{max_depth}"] == "NA"]["node"].values

    clade_palette = py.get.palette(tdata_slice, key="clade_depth4", cmap="rainbow")
    clade_palette['NA'] = 'lightgray'
    py.pl.tree(tdata_slice, depth_key="time", branch_color="clade_depth4", palette=clade_palette, legend=True)

    # And now go for 1 - (max_depth-1)
    for depth in range(0, max_depth):
        lcas = py.tl.clades(tdata_slice, depth=depth, depth_key="time", update=False, copy=True)
        lcas.set_index("node", inplace=True)
        # Create dict mapping nodes to clades, relabeling clades with <100 cells as 'NA'
        clade_counts = tdata_slice.obs["clade"].value_counts()
        node_to_clade = {}
        for node, clade in lcas["clade"].items():
            clade_str = str(clade)
            if clade_counts.get(clade_str, 0) < min_cells or clade_str in NA_nodes:
                node_to_clade[node] = "NA"
            else:
                node_to_clade[node] = clade_str

        # Find all unique non-NA clade labels and map them to new indices starting from 0
        non_na_clades = sorted({clade for clade in node_to_clade.values() if clade != "NA"})
        clade_reindex = {old: str(idx) for idx, old in enumerate(non_na_clades)}

        # Now, build reindexed node_to_clade
        node_to_clade_reindexed = {}
        for node, clade in node_to_clade.items():
            if clade == "NA":
                node_to_clade_reindexed[node] = "NA"
            else:
                node_to_clade_reindexed[node] = clade_reindex[clade]

        relabeled_lca = py.tl.clades(tdata_slice, clades=node_to_clade_reindexed, depth_key="time", update=False, copy=True, key_added=f"clade_depth{depth}")
        
        clade_palette = py.get.palette(tdata_slice, key=f"clade_depth{depth}", cmap="rainbow")
        clade_palette['NA'] = 'lightgray'
        py.pl.tree(tdata_slice, depth_key="time", branch_color=f"clade_depth{depth}", palette=clade_palette, legend=True);