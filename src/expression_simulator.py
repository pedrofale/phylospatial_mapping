from typing import Optional, List, Iterable

import numpy as np
import pandas as pd
import itertools
import ete3
from scipy.linalg import cholesky

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.DataSimulator import DataSimulator


class ExpressionSimulator(DataSimulator):
    """
    Simulate gene expression data from the lineage tree and spatial coordinates.

    This subclass of `ExpressionDataSimulator` simulates the gene expression of
    each cell in the provided `CassiopeiaTree` with tree-based correlations and spatial constraints.
    This requires that each cell has a spatial coordinate.

    The simulation procedure is as follows.
    1. Given a trait covariance matrix indicating how genes evolve, and a variance-covariance
    matrix encoding the tree structure, the trait values of the leaves are sampled from a multivariate
    normal distribution, corresponding to a Brownian Motion model.
    2. Exogenous spatial effects are added to the trait values according to a mixed effects model
    where effects have a spatial structure (sinusoid, circle, etc)
    3. The two terms are combined in a convex combination with an `alpha`. 
    The gene expression values for the leaves are added to the anndata object as a layer.

    Args:
        trait_covariances: trait-trait covariance matrix 
        spatial_covariances: Pixel-pixel covariance matrix
        random_seed: A seed for reproducibility
    """

    def __init__(
        self,
        trait_covariances: np.ndarray,
        trait_signatures: np.ndarray,
        spatial_map: np.ndarray,
        spatial_program: np.ndarray,
        random_seed: Optional[int] = None,
    ):
        self.trait_covariances = trait_covariances # gene-gene covariance matrix
        self.trait_cholesky = np.asarray(cholesky(self.trait_covariances.values, lower=True))
        self.trait_signatures = trait_signatures # trait-trait signature matrix
        self.spatial_map = spatial_map # spatial heatmap indicating the activity of the spatial program at each pixel
        self.spatial_program = spatial_program # spatial program indicating the activity of each trait
        self.random_seed = random_seed
        self.n_traits = trait_covariances.shape[0]
        self.n_genes = trait_signatures.shape[1]
        self.trait_names = ["T" + str(i) for i in range(self.n_traits)]
        self.gene_names = ["G" + str(i) for i in range(self.n_genes)]
        
    def make_leaf_cov_matrix(self, phylotree):
        # Create covariance matrix for species from tree 
        species_cov_matrix = pd.DataFrame(index=phylotree.get_leaf_names(), columns=phylotree.get_leaf_names())
        species_cov_matrix.index = species_cov_matrix.columns
        
        def flatten(x):
            if isinstance(x, list):
                out = []
                for item in x:
                    if isinstance(item, list):
                        out.extend(flatten(item))
                    else:
                        out.append(item)
                return out
            else:
                return [x]

        def descend(root, total_length=0):
            clades = []
            for child in root.children:
                clades.append(descend(child, total_length=total_length + root.dist))
            if root.is_leaf():
                species_cov_matrix.loc[root.name, root.name] = total_length + root.dist
                return root.name
            else:
                for c1, c2 in itertools.combinations(clades, 2):
                    a = np.array(flatten(c1), dtype=object)
                    b = np.array(flatten(c2), dtype=object)
                    pairs = np.stack(np.meshgrid(a, b), -1).reshape(-1, 2)
                    for pair in pairs:
                        species_cov_matrix.loc[pair[0], pair[1]] = total_length + root.dist
            return clades

        descend(phylotree.get_tree_root(), total_length=0)
        species_cov_matrix = species_cov_matrix.combine_first(species_cov_matrix.T)
        self.leaf_cov_matrix = species_cov_matrix.astype(float)
        self.leaf_cholesky = np.asarray(cholesky(self.leaf_cov_matrix.values, lower=True))
        self.n_leaves = len(self.leaf_cov_matrix)
        self.leaf_names = self.leaf_cov_matrix.index

    def _indices_for_clade(self, tip_labels: List[str], leaves: Iterable[str]) -> np.ndarray:
        idx = {t: i for i, t in enumerate(tip_labels)}
        I = np.array([idx[t] for t in leaves], dtype=int)
        if len(np.unique(I)) != len(I):
            raise ValueError("Duplicate tips in a clade.")
        return I

    def _clade_root_depth(self, Sigma: np.ndarray, I: np.ndarray) -> float:
        """
        Depth (shared path length) from the tree root to the clade root.
        For |I|>=2: min off-diagonal within the clade.
        For |I|==1: parent depth = max off-diagonal in that tip's row.
        """
        if I.size == 1:
            i = I[0]
            if Sigma.shape[0] == 1:
                return 0.0
            return float(np.max(np.delete(Sigma[i, :], i)))
        S = Sigma[np.ix_(I, I)]
        mask = ~np.eye(S.shape[0], dtype=bool)
        return float(np.min(S[mask]))

    def rescale_clade(self, leaves, lam):
        """
        Rescale the covariance matrix of a clade by a given rate.
        Args:
            tip_labels: The names of the tips of the clade to rescale.
            lam: The rescaling factor of the covariance matrix of the clade.
        Returns:
            The rescaled covariance matrix and Cholesky decomposition.
        """
        Sigma = self.leaf_cov_matrix.values
        n = Sigma.shape[0]
        Sigma_p = Sigma.copy()

        I = self._indices_for_clade(self.leaf_names, leaves)
        lam = float(lam)
        if lam < 0:
            raise ValueError("lam must be nonnegative.")
        d_root = self._clade_root_depth(Sigma, I)

        # Inside-clade component A_C = Σ[I,I] - d_root
        S_block = Sigma[np.ix_(I, I)] - d_root
        # Update Σ′ on that block
        Sigma_p[np.ix_(I, I)] += (lam - 1.0) * S_block        

        Sigma_p = 0.5 * (Sigma_p + Sigma_p.T)
        eps = 1e-10 * np.mean(np.diag(Sigma_p))
        L = np.linalg.cholesky(Sigma_p + eps*np.eye(Sigma_p.shape[0]))

        return Sigma_p, L

    def rescale_clades(self, clades, rates):
        Sigma = self.leaf_cov_matrix.values
        n = Sigma.shape[0]
        Sigma_p = Sigma.copy()

        # Check disjointness
        all_idx = []
        for leaves in clades:
            all_idx.extend(self._indices_for_clade(self.leaf_names, leaves).tolist())
        if len(set(all_idx)) != len(all_idx):
            raise ValueError("Clades overlap: at least one tip appears in multiple clades.")

        # Apply each clade’s rescale
        for leaves, lam in zip(clades, rates):
            I = self._indices_for_clade(self.leaf_names, leaves)
            lam = float(lam)
            if lam < 0:
                raise ValueError("rates must be nonnegative.")
            d_root = self._clade_root_depth(Sigma, I)

            # Inside-clade component A_C = Σ[I,I] - d_root
            S_block = Sigma[np.ix_(I, I)] - d_root
            # Update Σ′ on that block
            Sigma_p[np.ix_(I, I)] += (lam - 1.0) * S_block

        # Symmetrize to clean numerical dust
        Sigma_p = 0.5 * (Sigma_p + Sigma_p.T)
        eps = 1e-10 * np.mean(np.diag(Sigma_p))
        L = np.linalg.cholesky(Sigma_p + eps*np.eye(Sigma_p.shape[0]))

        return Sigma_p, L

    def sample_brownian_motion(self, clades=None, rates=None):
        leaf_cholesky = self.leaf_cholesky
        trait_cholesky = self.trait_cholesky

        if clades is not None:
            _, leaf_cholesky = self.rescale_clades(clades, rates)

        V_chol = np.kron(trait_cholesky, leaf_cholesky)
        # Sample standard normals
        z = np.random.randn(self.n_traits * self.n_leaves)
        # Transform to correlated samples
        leaf_trait_values = V_chol @ z
        leaf_trait_values = leaf_trait_values.reshape(self.n_leaves, -1, order='F')  # leaves by genes
        return pd.DataFrame(leaf_trait_values, index=self.leaf_names, columns=self.trait_names)  # is this in the right order?

        # V = np.kron(self.trait_covariances, self.leaf_cov_matrix)
        # # Sample from the multivariate normal distribution
        # leaf_trait_values = np.random.multivariate_normal(np.zeros((self.n_genes*self.n_leaves,)), V) # zero-mean
        # leaf_trait_values = leaf_trait_values.reshape(self.n_leaves, -1, order='F') # leaves by genes
        # return pd.DataFrame(leaf_trait_values, index=self.leaf_names, columns=self.trait_covariances.index) # is this in the right order?

    def sample_spatial_effects(self, tree):
        # For each cell, check where in the grid it is
        leaf_spatial_effects = pd.DataFrame(index=self.leaf_names, columns=self.gene_names)
        for leaf in self.leaf_names:
            x, y = tree.cell_meta.loc[leaf, ['spatial_0', 'spatial_1']]
            x = int(x * (self.spatial_map.shape[0] - 1))
            y = int(y * (self.spatial_map.shape[1] - 1))
            leaf_spatial_effects.loc[leaf] = self.spatial_map[x, y] * self.spatial_program
        return leaf_spatial_effects 

    def sample_combined_expression(self, tree, alpha=0.5, clades=None, rates=None):
        brownian_motion_activations = self.sample_brownian_motion(clades=clades, rates=rates)
        brownian_motion_expression = np.exp(brownian_motion_activations).dot(self.trait_signatures)
        expression = alpha*brownian_motion_expression + (1-alpha)*self.sample_spatial_effects(tree)
        return expression, brownian_motion_activations

    def overlay_data(
        self,
        tree: CassiopeiaTree,
        alpha: float = 0.5,
        clades: Optional[List[List[str]]] = None,
        rates: Optional[List[float]] = None,
    ):
        """Overlays gene expression onto the AnnData object via Brownian motion on the tree and exogenous spatial effects.

        Args:
            tree: The CassiopeiaTree to overlay spatial data on to.
            attribute_key: The name of the attribute to save the expression values as.
                This also serves as the prefix of the expression values saved into
                the `cell_meta` attribute as `{attribute_key}i` where i is
                an integer from 0...`n_genes-1`.
        """
        if self.random_seed:
            np.random.seed(self.random_seed)

        if not hasattr(self, 'leaf_cov_matrix'):
            phylotree = ete3.Tree(tree.get_newick())
            self.make_leaf_cov_matrix(phylotree)
        expression, brownian_motion_activations = self.sample_combined_expression(tree, alpha=alpha, clades=clades, rates=rates)

        # Set cell meta
        cell_meta = (
            tree.cell_meta.copy()
            if tree.cell_meta is not None
            else pd.DataFrame(index=tree.leaves)
        )

        columns = self.gene_names
        cell_meta[columns] = np.nan
        for leaf in tree.leaves:
            cell_meta.loc[leaf, columns] = expression.loc[leaf]

        columns = self.trait_names
        cell_meta[columns] = np.nan
        for leaf in tree.leaves:
            cell_meta.loc[leaf, columns] = brownian_motion_activations.loc[leaf]

        tree.cell_meta = cell_meta