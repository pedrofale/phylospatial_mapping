"""
This file defines the BrownianSpatialDataSimulator, which is a subclass of
the SpatialDataSimulator. The BrownianSpatialDataSimulator simulates spatial
coordinates by simulating Brownian motion of each cell.
"""
from re import A
from typing import Optional, Union, Iterable, List, Dict
import numpy as np
import pandas as pd
import itertools

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DataSimulatorError
from cassiopeia.simulator.SpatialDataSimulator import SpatialDataSimulator
import ete3
from scipy.linalg import cholesky

class BrownianSpatialDataSimulator(SpatialDataSimulator):
    """
    Simulate spatial data with a Brownian motion process.

    This subclass of `SpatialDataSimulator` simulates the spatial coordinates of
    each cell in the provided `CassiopeiaTree` through a Brownian motion process.

    The simulation procedure is as follows. The tree is traversed from the root
    to the leaves. The the root cell is placed at the origin. At each split
    (i.e. when a cell divides), two new cells are placed at new coordinate X
    relative to the position of the parent X' (so, the absolute coordinate is
    X' + X). X is a n-dimensional vector with x_i ~ Normal(0, 2*D*t), where D is
    the diffusion coefficient and t is the time since the last cell division. X
    is sampled independently for each dimension for each cell, so the two new
    cells will be placed at different coordinates. Note that this process is
    dependent on the scale of the branch lengths.

    Args:
        dim: Number of spatial dimensions. For instance, a value of 2 indicates
            a 2D slice.
        diffusion_coeficient: The diffusion coefficient to use in the Brownian
            motion process. Specifically, 2 * `diffusion_coefficient` * (branch
            length) is the variance of the Normal distribution.
        scale_unit_area: Whether or not the space should be scaled to
            have unit length in all dimensions. Defaults to `True`.
        random_seed: A seed for reproducibility

    Raises:
        DataSimulatorError if `dim` is less than equal to zero, or the diffusion
            coefficient is negative.
    """

    def __init__(
        self,
        dim: int,
        diffusion_coefficient: float,
        scale_unit_area: bool = True,
        random_seed: Optional[int] = None,
    ):
        if dim <= 0:
            raise DataSimulatorError("Number of dimensions must be positive.")
        if diffusion_coefficient < 0:
            raise DataSimulatorError(
                "Diffusion coefficient must be non-negative."
            )

        self.dim = dim
        self.diffusion_coefficient = diffusion_coefficient
        self.scale_unit_area = scale_unit_area
        self.random_seed = random_seed

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

    def pagels_lambda(self, lambda_brownian):
        leaf_cov_matrix = np.diag(np.diag(self.leaf_cov_matrix))
        leaf_cov_matrix = leaf_cov_matrix + lambda_brownian * (self.leaf_cov_matrix - leaf_cov_matrix)
        leaf_cholesky = np.asarray(cholesky(leaf_cov_matrix.values, lower=True))
        return leaf_cov_matrix, leaf_cholesky

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

    def sample_brownian_motion(self, D=1., lambda_brownian=1., clades=None, rates=None):
        # Use Cholesky decomposition for efficient sampling
        # Build the block covariance matrix as before
        if lambda_brownian != 1.:
            _, leaf_cholesky = self.pagels_lambda(lambda_brownian)
        else:
            leaf_cholesky = self.leaf_cholesky

        if clades is not None:
            _, leaf_cholesky = self.rescale_clades(clades, rates)

        V_chol = np.kron(np.diag(np.ones(self.dim)) * np.sqrt(D), leaf_cholesky)
        # Sample standard normals
        z = np.random.randn(self.dim * self.n_leaves)
        # Transform to correlated samples
        leaf_trait_values = V_chol @ z
        leaf_trait_values = leaf_trait_values.reshape(self.n_leaves, -1, order='F')  # leaves by dimensions
        return pd.DataFrame(leaf_trait_values, index=self.leaf_names, columns=['x', 'y'])  # is this in the right order?

    def overlay_data(
        self,
        tree: CassiopeiaTree,
        make_cov_matrix: bool = True,
        lambda_brownian: Union[float, np.ndarray] = 1.,
        clades: Optional[List[List[str]]] = None,
        rates: Optional[List[float]] = None,
        attribute_key: str = "spatial",
    ):
        """Overlays spatial data onto the CassiopeiaTree via Brownian motion.

        Args:
            tree: The CassiopeiaTree to overlay spatial data on to.
            attribute_key: The name of the attribute to save the coordinates as.
                This also serves as the prefix of the coordinates saved into
                the `cell_meta` attribute as `{attribute_key}_i` where i is
                an integer from 0...`dim-1`.
        """
        if self.random_seed:
            np.random.seed(self.random_seed)

        if make_cov_matrix and not hasattr(self, 'leaf_cov_matrix'):
            phylotree = ete3.Tree(tree.get_newick())
            self.make_leaf_cov_matrix(phylotree)

        if make_cov_matrix:
            locations_df = self.sample_brownian_motion(D=self.diffusion_coefficient, lambda_brownian=lambda_brownian, clades=clades, rates=rates)
            locations = {tree.root: np.zeros(self.dim)}
            for child in tree.leaves_in_subtree(tree.root): # this is iterating through edges, not leaves, which what I have in my locations_df
                locations[child] = locations_df.loc[child]
        else:
            # Using numpy arrays instead of tuples for easy vector operations
            locations = {tree.root: np.zeros(self.dim)}
            for parent, child in tree.depth_first_traverse_edges(source=tree.root):
                parent_location = locations[parent]
                branch_length = tree.get_branch_length(parent, child)

                locations[child] = parent_location + np.random.normal(
                    scale=np.sqrt(2 * self.diffusion_coefficient * branch_length),
                    size=self.dim,
                )

        # Scale if desired
        # Note that Python dictionaries preserve order since 3.6
        if self.scale_unit_area:
            all_coordinates = np.array(list(locations.values()))

            # Shift each dimension so that the smallest value is at 0.
            all_coordinates -= all_coordinates.min(axis=0)

            # Scale all dimensions (by the same value) so that all values are
            # between [0, 1]. We don't scale each dimension separately because
            # we want to retain the shape of the distribution.
            all_coordinates /= all_coordinates.max()
            locations = {
                node: coordinates
                for node, coordinates in zip(locations.keys(), all_coordinates)
            }

        # Set node attributes
        for node, loc in locations.items():
            tree.set_attribute(node, attribute_key, tuple(loc))

        # Set cell meta
        cell_meta = (
            tree.cell_meta.copy()
            if tree.cell_meta is not None
            else pd.DataFrame(index=tree.leaves)
        )
        columns = [f"{attribute_key}_{i}" for i in range(self.dim)]
        cell_meta[columns] = np.nan
        for leaf in tree.leaves:
            cell_meta.loc[leaf, columns] = locations[leaf]
        tree.cell_meta = cell_meta