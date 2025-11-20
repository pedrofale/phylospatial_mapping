import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csr_matrix, issparse
# from sklearn.utils.validation import check_array as check_arrays

import pandas as pd
import networkx as nx

import numpy as np
import torch
import torch.nn as nn
import ot

#!/usr/bin/env python
#
# OPTIMAL TRANSPORT NODE
# Implementation of differentiable optimal transport using implicit differentiation. Makes use of Sinkhorn normalization
# to solve the entropy regularized problem (Cuturi, NeurIPS 2013) in the forward pass. The problem can be written as
# Let us write the entropy regularized optimal transport problem in the following form,
#
#    minimize (over P) <P, M> + 1/gamma KL(P || rc^T)
#    subject to        P1 = r and P^T1 = c
#
# where r and c are m- and n-dimensional positive vectors, respectively, each summing to one. Here m-by-n matrix M is
# the input and m-by-n dimensional positive matrix P is the output. The above problem leads to a solution of the form
#
#   P_{ij} = alpha_i beta_j e^{-gamma M_{ij}}
#
# where alpha and beta are found by iteratively applying row and column normalizations.
#
# We also provide an option to parametrize the input in log-space as M_{ij} = -log Q_{ij} where Q is a positive matrix.
# The matrix Q becomes the input. This is more numerically stable for inputs M with large positive or negative values.
#
# See accompanying Jupyter Notebook at https://deepdeclarativenetworks.com.
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Fred Zhang <frederic.zhang@anu.edu.au>
#

import torch
import torch.nn as nn
import warnings


def sinkhorn(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False):
    """
    PyTorch function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector 1/H
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector 1/W

    You can back propagate through this function in O(TBWH) time where T is the number of iterations taken to converge.
    """

    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.amin(M, 2, keepdim=True)))

    for i in range(maxiters):
        alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        P = P / alpha.view(B, H, 1) * r

        beta = torch.sum(P, 1)
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        P = P / beta.view(B, 1, W) * c

    return P


def _sinkhorn_inline(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False):
    """As above but with inline calculations for when autograd is not needed."""

    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.amin(M, 2, keepdim=True)))

    for i in range(maxiters):
        alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        P /= alpha.view(B, H, 1)
        P *= r

        beta = torch.sum(P, 1)
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        P /= beta.view(B, 1, W)
        P *= c

    return P


class OptimalTransportFcn(torch.autograd.Function):
    """
    PyTorch autograd function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector

    Allows for approximate gradient calculations, which is faster and may be useful during early stages of learning,
    when exp(-gamma M) is already nearly doubly stochastic, or when gradients are otherwise noisy.

    Both r and c must be positive, if provided. They will be normalized to sum to one.
    """

    @staticmethod
    def forward(ctx, M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False, method='block'):
        """Solve optimal transport using skinhorn. Method can be 'block', 'full', 'fullchol' or 'approx'."""
        assert method in ('block', 'full', 'fullchol', 'approx')

        with torch.no_grad():
            # normalize r and c to ensure that they sum to one (and save normalization factor for backward pass)
            if r is not None:
                ctx.inv_r_sum = 1.0 / torch.sum(r, dim=1, keepdim=True)
                r = ctx.inv_r_sum * r
            if c is not None:
                ctx.inv_c_sum = 1.0 / torch.sum(c, dim=1, keepdim=True)
                c = ctx.inv_c_sum * c

            # run sinkhorn
            P = _sinkhorn_inline(M, r, c, gamma, eps, maxiters, logspace)

        ctx.save_for_backward(M, r, c, P)
        ctx.gamma = gamma
        ctx.logspace = logspace
        ctx.method = method

        return P

    @staticmethod
    def backward(ctx, dJdP):
        """Implement backward pass using implicit differentiation."""

        M, r, c, P = ctx.saved_tensors
        B, H, W = M.shape

        # initialize backward gradients (-v^T H^{-1} B with v = dJdP and B = I or B = -1/r or B = -1/c)
        dJdM = -1.0 * ctx.gamma * P * dJdP
        dJdr = None if not ctx.needs_input_grad[1] else torch.zeros_like(r)
        dJdc = None if not ctx.needs_input_grad[2] else torch.zeros_like(c)

        # return approximate gradients
        if ctx.method == 'approx':
            return dJdM, dJdr, dJdc, None, None, None, None, None, None

        # compute exact row and column sums (in case of small numerical errors or forward pass not converging)
        alpha = torch.sum(P, dim=2)
        beta = torch.sum(P, dim=1)

        # compute [vHAt1, vHAt2] = v^T H^{-1} A^T as two blocks
        vHAt1 = torch.sum(dJdM[:, 1:H, 0:W], dim=2).view(B, H - 1, 1)
        vHAt2 = torch.sum(dJdM, dim=1).view(B, W, 1)

        # compute [v1, v2] = -v^T H^{-1} A^T (A H^{-1] A^T)^{-1}
        if ctx.method == 'block':
            # by block inverse of (A H^{-1] A^T)
            PdivC = P[:, 1:H, 0:W] / beta.view(B, 1, W)
            RminusPPdivC = torch.diag_embed(alpha[:, 1:H]) - torch.bmm(P[:, 1:H, 0:W], PdivC.transpose(1, 2))
            try:
                block_11 = torch.linalg.cholesky(RminusPPdivC)
            except:
                # block_11 = torch.ones((B, H-1, H-1), device=M.device, dtype=M.dtype)
                block_11 = torch.eye(H - 1, device=M.device, dtype=M.dtype).view(1, H - 1, H - 1).repeat(B, 1, 1)
                for b in range(B):
                    try:
                        block_11[b, :, :] = torch.linalg.cholesky(RminusPPdivC[b, :, :])
                    except:
                        # keep initialized values (gradient will be close to zero)
                        warnings.warn("backward pass encountered a singular matrix")
                        pass

            block_12 = torch.cholesky_solve(PdivC, block_11)
            #block_22 = torch.diag_embed(1.0 / beta) + torch.bmm(block_12.transpose(1, 2), PdivC)
            block_22 = torch.bmm(block_12.transpose(1, 2), PdivC)

            v1 = torch.cholesky_solve(vHAt1, block_11) - torch.bmm(block_12, vHAt2)
            #v2 = torch.bmm(block_22, vHAt2) - torch.bmm(block_12.transpose(1, 2), vHAt1)
            v2 = vHAt2 / beta.view(B, W, 1) + torch.bmm(block_22, vHAt2) - torch.bmm(block_12.transpose(1, 2), vHAt1)

        else:
            # by full inverse of (A H^{-1] A^T)
            AinvHAt = torch.empty((B, H + W - 1, H + W - 1), device=M.device, dtype=M.dtype)
            AinvHAt[:, 0:H - 1, 0:H - 1] = torch.diag_embed(alpha[:, 1:H])
            AinvHAt[:, H - 1:H + W - 1, H - 1:H + W - 1] = torch.diag_embed(beta)
            AinvHAt[:, 0:H - 1, H - 1:H + W - 1] = P[:, 1:H, 0:W]
            AinvHAt[:, H - 1:H + W - 1, 0:H - 1] = P[:, 1:H, 0:W].transpose(1, 2)

            if ctx.method == 'fullchol':
                v = torch.cholesky_solve(torch.cat((vHAt1, vHAt2), dim=1), torch.linalg.cholesky(AinvHAt))
            else:
                v = torch.bmm(torch.inverse(AinvHAt), torch.cat((vHAt1, vHAt2), dim=1))
                #v = torch.linalg.solve(AinvHAt, torch.cat((vHAt1, vHAt2), dim=1))

            v1 = v[:, 0:H - 1].view(B, H - 1, 1)
            v2 = v[:, H - 1:H + W - 1].view(B, W, 1)

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} A H^{-1} B - v^T H^{-1} B
        dJdM[:, 1:H, 0:W] -= v1 * P[:, 1:H, 0:W]
        dJdM -= v2.view(B, 1, W) * P

        # multiply by derivative of log(M) if in log-space
        if ctx.logspace:
            dJdM /= -1.0 * M

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} (A H^{-1} B - C) - v^T H^{-1} B
        if dJdr is not None:
            dJdr = ctx.inv_r_sum.view(r.shape[0], 1) / ctx.gamma * \
                   (torch.sum(r[:, 1:H] * v1.view(B, H - 1), dim=1, keepdim=True) - torch.cat((torch.zeros(B, 1, device=r.device), v1.view(B, H - 1)), dim=1))

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} (A H^{-1} B - C) - v^T H^{-1} B
        if dJdc is not None:
            dJdc = ctx.inv_c_sum.view(c.shape[0], 1) / ctx.gamma * (torch.sum(c * v2.view(B, W), dim=1, keepdim=True) - v2.view(B, W))

        # return gradients (None for gamma, eps, maxiters and logspace)
        return dJdM, dJdr, dJdc, None, None, None, None, None, None


class OptimalTransportLayer(nn.Module):
    """
    Neural network layer to implement optimal transport.

    Parameters:
    -----------
    gamma: float, default: 1.0
        Inverse of the coefficient on the entropy regularisation term.
    eps: float, default: 1.0e-6
        Tolerance used to determine the stop condition.
    maxiters: int, default: 1000
        The maximum number of iterations.
    logspace: bool, default: False
        If `True`, assumes that the input is provided as log M
        If `False`, assumes that the input is provided as M (standard optimal transport)
    method: str, default: 'block'
        If `approx`, approximate the gradient by assuming exp(-gamma M) is already nearly doubly stochastic.
        It is faster and could potentially be useful during early stages of training.
        If `block`, exploit the block structure of matrix A H^{-1] A^T.
        If `full`, invert the full A H^{-1} A^T matrix without exploiting the block structure
    """

    def __init__(self, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False, method='block'):
        super(OptimalTransportLayer, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.maxiters = maxiters
        self.logspace = logspace
        self.method = method

    def forward(self, M, r=None, c=None):
        """
        Parameters:
        -----------
        M: torch.Tensor
            Input matrix/matrices of size (H, W) or (B, H, W)
        r: torch.Tensor, optional
            Row sum constraint in the form of a 1xH or BxH matrix. Are assigned uniformly as 1/H by default.
        c: torch.Tensor, optional
            Column sum constraint in the form of a 1xW or BxW matrix. Are assigned uniformly as 1/W by default.

        Returns:
        --------
        torch.Tensor
            Normalised matrix/matrices, with the same shape as the inputs
        """
        M_shape = M.shape
        # Check the number of dimensions
        ndim = len(M_shape)
        if ndim == 2:
            M = M.unsqueeze(dim=0)
        elif ndim != 3:
            raise ValueError(f"The shape of the input tensor {M_shape} does not match that of an matrix")

        # Handle special case of 1x1 matrices
        nr, nc = M_shape[-2:]
        if nr == 1 and nc == 1:
            P = torch.ones_like(M)
        else:
            P = OptimalTransportFcn.apply(M, r, c, self.gamma, self.eps, self.maxiters, self.logspace, self.method)

        return P.view(*M_shape)


#
# --- testing ---
#

if __name__ == '__main__':
    from torch.autograd import gradcheck
    from torch.nn.functional import normalize

    torch.manual_seed(0)
    print(torch.__version__)

    M = torch.randn((3, 5, 7), dtype=torch.double, requires_grad=True)
    f = OptimalTransportFcn().apply

    print(torch.all(torch.isclose(sinkhorn(M), f(M))))
    print(torch.all(torch.isclose(sinkhorn(M), sinkhorn(torch.exp(-1.0 * M), logspace=True))))

    test = gradcheck(f, (M, None, None, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (torch.exp(-1.0 * M), None, None, 1.0, 1.0e-6, 1000, True, 'block'), eps=1e-6, atol=1e-3,
                     rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, None, 1.0, 1.0e-6, 1000, False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, None, 10.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, None, 10.0, 1.0e-6, 1000, False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, None, 10.0, 1.0e-6, 1000, False, 'fullchol'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    r = normalize(torch.rand((M.shape[0], M.shape[1]), dtype=torch.double, requires_grad=False), p=1.0)
    c = normalize(torch.rand((M.shape[0], M.shape[2]), dtype=torch.double, requires_grad=False), p=1.0)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-9, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    # with r and c inputs
    r = normalize(torch.rand((M.shape[0], M.shape[1]), dtype=torch.double, requires_grad=True), p=1.0)
    c = normalize(torch.rand((M.shape[0], M.shape[2]), dtype=torch.double, requires_grad=True), p=1.0)

    test = gradcheck(f, (M, r, None, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, c, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, r, c, 10.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    # shared r and c
    r = normalize(torch.rand((1, M.shape[1]), dtype=torch.double, requires_grad=True), p=1.0)
    c = normalize(torch.rand((1, M.shape[2]), dtype=torch.double, requires_grad=True), p=1.0)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False, 'fullchol'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    # shared M, different r and c
    r = normalize(torch.rand((M.shape[0], M.shape[1]), dtype=torch.double, requires_grad=True), p=1.0)
    c = normalize(torch.rand((M.shape[0], M.shape[2]), dtype=torch.double, requires_grad=True), p=1.0)
    M_shared = torch.randn((M.shape[1], M.shape[2]), dtype=torch.double, requires_grad=True)

    test = gradcheck(f, (M_shared.view(1, M.shape[1], M.shape[2]).repeat(M.shape[0], 1, 1),
                         r, c, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M_shared.view(1, M.shape[1], M.shape[2]).repeat(M.shape[0], 1, 1),
                         r, c, 1.0, 1.0e-6, 1000, False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M_shared.view(1, M.shape[1], M.shape[2]).repeat(M.shape[0], 1, 1),
                         r, c, 1.0, 1.0e-6, 1000, False, 'fullchol'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)


def safe_fro(X, squared=False):
    if issparse(X):
        nrm = np.sum(X.data ** 2)
    else:
        if hasattr(X, 'A'):
            X = X.A
        nrm = np.sum(X ** 2)
    return nrm if squared else np.sqrt(nrm)


# Author: Vlad Niculae
#         Lars Buitinck
# Author: Chih-Jen Lin, National Taiwan University (original projected gradient
#     NMF implementation)
# Author: Anthony Di Franco (original Python and NumPy port)
# License: BSD 3 clause
def nls_projgrad(X, Y, W_init=None, l1_reg=0, l2_reg=0, tol=1e-3, max_iter=5000,
                 sigma=0.01, beta=0.1, callback=None):
    """Non-negative least square solver

    Solves a non-negative least squares subproblem using the
    projected gradient descent algorithm.
    min 0.5 * || XW - Y ||^2_F + l1_reg * sum(W) + 0.5 * l2_reg * * ||W||^2_F

    Parameters
    ----------
    Y, X : array-like
        Constant matrices.

    W_init : array-like
        Initial guess for the solution.

    l1_reg, l2_reg : float,
        Regularization factors

    tol : float
        Tolerance of the stopping condition.

    max_iter : int
        Maximum number of iterations before timing out.

    sigma : float
        Constant used in the sufficient decrease condition checked by the line
        search.  Smaller values lead to a looser sufficient decrease condition,
        thus reducing the time taken by the line search, but potentially
        increasing the number of iterations of the projected gradient
        procedure. 0.01 is a commonly used value in the optimization
        literature.

    beta : float
        Factor by which the step size is decreased (resp. increased) until
        (resp. as long as) the sufficient decrease condition is satisfied.
        Larger values allow to find a better step size but lead to longer line
        search. 0.1 is a commonly used value in the optimization literature.

    Returns
    -------
    W : array-like
        Solution to the non-negative least squares problem.

    grad : array-like
        The gradient.

    n_iter : int
        The number of iterations done by the algorithm.

    Reference
    ---------

    C.-J. Lin. Projected gradient methods
    for non-negative matrix factorization. Neural
    Computation, 19(2007), 2756-2779.
    http://www.csie.ntu.edu.tw/~cjlin/nmf/

    """
    # X, Y = check_arrays(X, Y)
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    XY = safe_sparse_dot(X.T, Y)
    G = np.dot(X.T, X)

    if W_init is None:
        W = np.zeros((n_features, n_targets), dtype=np.float64)
    else:
        W = W_init.copy()
    init_grad_norm = safe_fro(np.dot(G, W) - XY + l2_reg * W + l1_reg)
    # values justified in the paper
    alpha = 1
    for n_iter in range(1, max_iter + 1):
        grad = np.dot(G, W) - XY  # X'(XW - Y) using precomputation
        if l2_reg:
            grad += l2_reg * W
        if l1_reg:
            grad += l1_reg

        # The following multiplication with a boolean array is more than twice
        # as fast as indexing into grad.
        proj_grad_norm = np.linalg.norm(grad * np.logical_or(grad < 0, W > 0))
        if proj_grad_norm / init_grad_norm < tol:
            break

        W_prev = W

        for inner_iter in range(20):
            # Gradient step.
            W_next = W - alpha * grad
            # Projection step.
            W_next *= W_next > 0
            d = W_next - W
            gradd = np.dot(grad.ravel(), d.ravel())
            dGd = np.dot(np.dot((G + l2_reg), d).ravel(), d.ravel())
            suff_decr = (1 - sigma) * gradd + 0.5 * dGd < 0
            if inner_iter == 0:
                decr_alpha = not suff_decr

            if decr_alpha:
                if suff_decr:
                    W = W_next
                    break
                else:
                    alpha *= beta
            elif not suff_decr or (W_prev == W_next).all():
                W = W_prev
                break
            else:
                alpha /= beta
                W_prev = W_next
        if callback:
            callback(W)

    if n_iter == max_iter:
        print("PG failed to converge")
    residual = safe_fro(np.dot(X, W) - Y)
    return W, residual


def dec_byot(sc_adata, st_adata, annotations, coupling):
    sc_adata = sc_adata
    st_adata = st_adata
    annotations = annotations
    coupling = coupling

    ctxc = pd.DataFrame(sc_adata.obs[annotations]).reset_index()
    G = nx.from_pandas_edgelist(
        ctxc,
        source=annotations,
        target='index',
        # create_using=nx.DiGraph()
    )
    adj_df = pd.DataFrame(
        nx.adjacency_matrix(G).todense(),
        index=G.nodes,
        columns=G.nodes
    )
    ct = adj_df.loc[np.unique(ctxc[annotations]).tolist()] \
        [np.unique(ctxc['index']).tolist()]
    ct = ct.T.reindex(sc_adata.obs_names)

    P = coupling.T @ ct.to_numpy()
    P = P / P.sum(1)[:, np.newaxis]
    P_df = pd.DataFrame(P, columns=ct.columns,
                        index=st_adata.obs_names)
    st_adata.obsm['deconvolution'] = P_df

    # also copy as single field in the anndata for visualization
    for ct in st_adata.obsm["deconvolution"].columns:
        st_adata.obs[ct] = st_adata.obsm["deconvolution"][ct]

    return st_adata

class scDOT(nn.Module):
    def __init__(self, sc_adata, st_adata):
        super(scDOT, self).__init__()
        self.NNLS = NNLS(sc_adata, st_adata)
        self.OT = OptimalTransportLayer(method='approx')
        self.M = nn.Parameter(torch.from_numpy(
            ot.dist(st_adata.X, sc_adata.X, metric='cosine'))
        )

    def forward(self, sc_adata, st_adata):
        NNLS_output = self.NNLS(sc_adata, st_adata)
        OT_output = self.OT(self.M)
        OT_output = OT_output/OT_output.sum(0) # col (cell) sum to 1
        return NNLS_output, OT_output

class NNLS(torch.nn.Module):
    def __init__(self, sc_adata, st_adata):
        super(NNLS, self).__init__()
        markers = st_adata.uns['markers']  #.to_numpy()
        st = st_adata.X
        self.W = nn.Parameter(torch.randn(st.shape[0], markers.shape[0]))

    def forward(self, sc_adata, st_adata):
        markers = st_adata.uns['markers']  #.to_numpy()
        st = st_adata.X
        W_nnls, _ = nls_projgrad(markers.T, st.T)
        W_nnls = W_nnls.T
        W_nnls = W_nnls/W_nnls.sum(1)[:,None]
        self.W.data.copy_(torch.tensor(W_nnls).float())
        return self.W


def train_scdot(sc_adata, st_adata, lr=1.0e-1, iters=10):
    # The code in this function is copied from the scDOT readme.md (https://github.com/namtk/scDOT/blob/main/README.md)
    model = scDOT(sc_adata, st_adata)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for i in range(iters):
        # Forward pass
        NNLS_output, OT_output = model(sc_adata, st_adata)

        # ct = torch.tensor(sc_adata.obsm['cell_type'].values.T)
        ct = torch.tensor(sc_adata.obsm['cell_type'].T)
        P_true = NNLS_output @ ct.float()  # spots by cells
        P = P_true / P_true.sum(0)  # col (cell) sum to 1
        loss_fn = torch.nn.CosineEmbeddingLoss()
        loss = loss_fn(OT_output, P_true, torch.ones(P.shape[0]))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        optimizer.step()

    # This segment is added to return the final mapping
    with torch.no_grad():
        NNLS_output, OT_output = model(sc_adata, st_adata)
        # ct = torch.tensor(sc_adata.obsm['cell_type'].values.T)
        ct = torch.tensor(sc_adata.obsm['cell_type'].T)
        P_true = NNLS_output @ ct.float()
        P = P_true / P_true.sum(0)

    return model, P.numpy()