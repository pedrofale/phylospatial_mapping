import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import optax
from tqdm import tqdm
import numpy as np
import ot

def deconvolution_loss(Y, gamma, cell_type_assignments, cell_type_signatures, sigma):
    """
    Outer loss: scalar function of the transport plan gamma [n, m].
    """
    spot_cell_type_proportions = jnp.einsum('ji,ik->jk', gamma.T * gamma.shape[1], cell_type_assignments)
    spot_mean = jnp.einsum('jk,kg->jg', spot_cell_type_proportions, cell_type_signatures)
    return -tfp.distributions.Normal(spot_mean, sigma).log_prob(Y).sum()

@jax.jit
def compute_Lgw(C_T, C_S, a, b, gamma):
    """
    C_T: [n, n]   cost matrix for target
    C_S: [m, m]   cost matrix for source
    a:   [n]      histogram on target
    b:   [m]      histogram on source
    gamma: [n, m] transport plan

    Returns:
      L: [n, m] with
         L_ij = [(C_T)^{∘2} a]_i + [(C_S)^{∘2} b]_j - 2 [C_T γ (C_S)^T]_{ij}
    """
    # CT2_a = jnp.einsum('ik,k->i', C_T * C_T, a)
    # CS2_b = jnp.einsum('jl,l->j', C_S * C_S, b)
    # cross = jnp.einsum('ik,kl,jl->ij', C_T, gamma, C_S)
    # return CT2_a[:, None] + CS2_b[None, :] - 2.0 * cross
    CT2, CS2 = C_T*C_T, C_S*C_S
    return (CT2 @ a)[:,None] + (CS2 @ b)[None,:] - 2.0 * (C_T @ gamma @ C_S.T)

@jax.jit
def compute_Lcladegw(C_T, C_S, a, b, gamma, Omega):
    """
    C_T   : [n, n]
    C_S   : [m, m]
    a     : [n]
    b     : [m]   (unused here but kept for signature symmetry)
    gamma : [n, m]
    Omega : [n, n]

    Returns:
      L_omega : [n, m]
        L^Ω = ((Ω ⊙ C_T^2) @ a)[:, None]
              + Ω @ gamma @ (C_S^2).T
              - 2 * Ω @ (C_T @ gamma @ C_S.T)
    """    
    CT2 = C_T * C_T
    CS2 = C_S * C_S
    # term1_i = sum_k (Omega_ik * CT2_ik) * a_k
    term1 = (Omega * CT2) @ a
    # term1_row = jnp.einsum('ik,ik,k->i', Omega, CT2, a)
    # term1 = term1_row[:, None]
    term2 = Omega @ gamma @ CS2.T          # Ω γ (C_S^2)^T
    # term2 = jnp.einsum('ip,pm,mq->iq', Omega, gamma, CS2)          # Ω γ (C_S^2)^T
    cross = (Omega * C_T) @ gamma @ C_S.T          # C_T γ C_S^T
    return term1[:,None] + term2 - 2.0 * cross

def dc(M, a, b):
    m = jnp.mean(M)
    M = M - jnp.mean(M,axis=0)[None,:] - jnp.mean(M,axis=1)[:,None] + m
    return M

def mad_abs(M, eps=1e-12):
    return jnp.median(jnp.abs(M)) + eps

def build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, sF_ref, sL_ref, zero_thresh=1e-8, *args, **kwargs):
    """
    Return the FGW cost matrix C(alpha) with shape [n, m].
    Example (blend): C = (1 - alpha) * C_struct + alpha * C_feat
    """
    # center
    F_c = dc(C_feature, a, b)
    L_c = dc(compute_Lgw(C_tree, C_space, a, b, gamma), a, b)
    # fixed scales (computed once; pass in as scalars, no grad)
    # normalize
    F_hat = F_c #/ jnp.maximum(sF_ref, zero_thresh)
    L_hat = L_c#jnp.where(sL_ref < zero_thresh, 0.0, L_c / sL_ref)
    C = ((1-alpha) * F_hat + alpha * L_hat)
    C = C / 1
    return C


def build_cladefgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, omega, Omega, *args, **kwargs):
    """
    Return the FGW cost matrix C(alpha) with shape [n, m].
    Example (blend): C = (1 - alpha) * C_struct + alpha * C_feat
    alpha is a vector of size n_clades
    omega is a matrix of size n_cells x n_clades indicating the clade of each cell
    Omega is a symmetric matrix of size n_cells x n_cells indicating wether two cells belong to the same clade
    """
    L_cladegw = compute_Lcladegw(C_tree, C_space, a, b, gamma, Omega)
    F_c = dc(C_feature, a, b)
    L_c = dc(L_cladegw, a, b)    
    alphas = omega @ alpha
    C = (1-alphas[:,None])*F_c + alphas[:,None]*L_c
    s = 1./jnp.median(jnp.abs(C))
    C = C #* s
    return C


def sinkhorn_unrolled(C, a, b, eps, T, uv0=None):
    """
    C:  [n, m]
    a:  [n]  (sum=1)
    b:  [m]  (sum=1)
    eps: float
    T:   int (iterations)
    uv0: optional warm-start tuple (u0, v0)

    Returns: (gamma, (u, v)) with shapes [n, m], [n], [m]
    """
    K = jnp.exp(-C / eps)  # [n, m]

    if uv0 is None:
        # normalized ones is usually fine
        u0 = jnp.ones_like(a)
        v0 = jnp.ones_like(b)
    else:
        u0, v0 = uv0

    def body(_, carry):
        u, v = carry
        Kv = K @ v + 1e-12
        u = a / Kv
        KTu = K.T @ u + 1e-12
        v = b / KTu
        return (u, v)

    u, v = jax.lax.fori_loop(0, T, body, (u0, v0))
    gamma = (u[:, None]) * K * (v[None, :])  # diag(u) @ K @ diag(v)
    return gamma, (u, v)


def sinkhorn_unrolled_safe(C, a, b, eps, T, uv0=None, tiny=1e-300):
    # real-domain, guarded
    Cmax = 60.0 * eps
    C = jnp.clip(C, -Cmax, Cmax)
    K = jnp.exp(-C / eps) + tiny
    u = jnp.ones_like(a) if uv0 is None else jnp.maximum(uv0[0], tiny)
    v = jnp.ones_like(b) if uv0 is None else jnp.maximum(uv0[1], tiny)
    supp_a, supp_b = (a > 0), (b > 0)
    def body(_, carry):
        u, v = carry
        u = jnp.where(supp_a, a / jnp.maximum(K @ v, tiny), 0.0)
        v = jnp.where(supp_b, b / jnp.maximum(K.T @ u, tiny), 0.0)
        return (u, v)
    u, v = jax.lax.fori_loop(0, T, body, (u, v))
    gamma = (u[:,None]) * K * (v[None,:])
    return gamma, (u, v)

def sinkhorn_fgw(C_feature, C_tree, C_space, a, b, eps, T_sinkhorn=50, J_alt=3, alpha=0.5, gamma0=None, uv0=None):
    if gamma0 is None:
        gamma0 = (a[:, None] * b[None, :])
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    sF_ref = mad_abs(dc(C_feature,a,b))                      # fixed
    sL_ref = mad_abs(dc(compute_Lgw(C_tree,C_space,a,b,gamma0),a,b))  # at a reference gamma_ref

    # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
    def one_round(gamma, uv):
        C = build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, sF_ref, sL_ref)
        return sinkhorn_unrolled_safe(C, a, b, eps, T_sinkhorn, uv)

    # Unroll J_alt rounds with carry
    def body(carry, _):
        gamma, uv = carry
        gamma, uv = one_round(gamma, uv)
        return (gamma, uv), None

    (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)
    return gamma_star, uv_star

def sinkhorn_cladefgw(C_feature, C_tree, C_space, a, b, eps, omega, Omega, T_sinkhorn=50, J_alt=3, alpha=0.5, gamma0=None, uv0=None):
    # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
    def one_round(gamma, uv):
        C = build_cladefgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, omega, Omega)
        return sinkhorn_unrolled(C, a, b, eps, T_sinkhorn, uv)

    # Unroll J_alt rounds with carry
    def body(carry, _):
        gamma, uv = carry
        gamma, uv = one_round(gamma, uv)
        return (gamma, uv), None

    if gamma0 is None:
        gamma0 = (a[:, None] * b[None, :])
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)
    return gamma_star, uv_star    

def make_step_fn_fgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, sF_ref, sL_ref, T_sinkhorn=50, J_alt=3):
    def loss_fn(beta, gamma_uv):
        gamma0, uv0 = gamma_uv
        alpha = jax.nn.sigmoid(beta)  # α ∈ (0,1)

        # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
        def one_round(gamma, uv):
            C = build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, sF_ref, sL_ref)
            return sinkhorn_unrolled_safe(C, a, b, eps, T_sinkhorn, uv)

        # Unroll J_alt rounds with carry
        def body(carry, _):
            gamma, uv = carry
            gamma, uv = one_round(gamma, uv)
            return (gamma, uv), None

        (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)

        loss = deconvolution_loss(Y, gamma_star, cell_type_assignments, cell_type_signatures, sigma)
        return loss, ((gamma_star, uv_star), alpha)

    @jax.jit
    def step(beta, opt_state, gamma_uv):
        (loss_value, (gamma_uv_new, alpha)), g = jax.value_and_grad(loss_fn, has_aux=True)(beta, gamma_uv)
        updates, opt_state = optimizer.update(g, opt_state, params=beta)
        beta = optax.apply_updates(beta, updates)
        return beta, opt_state, gamma_uv_new, loss_value, alpha

    return step

def learn_alpha_gamma_fgw(
    C_feature, Y, C_tree, C_space, a, b,
    cell_type_assignments, cell_type_signatures, sigma,
    eps=0.05, T_sinkhorn=50, J_alt=3,
    K_outer=200, lr=1e-2, beta0=0.0,
    gamma0=None, uv0=None,
):
    n = a.shape[0]; m = b.shape[0]
    if gamma0 is None:
        # uniform feasible plan as a warm start
        gamma0 = (a[:, None] * b[None, :])
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    sF_ref = mad_abs(dc(C_feature,a,b))                      # fixed
    sL_ref = mad_abs(dc(compute_Lgw(C_tree,C_space,a,b,gamma0), a, b))  # at a reference gamma_ref

    optimizer = optax.adam(lr)
    beta = jnp.array(beta0)
    opt_state = optimizer.init(beta)

    step = make_step_fn_fgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, sF_ref, sL_ref, T_sinkhorn, J_alt)

    gamma_uv = (gamma0, uv0)
    loss_hist, alpha_hist = [], []

    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            beta, opt_state, gamma_uv, loss_value, alpha = step(beta, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alpha_hist.append(float(alpha))
            pbar.set_postfix({'loss': f"{float(loss_value):.6g}"})

    alpha_final = jax.nn.sigmoid(beta)
    return float(alpha_final), jnp.array(alpha_hist), jnp.array(loss_hist), gamma_uv[0]

def make_step_fn_cladefgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, omega, Omega,T_sinkhorn=50, J_alt=3):
    def loss_fn(betas, gamma_uv):
        gamma0, uv0 = gamma_uv
        alphas = jax.nn.sigmoid(betas)  # α ∈ (0,1)

        # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
        def one_round(gamma, uv):
            C = build_cladefgw_cost(alphas, C_feature, C_tree, C_space, a, b, gamma, omega, Omega)
            return sinkhorn_unrolled(C, a, b, eps, T_sinkhorn, uv)

        # Unroll J_alt rounds with carry
        def body(carry, _):
            gamma, uv = carry
            gamma, uv = one_round(gamma, uv)
            return (gamma, uv), None

        (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)

        loss = deconvolution_loss(Y, gamma_star, cell_type_assignments, cell_type_signatures, sigma)
        return loss, ((gamma_star, uv_star), alphas)

    @jax.jit
    def step(betas, opt_state, gamma_uv):
        (loss_value, (gamma_uv_new, alphas)), g = jax.value_and_grad(loss_fn, has_aux=True)(betas, gamma_uv)
        updates, opt_state = optimizer.update(g, opt_state, params=betas)
        betas = optax.apply_updates(betas, updates)
        return betas, opt_state, gamma_uv_new, loss_value, alphas

    return step

def learn_alpha_gamma_cladefgw(
    C_feature, Y, C_tree, C_space, a, b,
    cell_type_assignments, cell_type_signatures, sigma,
    omega, Omega,
    eps=0.05, T_sinkhorn=50, J_alt=3,
    K_outer=200, lr=1e-2, beta0=0.0,
    gamma0=None, uv0=None,
):
    n = a.shape[0]; m = b.shape[0]
    if gamma0 is None:
        # uniform feasible plan as a warm start
        gamma0 = (a[:, None] * b[None, :])
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    optimizer = optax.adam(lr)
    betas = jnp.array(beta0)
    opt_state = optimizer.init(betas)

    step = make_step_fn_cladefgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, omega, Omega, T_sinkhorn, J_alt)

    gamma_uv = (gamma0, uv0)
    loss_hist, alphas_hist = [], []

    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            betas, opt_state, gamma_uv, loss_value, alphas = step(betas, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alphas_hist.append(alphas)
            pbar.set_postfix({'loss': float(loss_value)})

    alpha_final = jax.nn.sigmoid(betas)
    return alpha_final, jnp.stack(alphas_hist, axis=1), jnp.array(loss_hist), gamma_uv[0]


def prepare_ot_inputs(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix):
    X = jnp.asarray(ss_simulated_adata.X)
    Y = jnp.asarray(spatial_simulated_adata.X)
    a = jnp.ones(X.shape[0]) / X.shape[0]
    b = jnp.ones(Y.shape[0]) / Y.shape[0]
    C_tree = jnp.asarray(tree_distance_matrix)
    C_space = jnp.asarray(spatial_distance_matrix)
    C_feature = jnp.array(ot.dist(np.array(X), np.array(Y)))
    C_feature = C_feature / C_feature.max()
    return C_feature, C_tree, C_space, a, b

def run_spotr(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix, 
            cell_type_assignments, cell_type_signatures, clade_column='clade_level2', sigma=0.01,
            eps=0.01, T_sinkhorn=100, J_alt=20, K_outer=100, lr=1e-2):
    
    C_feature, C_tree, C_space, a, b = prepare_ot_inputs(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix)
    Y = jnp.asarray(spatial_simulated_adata.X)            
    n_cells = ss_simulated_adata.shape[0]
    n_clades = len(ss_simulated_adata.obs[clade_column].unique())

    beta0 = jnp.zeros((n_clades,))

    # omega: (n_cells, n_clades), one-hot encoding each cell's clade
    cell_clades = ss_simulated_adata.obs[clade_column].values
    clade_to_idx = {clade: i for i, clade in enumerate(np.unique(cell_clades))}
    omega = np.zeros((n_cells, n_clades), dtype=np.float32)
    for i, clade in enumerate(cell_clades):
        omega[i, clade_to_idx[clade]] = 1.0

    # Omega is (n_cells, n_cells), 1 if same clade, else 0. Now rescale so within each clade, values are 1/(size of clade).
    Omega = (omega @ omega.T).astype(np.float32)
    row_sums = Omega.sum(axis=1, keepdims=True)
    Omega = Omega / np.maximum(row_sums, 1e-8)
    omega = jnp.array(omega)
    Omega = jnp.array(Omega)

    alphas, alphas_hist, loss_hist, coupling = learn_alpha_gamma_cladefgw(C_feature, Y, C_tree, C_space, a, b,
                                        jnp.array(cell_type_assignments), jnp.array(cell_type_signatures), sigma,
                                        omega, Omega,
                                        eps=eps, T_sinkhorn=T_sinkhorn, J_alt=J_alt, 
                                        K_outer=K_outer, lr=lr, beta0=beta0, gamma0=None, uv0=None)    

    return alphas, alphas_hist, loss_hist, coupling