import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import optax
from tqdm import tqdm


def deconvolution_loss(Y, gamma, cell_type_assignments, cell_type_signatures, sigma):
    """
    Outer loss: scalar function of the transport plan gamma [n, m].
    """
    spot_cell_type_proportions = jnp.einsum('ji,ik->jk', gamma.T, cell_type_assignments)
    spot_mean = jnp.einsum('jk,kg->jg', spot_cell_type_proportions, cell_type_signatures)
    return tfp.distributions.Normal(spot_mean, sigma).log_prob(Y).sum()

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
    CT2_a = jnp.einsum('ik,k->i', C_T * C_T, a)
    CS2_b = jnp.einsum('jl,l->j', C_S * C_S, b)
    cross = jnp.einsum('ik,kl,jl->ij', C_T, gamma, C_S)
    return CT2_a[:, None] + CS2_b[None, :] - 2.0 * cross

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

def build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, *args, **kwargs):
    """
    Return the FGW cost matrix C(alpha) with shape [n, m].
    Example (blend): C = (1 - alpha) * C_struct + alpha * C_feat
    """
    L_gw = compute_Lgw(C_tree, C_space, a, b, gamma)
    return (1.0 - alpha) * C_feature + alpha * L_gw

def build_cladefgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, omega, Omega, *args, **kwargs):
    """
    Return the FGW cost matrix C(alpha) with shape [n, m].
    Example (blend): C = (1 - alpha) * C_struct + alpha * C_feat
    alpha is a vector of size n_clades
    omega is a matrix of size n_cells x n_clades indicating the clade of each cell
    Omega is a symmetric matrix of size n_cells x n_cells indicating wether two cells belong to the same clade
    """
    L_cladegw = compute_Lcladegw(C_tree, C_space, a, b, gamma, Omega)
    alphas = omega @ alpha
    C = (1-alphas[:,None])*C_feature + alphas[:,None]*L_cladegw
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

def sinkhorn_fgw(C_feature, C_tree, C_space, a, b, eps, T_sinkhorn=50, J_alt=3, alpha=0.5, gamma0=None, uv0=None):
    # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
    def one_round(gamma, uv):
        C = build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma)
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

def make_step_fn(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, T_sinkhorn=50, J_alt=3):
    def loss_fn(beta, gamma_uv):
        gamma0, uv0 = gamma_uv
        alpha = jax.nn.sigmoid(beta)  # α ∈ (0,1)

        # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
        def one_round(gamma, uv):
            C = build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma)
            return sinkhorn_unrolled(C, a, b, eps, T_sinkhorn, uv)

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

def learn_alpha_gamma(
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

    optimizer = optax.adam(lr)
    beta = jnp.array(beta0)
    opt_state = optimizer.init(beta)

    step = make_step_fn(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, T_sinkhorn, J_alt)

    gamma_uv = (gamma0, uv0)
    loss_hist, alpha_hist = [], []

    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            beta, opt_state, gamma_uv, loss_value, alpha = step(beta, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alpha_hist.append(float(alpha))
            pbar.set_postfix({'loss': float(loss_value)})

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


