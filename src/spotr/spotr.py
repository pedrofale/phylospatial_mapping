import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import optax
from tqdm import tqdm
import numpy as np
import ot

def row_norm(M, eps=1e-12):
    return M / (M.sum(axis=1, keepdims=True) + eps)

def make_omega_rho(W_within, rho):
    # W_within: [n,n] ∈ {0,1}, X = 1-W
    X = jnp.ones_like(W_within) - W_within
    Wn, Xn = row_norm(W_within), row_norm(X)
    return (1.0 - rho) * Wn + rho * Xn  # [n,n]    

def geom_eps(k, eps0, eps_min, decay):
    return jnp.maximum(eps_min, eps0 * (decay ** k))

def cosine_eps(k, K_outer, eps0, eps_min):
    if K_outer <= 1: return eps_min
    t = k / (K_outer - 1.0)
    return eps_min + 0.5 * (eps0 - eps_min) * (1.0 + jnp.cos(jnp.pi * t))

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


# ---------- make_step for (alpha_vec, rho) ----------
def make_step_fn_cladefgw_rho(
    C_feature, Y, C_T, C_S, a, b, eps,
    optimizer, cell_type_assignments, cell_type_signatures, sigma,
    W_assign, W_within,
    T_sinkhorn=50, J_alt=3,
):
    def loss_fn(params, gamma_uv):
        # params: {"betas_alpha": [K], "beta_rho": ()}
        betas_alpha = params["betas_alpha"]
        beta_rho    = params["beta_rho"]

        alpha_vec = jax.nn.sigmoid(betas_alpha)  # [K] per-clade feature↔structure
        rho       = jax.nn.sigmoid(beta_rho)     # scalar within↔cross

        gamma0, uv0 = gamma_uv

        def one_round(gamma, uv):
            C = build_cladefgw_cost_rho(
                alpha_vec, rho, C_feature, C_T, C_S, a, b, gamma,
                W_assign, W_within,
            )
            return sinkhorn_unrolled(C, a, b, eps, T_sinkhorn, uv)

        def body(carry, _):
            gamma, uv = carry
            gamma, uv = one_round(gamma, uv)
            return (gamma, uv), None

        (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)
        loss = deconvolution_loss(Y, gamma_star, cell_type_assignments, cell_type_signatures, sigma)
        aux  = ( (gamma_star, uv_star), dict(alpha_vec=alpha_vec, rho=rho) )
        return loss, aux

    # jit-compilable step (optimizer captured by closure)
    @jax.jit
    def step(params, opt_state, gamma_uv):
        (loss_value, (gamma_uv_new, alphas_rho)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, gamma_uv)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, gamma_uv_new, loss_value, alphas_rho

    return step

# ---------- learn loop ----------
def learn_cladefgw_with_rho(
    C_feature, Y, C_T, C_S, a, b,
    cell_type_assignments, cell_type_signatures, sigma,
    W_assign, W_within,
    eps=0.05, T_sinkhorn=50, J_alt=3,
    K_outer=200, lr=1e-2,
    betas_alpha0=None, beta_rho0=0.0,
    gamma0=None, uv0=None,
):
    n_clades = W_assign.shape[1]
    if betas_alpha0 is None:
        betas_alpha0 = jnp.zeros(n_clades)
    if gamma0 is None:
        gamma0 = a[:, None] * b[None, :]
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    params = {
        "betas_alpha": jnp.array(betas_alpha0),  # → alpha_vec = sigmoid
        "beta_rho":    jnp.array(beta_rho0),     # → rho       = sigmoid
    }
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    step = make_step_fn_cladefgw_rho(
        C_feature, Y, C_T, C_S, a, b, eps,
        optimizer, cell_type_assignments, cell_type_signatures, sigma,
        W_assign, W_within, T_sinkhorn, J_alt,
    )

    gamma_uv = (gamma0, uv0)
    loss_hist   = []
    alpha_hist  = []
    rho_hist    = []

    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            params, opt_state, gamma_uv, loss_value, ar = step(params, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alpha_hist.append(ar["alpha_vec"])
            rho_hist.append(ar["rho"])
            pbar.set_postfix({'loss': f"{float(loss_value):.6g}"})

    alpha_final = jax.nn.sigmoid(params["betas_alpha"])
    rho_final   = jax.nn.sigmoid(params["beta_rho"])
    gamma_final, _ = gamma_uv

    return (
        dict(alpha=alpha_final, rho=rho_final),
        jnp.stack(alpha_hist, axis=1),  # [K, K_outer]
        jnp.array(rho_hist),            # [K_outer]
        jnp.array(loss_hist),           # [K_outer]
        gamma_final                     # [n,m]
    )

def make_step_fn_cladefgw_rho_annealed(
    C_feature, Y, C_T, C_S, a, b,
    optimizer, cell_type_assignments, cell_type_signatures, sigma,
    W_assign, W_within,
    T_sinkhorn=50, J_alt=3,
):
    # exactly like your make_step_fn_* before, except eps is a runtime arg
    def loss_fn(params, gamma_uv, eps):
        betas_alpha = params["betas_alpha"]
        beta_rho    = params["beta_rho"]
        alpha_vec = jax.nn.sigmoid(betas_alpha)
        rho       = jax.nn.sigmoid(beta_rho)
        gamma0, uv0 = gamma_uv

        def one_round(gamma, uv):
            C = build_cladefgw_cost_rho(
                alpha_vec, rho, C_feature, C_T, C_S, a, b, gamma,
                W_assign, W_within,
            )
            return sinkhorn_unrolled(C, a, b, eps, T_sinkhorn, uv)

        def body(carry, _):
            gamma, uv = carry
            gamma, uv = one_round(gamma, uv)
            return (gamma, uv), None

        (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)
        loss = deconvolution_loss(Y, gamma_star, cell_type_assignments, cell_type_signatures, sigma)
        aux  = ((gamma_star, uv_star), dict(alpha_vec=alpha_vec, rho=rho))
        return loss, aux

    @jax.jit
    def step(params, opt_state, gamma_uv, eps):
        (loss_value, (gamma_uv_new, ar)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, gamma_uv, eps)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, gamma_uv_new, loss_value, ar

    return step    

def learn_cladefgw_with_rho_annealed(
    C_feature, Y, C_T, C_S, a, b,
    cell_type_assignments, cell_type_signatures, sigma,
    W_assign, W_within,
    # annealing knobs
    eps0=0.2, eps_min=0.05, schedule="geometric", decay=0.95,
    # sinkhorn/outer
    T_sinkhorn=50, J_alt=3, K_outer=200, lr=1e-2,
    betas_alpha0=None, beta_rho0=0.0, gamma0=None, uv0=None,
):
    n_clades = W_assign.shape[1]
    if betas_alpha0 is None: betas_alpha0 = jnp.zeros(n_clades)
    if gamma0 is None:       gamma0 = a[:, None] * b[None, :]
    if uv0 is None:          uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    params = {"betas_alpha": jnp.array(betas_alpha0),
              "beta_rho":    jnp.array(beta_rho0)}
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    step = make_step_fn_cladefgw_rho_annealed(
        C_feature, Y, C_T, C_S, a, b,
        optimizer, cell_type_assignments, cell_type_signatures, sigma,
        W_assign, W_within,
        T_sinkhorn=T_sinkhorn, J_alt=J_alt,
    )

    gamma_uv = (gamma0, uv0)
    loss_hist, alpha_hist, rho_hist, eps_hist = [], [], [], []

    with tqdm(range(K_outer)) as pbar:
        for k in pbar:
            # pick epsilon_k
            if schedule == "geometric":
                eps_k = float(geom_eps(k, eps0, eps_min, decay))
            elif schedule == "cosine":
                eps_k = float(cosine_eps(k, K_outer, eps0, eps_min))
            else:
                raise ValueError("schedule must be 'geometric' or 'cosine'")

            params, opt_state, gamma_uv, loss_val, ar = step(params, opt_state, gamma_uv, eps_k)

            loss_hist.append(float(loss_val))
            alpha_hist.append(ar["alpha_vec"])
            rho_hist.append(ar["rho"])
            eps_hist.append(eps_k)
            pbar.set_postfix({'loss': f"{float(loss_val):.6g}"})

    alpha_final = jax.nn.sigmoid(params["betas_alpha"])
    rho_final   = jax.nn.sigmoid(params["beta_rho"])
    gamma_final, _ = gamma_uv

    return (
        dict(alpha=alpha_final, rho=rho_final),
        jnp.stack(alpha_hist, axis=1),   # [K, K_outer]
        jnp.array(rho_hist),             # [K_outer]
        jnp.array(eps_hist),             # [K_outer]
        jnp.array(loss_hist),            # [K_outer]
        gamma_final
    )    


### Including all levels above 
def zero_diag(M):  # keep off-diagonal only
    return M * (1.0 - jnp.eye(M.shape[0], dtype=M.dtype))

def build_M_levels(partitions, n, zero_diagonal=True):
    """
    partitions: list of levels; level ℓ is a list of 1D arrays with leaf indices in each cluster
      e.g., for the 8-leaf example:
        L0: [jnp.array([0,1,2,3,4,5,6,7])]
        L1: [jnp.array([0,1,2,3]), jnp.array([4,5,6,7])]
        L2: [jnp.array([0,1]), jnp.array([2,3]), jnp.array([4,5]), jnp.array([6,7])]
        L3: [jnp.array([0]), ..., jnp.array([7])]
    Returns: list M_levels with M^(ℓ) ∈ {0,1}^{n×n}
    """
    M_levels = []
    for clusts in partitions:
        M = jnp.zeros((n, n), dtype=jnp.float32)
        for S in clusts:
            S = jnp.asarray(S, dtype=jnp.int32)
            mask = jnp.zeros((n,), dtype=jnp.float32).at[S].set(1.0)
            M = M + mask[:, None] * mask[None, :]
        if zero_diagonal:
            M = zero_diag(M)
        M_levels.append(jnp.clip(M, 0.0, 1.0))
    return M_levels

def build_R_levels_from_M(M_levels):
    """
    R^(ℓ) = M^(ℓ) - M^(ℓ+1), ℓ=0..L-1. Assumes M_levels are ordered 0..L.
    """
    R_levels = []
    for l in range(len(M_levels)-1):
        R = jnp.clip(M_levels[l] - M_levels[l+1], 0.0, 1.0)
        R_levels.append(R)
    # optional: R^(L) = M^(L) (usually diagonal only -> dropped if zero_diag True)
    return R_levels  # list of [n,n]

def row_norm(M, eps=1e-12):
    return M / (M.sum(axis=1, keepdims=True) + eps)

def omega_eff_from_levels(w_levels, R_levels, normalize=True):
    """
    w_levels: [D+1] nonnegative weights for levels 0..D
    R_levels: list [R^(0),...,R^(D)] each [n,n]
    """
    if normalize:
        Rn = [row_norm(R) for R in R_levels]
    else:
        Rn = R_levels
    R_stack = jnp.stack(Rn, axis=0)  # [D+1, n, n]
    return jnp.einsum('l, lij -> ij', w_levels, R_stack)

def build_cost_per_level(alpha_vec, w_levels,  # α per clade (mapped to rows), weights per level
                         C_feature, C_T, C_S, a, b, gamma,
                         W_assign, R_levels,       # R_levels up to depth D
                         detach_gamma_in_C=False, normalize_masks=True):
    Omega_eff = omega_eff_from_levels(w_levels, R_levels, normalize=normalize_masks)
    gamma_arg = jax.lax.stop_gradient(gamma) if detach_gamma_in_C else gamma

    # your masked FGW structural term:
    CT2 = C_T * C_T; CS2 = C_S * C_S
    term1 = jnp.einsum('ik,ik,k->i', Omega_eff, CT2, a)[:, None]
    term2 = jnp.einsum('ip,pm,mq->iq', Omega_eff, gamma_arg, CS2)
    cross = jnp.einsum('ip,pm,mq->iq', C_T, gamma_arg, C_S.T)
    L_struct = term1 + term2 - 2.0 * (Omega_eff @ cross)

    alpha_row = W_assign @ alpha_vec
    return (1.0 - alpha_row)[:, None] * C_feature + alpha_row[:, None] * L_struct

def make_step_fn_cladefgw_levels(
    C_feature, Y, C_T, C_S, a, b,
    optimizer, A_types, S_sigs, sigma,          # outer loss pieces
    W_assign, R_levels,                          # α level map & residual masks (up to D)
    *, T_sinkhorn=50, J_alt=3,
    normalize_masks=True, detach_gamma_in_C=False,
    lambda_w=0.0, w_prior=0.0                    # regularizer: λ * ||w - w_prior||^2
):
    L_used = len(R_levels)

    def loss_fn(params, gamma_uv, eps):
        betas_alpha = params["betas_alpha"]                 # [K]
        betas_w     = params["betas_w"]                     # [L_used]

        alpha_vec = jax.nn.sigmoid(betas_alpha)             # [K] in (0,1)
        # choose softplus (>=0) or sigmoid; softplus + tiny ε keeps strictly positive if you like
        w_levels  = jax.nn.softplus(betas_w)                # [L_used] ≥ 0

        gamma0, uv0 = gamma_uv

        def one_round(gamma, uv):
            C = build_cost_per_level(
                alpha_vec, w_levels, C_feature, C_T, C_S, a, b, gamma,
                W_assign, R_levels,
                normalize_masks=normalize_masks, detach_gamma_in_C=detach_gamma_in_C
            )
            return sinkhorn_unrolled(C, a, b, eps, T_sinkhorn, uv)

        def body(carry, _):
            gamma, uv = carry
            gamma, uv = one_round(gamma, uv)
            return (gamma, uv), None

        (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), None, length=J_alt)

        # outer loss
        loss = deconvolution_loss(Y, gamma_star, A_types, S_sigs, sigma)

        # L2 prior on w
        if lambda_w > 0:
            loss = loss + lambda_w * jnp.sum((w_levels - w_prior) ** 2)

        aux = ((gamma_star, uv_star), dict(alpha=alpha_vec, w=w_levels))
        return loss, aux

    @jax.jit
    def step(params, opt_state, gamma_uv, eps):
        (loss_value, (gamma_uv_new, stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, gamma_uv, eps
        )
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, gamma_uv_new, loss_value, stats

    return step

def learn_cladefgw_levels(
    C_feature, Y, C_T, C_S, a, b,
    A_types, S_sigs, sigma,
    W_assign, R_levels,
    *,
    # Sinkhorn / outer
    T_sinkhorn=50, J_alt=3, K_outer=200, lr=1e-2,
    # ε schedule
    eps0=0.2, eps_min=0.05, schedule="geometric", decay=0.95,
    # inits
    betas_alpha0=None, betas_w0=None,
    gamma0=None, uv0=None,
    # options
    normalize_masks=True, detach_gamma_in_C=False,
    lambda_w=0.0, w_prior=0.0
):
    n = a.shape[0]
    K = W_assign.shape[1]
    L_used = len(R_levels)

    if betas_alpha0 is None:
        betas_alpha0 = jnp.zeros(K)
    if betas_w0 is None:
        betas_w0 = jnp.zeros(L_used)  # softplus→~0 initially; or set to log(exp(1)-1) to start ~1
    if gamma0 is None:
        gamma0 = a[:, None] * b[None, :]
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    params = {
        "betas_alpha": jnp.array(betas_alpha0),
        "betas_w":     jnp.array(betas_w0),
    }
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    step = make_step_fn_cladefgw_levels(
        C_feature, Y, C_T, C_S, a, b,
        optimizer, A_types, S_sigs, sigma,
        W_assign, R_levels,
        T_sinkhorn=T_sinkhorn, J_alt=J_alt,
        normalize_masks=normalize_masks, detach_gamma_in_C=detach_gamma_in_C,
        lambda_w=lambda_w, w_prior=w_prior
    )

    def eps_k(k):
        if schedule == "geometric":
            return geom_eps(k, eps0, eps_min, decay)
        elif schedule == "cosine":
            return cosine_eps(k, K_outer, eps0, eps_min)
        else:
            return eps0

    gamma_uv = (gamma0, uv0)
    loss_hist, alpha_hist, w_hist, eps_hist = [], [], [], []

    with tqdm(range(K_outer)) as pbar:
        for k in pbar:
            eps = eps_k(k)
            params, opt_state, gamma_uv, loss_val, stats = step(params, opt_state, gamma_uv, eps)
            loss_hist.append(float(loss_val))
            alpha_hist.append(stats["alpha"])
            w_hist.append(stats["w"])
            eps_hist.append(eps)
            pbar.set_postfix({'loss': f"{float(loss_val):.6g}"})

    alpha_final = jax.nn.sigmoid(params["betas_alpha"])    # [K]
    w_final     = jax.nn.softplus(params["betas_w"])       # [L_used] ≥ 0
    gamma_final, _ = gamma_uv

    return (
        dict(alpha=alpha_final, w=w_final),
        jnp.stack(alpha_hist, axis=1),   # [K, K_outer]
        jnp.stack(w_hist, axis=1),       # [L_used, K_outer]
        jnp.array(eps_hist),             # [K_outer]
        jnp.array(loss_hist),            # [K_outer]
        gamma_final
    )

def _zero_diag(M):
    return M * (1.0 - jnp.eye(M.shape[0], dtype=M.dtype))

def _row_norm(M, eps=1e-12):
    return M / (M.sum(axis=1, keepdims=True) + eps)

def make_R_levels_from_Wwithin(
    W_within,
    *,
    include_root=True,
    include_leaves=False,
    zero_diagonal=True,
    normalize_rows=True,
    eps=1e-12
):
    """
    Build per-level residual masks R^(ℓ) from within-cluster mask(s) M^(ℓ) = W_within^(ℓ).

    Parameters
    ----------
    W_within : array [n,n] or list/tuple of arrays each [n,n]
        - If a single [n,n] matrix is given (your clade partition), this is treated as M^(1).
          We'll create:
            M^(0) = all-ones (off-diagonal if zero_diagonal=True),
            M^(1) = W_within,
            M^(2) = identity (or zeros if include_leaves=False).
        - If a sequence is given, it must be ordered from coarse→fine (level 0..L).
          Example: [M^(0), M^(1), ..., M^(L)] where M^(0) ≈ all-ones, M^(L) ≈ identity.

    include_root : bool
        If True (default), ensure a root level M^(0) is present (all-ones off-diagonal).
    include_leaves : bool
        If True, include the leaf level in residuals (R^(L) := M^(L)); typically False for off-diagonal usage.
    zero_diagonal : bool
        If True (default), zero the diagonal of M^(ℓ) and the residuals.
    normalize_rows : bool
        If True (default), row-normalize each residual R^(ℓ) so weights w_ℓ compare semantics, not mask size.
    eps : float
        Small constant for row normalization.

    Returns
    -------
    R_levels : list of [n,n] arrays
        Residual masks [R^(0), R^(1), ..., R^(D)] where D is the last included level.
        By default (include_leaves=False), R^(ℓ) = M^(ℓ) - M^(ℓ+1) for ℓ=0..L-1.
    M_levels : list of [n,n] arrays
        The processed within masks used internally (after optional root/leaf insertion & zeroing diag).

    Notes
    -----
    - Assumes monotonicity: M^(ℓ+1) ≤ M^(ℓ) elementwise (pairs that split never re-merge).
    - We clip to [0,1] after each operation to keep masks clean.
    """
    # Make a list of M-levels
    if isinstance(W_within, (list, tuple)):
        M_levels = [jnp.array(M, dtype=jnp.float32) for M in W_within]
    else:
        # Single level provided: synthesize root and (optionally) leaves
        M1 = jnp.array(W_within, dtype=jnp.float32)
        n = M1.shape[0]
        M0 = jnp.ones((n, n), dtype=jnp.float32)
        ML = jnp.eye(n, dtype=jnp.float32)
        M_levels = [M0, M1, ML]  # levels 0,1,2
        include_root = True  # already included

    # Ensure root present if requested
    n = M_levels[0].shape[0]
    if include_root:
        # If the first level isn't already "all ones", prepend one
        if not jnp.allclose(M_levels[0], jnp.ones((n, n), dtype=M_levels[0].dtype)):
            M_levels = [jnp.ones((n, n), dtype=M_levels[0].dtype)] + M_levels

    # Zero diagonal if requested
    if zero_diagonal:
        M_levels = [_zero_diag(M) for M in M_levels]

    # Clip to [0,1]
    M_levels = [jnp.clip(M, 0.0, 1.0) for M in M_levels]

    # Build residuals: R^(ℓ) = M^(ℓ) - M^(ℓ+1)
    R_levels = []
    for l in range(len(M_levels) - 1):
        R = jnp.clip(M_levels[l] - M_levels[l + 1], 0.0, 1.0)
        if zero_diagonal:
            R = _zero_diag(R)
        if normalize_rows:
            R = _row_norm(R, eps)
        R_levels.append(R)

    # Optionally include the leaf level residual R^(L) := M^(L)
    if include_leaves:
        ML = M_levels[-1]
        if zero_diagonal:
            ML = _zero_diag(ML)
        ML = jnp.clip(ML, 0.0, 1.0)
        if normalize_rows:
            ML = _row_norm(ML, eps)
        R_levels.append(ML)

    return R_levels, M_levels


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