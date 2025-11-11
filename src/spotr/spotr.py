import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import optax
from tqdm import tqdm
import numpy as np
import ot

def alpha_penalty(alpha, mF, mL, tau=1e-2, lam=1e-2):
    S = jnp.clip(mL / jnp.maximum(mF, 1e-12), 0.0, 1e6)
    phi = jax.lax.stop_gradient(jnp.maximum((tau - S) / tau, 0.0))
    return lam * (alpha**2) * phi

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
    CT2, CS2 = C_T * C_T, C_S * C_S
    # term2 = jnp.einsum('ip,pm,mq->iq', Omega, gamma, CS2)          # Ω γ (C_S^2)^T
    return ((Omega * CT2) @ a)[:,None] + Omega @ gamma @ CS2.T - 2.0 * ((Omega * C_T) @ gamma @ C_S.T )

def perclade_dc_L(L, a, b, omega):
    """
    L      : [n, m]  (your clade L before centering)
    a      : [n]     (row weights, sum=1)
    b      : [m]     (col weights, sum=1)
    omega  : [n, K]  (row->clade membership; one-hot or soft, rows sum≈1)

    Returns:
      L_pc  : [n, m]  per-clade double-centered L
    """
    n, m = L.shape
    # global row-mean term (same as usual dc)
    r = L @ b                      # [n]

    # clade-restricted row weights (normalize within each clade)
    a_k_num = omega * a[:, None]   # [n, K]
    a_k_den = jnp.clip(a_k_num.sum(axis=0, keepdims=True), 1e-12, None)  # [1, K]
    a_k = a_k_num / a_k_den        # [n, K], each column sums to 1

    # clade-specific column means: c_k[j] = (L^T a_k)_j
    c_k = L.T @ a_k                # [m, K]

    # broadcast to rows: for row i, pick its clade’s c_k
    C_rows = omega @ c_k.T         # [n, m]

    # clade baselines: mu_k = (a_k^T r) = a_k^T L b
    mu_k = (a_k.T @ r)             # [K]
    MU_rows = (omega @ mu_k)[:, None]  # [n, 1] -> broadcast over cols

    # per-clade double-centering
    L_pc = L - r[:, None] - C_rows + MU_rows
    return L_pc


def dc(M, a, b):
    m = jnp.mean(M)
    M = M - jnp.mean(M,axis=0)[None,:] - jnp.mean(M,axis=1)[:,None] + m
    return M

def mad_abs(M, eps=1e-12):
    return jnp.median(jnp.abs(M)) + eps

def build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, sF_ref, sL_ref, zero_thresh=1e-6, *args, **kwargs):
    """
    Return the FGW cost matrix C(alpha) with shape [n, m].
    Example (blend): C = (1 - alpha) * C_struct + alpha * C_feat
    """
    # center
    F_c = dc(C_feature, a, b)
    L_c = dc(compute_Lgw(C_tree, C_space, a, b, gamma), a, b)
    # fixed scales (computed once; pass in as scalars, no grad)
    # normalize
    mL = jnp.clip(jax.lax.stop_gradient(sL_ref), 1e-6, 1e6)
    mF = jnp.clip(jax.lax.stop_gradient(sF_ref), 1e-6, 1e6)
    # 3) detect “no structure” relative to features
    no_struct = (mL < .01 * mF)

    # 4) normalize and mix (if no_struct → drop L̂)
    F_hat = F_c / mF
    L_hat = jnp.where(no_struct, 0.0, L_c / mL)
    # L_c = L_c*0
    C = (1-alpha) * F_hat + alpha * L_hat
    Z = jnp.median(jnp.abs(F_hat)) + jnp.median(jnp.abs(L_hat))
    den = ((1-alpha)*jnp.median(jnp.abs(F_hat)) + alpha*jnp.median(jnp.abs(L_hat)))
    den = jnp.clip(den, 1e-2, 1e2) # if too clipped, large alpahs start modelling the epsilon. if not enough clipped, gets NaN gradients
    s = Z / den # problematic when L is small and alpha is large
    s = jax.lax.stop_gradient(s)    
    C = C * s
    return C

import jax.numpy as jnp

def _dc_rect(M, a, b):
    # double-centering for n×m with weights a (rows), b (cols)
    r = M @ b
    c = M.T @ a
    mu = jnp.dot(a, r)
    return M - r[:, None] - c[None, :] + mu

def _dc_square(M, w):
    # double-centering for n×n (or m×m) with weights w
    r = M @ w
    c = M.T @ w
    mu = jnp.dot(w, r)
    return M - r[:, None] - c[None, :] + mu

def preprocess_costs(
    C_feature, C_tree, C_space, a, b,
    mode: str = "per_matrix",   # "per_matrix" or "gw_balanced"
    eps: float = 1e-6
):
    """
    Returns:
      C_feat_sc, C_tree_sc, C_space_sc, info (dict of scales)
    Modes:
      - "per_matrix": DC each matrix, scale by median(|.|) of its DC.
      - "gw_balanced": same, then calibrate C_tree/C_space so the GW linearization
        terms have comparable magnitude (using gamma_ref = a b^T).
    """
    # 1) DC each matrix
    F_dc = _dc_rect(C_feature, a, b)        # [n,m]
    T_dc = _dc_square(C_tree, a)            # [n,n]
    S_dc = _dc_square(C_space, b)           # [m,m]

    # 2) robust per-matrix scales
    sF = jnp.clip(jnp.median(jnp.abs(F_dc)), eps, 1e12)
    sT = jnp.clip(jnp.median(jnp.abs(T_dc)), eps, 1e12)
    sS = jnp.clip(jnp.median(jnp.abs(S_dc)), eps, 1e12)

    # 3) normalize originals (so later L uses these rescaled inputs)
    C_feat_sc = C_feature / sF
    C_tree_sc = C_tree   / sT
    C_space_sc= C_space  / sS

    info = {"sF_dc": sF, "sT_dc": sT, "sS_dc": sS}

    if mode == "gw_balanced":
        # Calibrate structural parts for L: (CT^2 a), (CS^2 b), (CT γ CS^T)
        gamma_ref = a[:, None] * b[None, :]   # simple, stable reference
        CT = C_tree_sc
        CS = C_space_sc
        mT2 = jnp.clip(jnp.median(jnp.abs((CT*CT) @ a)), eps, 1e12)
        mS2 = jnp.clip(jnp.median(jnp.abs((CS*CS) @ b)), eps, 1e12)
        mTS = jnp.clip(jnp.median(jnp.abs(CT @ gamma_ref @ CS.T)), eps, 1e12)

        cT  = jnp.sqrt(mTS / mT2)
        cS  = jnp.sqrt(mTS / mS2)

        C_tree_sc  = CT * cT
        C_space_sc = CS * cS

        info.update({"cT": cT, "cS": cS, "mT2": mT2, "mS2": mS2, "mTS": mTS})

    return C_feat_sc, C_tree_sc, C_space_sc, info



def build_cladefgw_cost(
    alpha,                 # [K] per-clade alpha in (0,1); pass sigmoid(beta_k) if needed
    C_feature, C_tree, C_space,
    a, b, gamma,
    omega,                 # [n,K] row->clade membership (one-hot or soft; rows sum≈1)
    Omega,                 # [n,n] mask used in L^Ω
    sF_ref, sL_ref,        # [K] per-clade reference scales (fixed/EMA); no grad inside
    # --- stability knobs ---
    alpha_ref=None,        # [K] optional fixed/EMA reference alphas for equalizer (no grad); if None uses 0.5
    zero_rel=1e-1,
    s_min=1e-6, s_max=1e6,
    s_lo=5e-1, s_hi=2.0,
    rel_mid=1e-2, ksharp=6.0,
    rel_floor=5e-2
):
    """
    Requires helper funcs:
      dc(M,a,b), perclade_dc_L(L,a,b,omega),
      compute_Lcladegw(C_T,C_S,a,b,gamma,Omega)
    Returns: C [n,m]
    """
    n, m = C_feature.shape

    # 1) centered parts
    F_c = dc(C_feature, a, b)  # [n,m]
    L_c = perclade_dc_L(compute_Lcladegw(C_tree, C_space, a, b, gamma, Omega), a, b, omega)  # [n,m]

    # 2) per-clade base scales -> rows (stop-grad)
    sF_k = jnp.clip(jax.lax.stop_gradient(sF_ref), s_min, s_max)     # [K]
    sL_k = jnp.clip(jax.lax.stop_gradient(sL_ref), s_min, s_max)     # [K]
    sF_i = jnp.clip(omega @ sF_k, s_min, s_max)                      # [n]
    sL_i = jnp.clip(omega @ sL_k, s_min, s_max)                      # [n]

    # 3) normalize + no-structure mask
    no_struct_k = (sL_k < (zero_rel * sF_k))
    no_struct_i = (omega @ no_struct_k.astype(F_c.dtype)) > 0
    F_hat = F_c / sF_i[:, None]
    L_hat = jnp.where(no_struct_i[:, None], 0.0, L_c / sL_i[:, None])

    # 4) per-clade medians from row medians (stop-grad)
    mF_i = jnp.median(jnp.abs(F_hat), axis=1)                        # [n]
    mL_i = jnp.median(jnp.abs(L_hat), axis=1)                        # [n]
    cnt_k = jnp.maximum(omega.sum(axis=0), 1e-8)                     # [K]
    mF_k = (omega.T @ mF_i) / cnt_k
    mL_k = (omega.T @ mL_i) / cnt_k
    mF_k = jax.lax.stop_gradient(jnp.clip(mF_k, s_min, s_max))
    mL_k = jax.lax.stop_gradient(jnp.clip(mL_k, s_min, s_max))

    # 5) structure gate per clade; gated alpha used in MIX
    S_k = jnp.clip(mL_k / jnp.maximum(mF_k, 1e-12), 0.0, 1e6)
    g_k = jax.lax.stop_gradient(jax.nn.sigmoid(ksharp * (jnp.log(S_k + 1e-12) - jnp.log(rel_mid))))
    alpha_g_k = alpha * g_k                                          # [K]
    alpha_eff_i = jnp.clip(omega @ alpha_g_k, 0.0, 1.0)              # [n]

    # 6) content-only mix (no α-dependent scaling)
    C0 = (1.0 - alpha_eff_i)[:, None] * F_hat + alpha_eff_i[:, None] * L_hat  # [n,m]

    # 7) α-decoupled equalizer: use α_ref (constant/EMA) in denom (stop-grad)
    if alpha_ref is None:
        alpha_ref_k = jnp.full_like(alpha, 0.5)
    else:
        alpha_ref_k = alpha_ref
    alpha_ref_k = jax.lax.stop_gradient(jnp.clip(alpha_ref_k, 0.0, 1.0))      # [K]

    mLk_safe = jnp.maximum(mL_k, rel_floor * mF_k)                   # [K]
    den_k = (1.0 - alpha_ref_k) * mF_k + alpha_ref_k * mLk_safe      # [K]
    den_k = jnp.clip(den_k, s_min, s_max)
    s_k   = (mF_k + mL_k) / den_k                                    # [K]
    s_k   = jnp.clip(s_k, s_lo, s_hi)
    s_k   = jax.lax.stop_gradient(s_k)

    # 8) row equalization and return
    s_i = omega @ s_k                                                # [n]
    C   = C0 * s_i[:, None]
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

# Safe real-domain Sinkhorn (keeps gradients)
def exp_smooth_clipped(x, M=60.0):
    return jnp.exp(M * jnp.tanh(x / M))

def sinkhorn_unrolled_safe(C, a, b, eps, T, uv0=None, tiny=1e-300, M=60.0):
    K = exp_smooth_clipped(-C/eps, M=M) + tiny
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

    F_c = dc(C_feature,a,b)
    L_c = dc(compute_Lgw(C_tree,C_space,a,b,gamma0),a,b)
    sF_ref = mad_abs(F_c)                      # fixed
    sL_ref = mad_abs(L_c)  # at a reference gamma_ref
    
    # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
    def one_round(gamma, uv):
        C = build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, sF_ref, sL_ref)
        # jax.debug.print(
        #     "α={a:.3f} | C med={med:.3e} min={mn:.3e} max={mx:.3e}",
        #     a=alpha, med=jnp.median(jnp.abs(C)), mn=jnp.min(C), mx=jnp.max(C)
        # )
        return sinkhorn_unrolled_safe(C, a, b, eps, T_sinkhorn, uv)

    # Unroll J_alt rounds with carry
    def body(carry, _):
        gamma, uv = carry
        gamma, uv = one_round(gamma, uv)
        return (gamma, uv), None

    (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)
    return gamma_star, uv_star

def sinkhorn_cladefgw(C_feature, C_tree, C_space, a, b, eps, omega, Omega, T_sinkhorn=50, J_alt=3, alpha=0.5, gamma0=None, uv0=None):
    if gamma0 is None:
        gamma0 = (a[:, None] * b[None, :])
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    F_c = dc(C_feature, a, b)                              # [n,m]
    L_c_ref = perclade_dc_L(compute_Lcladegw(C_tree, C_space, a, b, gamma0, Omega), a, b, omega)
    
    # Row-wise robust magnitudes
    rF = row_mad_abs(F_c)                                  # [n]
    rL = row_mad_abs(L_c_ref)                              # [n]

    # Aggregate to clades with omega (one-hot or soft), using weighted mean
    cnt_k = jnp.maximum(omega.sum(axis=0), 1e-8)           # [K]
    sF_ref_k = (omega.T @ rF) / cnt_k                      # [K]
    sL_ref_k = (omega.T @ rL) / cnt_k                      # [K]

    # Clamp
    smin, smax = (1e-6, 1e6)
    sF_ref_k = jnp.clip(sF_ref_k, smin, smax)
    sL_ref_k = jnp.clip(sL_ref_k, smin, smax)

    sF_ref = sF_ref_k
    sL_ref = sL_ref_k

    # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
    def one_round(gamma, uv):
        C = build_cladefgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, omega, Omega, sF_ref, sL_ref)
        # jax.debug.print("α_min/max={:.2f}/{:.2f} | med|C|={:.2e} max|-C/ε|={:.1f}",
        #                 jnp.min(alpha), jnp.max(alpha),
        #                 jnp.median(jnp.abs(C)), jnp.max(jnp.abs(-C/eps)))
        return sinkhorn_unrolled_safe(C, a, b, eps, T_sinkhorn, uv)

    # Unroll J_alt rounds with carry
    def body(carry, _):
        gamma, uv = carry
        gamma, uv = one_round(gamma, uv)
        return (gamma, uv), None

    (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)
    return gamma_star, uv_star    

def make_step_fn_fgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, sF_ref, sL_ref, T_sinkhorn=50, J_alt=3):
    def loss_fn(beta, gamma_uv):
        gamma0, uv0 = gamma_uv
        alpha = jax.nn.sigmoid(beta)  # α ∈ (0,1)        
        alpha = jnp.clip(alpha, 0.0, 0.95)   # gradients are 0 when saturated

        # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
        def one_round(gamma, uv):
            # gamma_sg = jax.lax.stop_gradient(gamma)
            C = build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, sF_ref, sL_ref)
            return sinkhorn_unrolled_safe(C, a, b, eps, T_sinkhorn, uv)

        # Unroll J_alt rounds with carry
        def body(carry, _):
            gamma, uv = carry
            gamma, uv = one_round(gamma, uv)
            return (gamma, uv), None

        (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)

        loss = deconvolution_loss(Y, gamma_star, cell_type_assignments, cell_type_signatures, sigma)
        loss += alpha_penalty(alpha, sF_ref, sL_ref)
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

    F_c = dc(C_feature,a,b)
    L_c = dc(compute_Lgw(C_tree,C_space,a,b,gamma0),a,b)
    sF_ref = mad_abs(F_c)                      # fixed
    sL_ref = mad_abs(L_c)  # at a reference gamma_ref

    optimizer = optax.adam(lr)
    beta = jnp.array(beta0)
    opt_state = optimizer.init(beta)

    step = make_step_fn_fgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, sF_ref, sL_ref, T_sinkhorn, J_alt)

    gamma_uv = (gamma0, uv0)
    loss_hist, alpha_hist = [], []
    gamma_hist = []
    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            beta, opt_state, gamma_uv, loss_value, alpha = step(beta, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alpha_hist.append(float(alpha))
            gamma_hist.append(gamma_uv[0])
            pbar.set_postfix({'loss': f"{float(loss_value):.6g}"})

    alpha_final = jax.nn.sigmoid(beta)
    return float(alpha_final), jnp.array(alpha_hist), jnp.array(loss_hist), gamma_uv[0], jnp.array(gamma_hist)

def logit(x, eps=1e-6):
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)

def make_step_fn_cladefgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, omega, Omega, sF_ref, sL_ref, T_sinkhorn=50, J_alt=3, train_mask=None, alpha_init=None):
    if train_mask is None:
        # default: train all
        # infer K from omega
        K = omega.shape[1]
        train_mask = jnp.ones((K,), dtype=jnp.float32)
    if alpha_init is None:
        alpha_init = jnp.full((K,), 0.5)

    def loss_fn(betas, gamma_uv):
        gamma0, uv0 = gamma_uv
        alphas = jax.nn.sigmoid(betas)  # α ∈ (0,1)
        alphas = jnp.clip(alphas, 0.0, 0.95)   # gradients are 0 when saturated
        alphas = jnp.where(train_mask > 0.5, alphas, jax.lax.stop_gradient(alpha_init))

        # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
        def one_round(gamma, uv):
            C = build_cladefgw_cost(alphas, C_feature, C_tree, C_space, a, b, gamma, omega, Omega, sF_ref, sL_ref)
            return sinkhorn_unrolled_safe(C, a, b, eps, T_sinkhorn, uv)

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
        # zero gradients on fixed alphas
        g = g * train_mask
        updates, opt_state = optimizer.update(g, opt_state, params=betas)
        # also zero updates on fixed alphas (belt & suspenders)
        updates = updates * train_mask        
        betas = optax.apply_updates(betas, updates)
        return betas, opt_state, gamma_uv_new, loss_value, alphas

    return step

def row_mad_abs(X):                     # [n,m] -> [n]
    return jnp.median(jnp.abs(X), axis=1) + 1e-12

def learn_alpha_gamma_cladefgw(
    C_feature, Y, C_tree, C_space, a, b,
    cell_type_assignments, cell_type_signatures, sigma,
    omega, Omega,
    eps=0.05, T_sinkhorn=50, J_alt=3,
    K_outer=200, lr=1e-2, alpha_init=None, train_mask=None,
    gamma0=None, uv0=None,
):
    K = omega.shape[1]
    if alpha_init is None:
        alpha_init = jnp.full((K,), 0.5)

    if train_mask is None:
        train_mask = jnp.ones((K,), dtype=jnp.float32)

    n = a.shape[0]; m = b.shape[0]
    if gamma0 is None:
        # uniform feasible plan as a warm start
        gamma0 = (a[:, None] * b[None, :])
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    F_c = dc(C_feature, a, b)                              # [n,m]
    L_c_ref = perclade_dc_L(compute_Lcladegw(C_tree, C_space, a, b, gamma0, Omega), a, b, omega)
    
    # Row-wise robust magnitudes
    rF = row_mad_abs(F_c)                                  # [n]
    rL = row_mad_abs(L_c_ref)                              # [n]

    # Aggregate to clades with omega (one-hot or soft), using weighted mean
    cnt_k = jnp.maximum(omega.sum(axis=0), 1e-8)           # [K]
    sF_ref_k = (omega.T @ rF) / cnt_k                      # [K]
    sL_ref_k = (omega.T @ rL) / cnt_k                      # [K]

    # Clamp
    smin, smax = (1e-6, 1e6)
    sF_ref_k = jnp.clip(sF_ref_k, smin, smax)
    sL_ref_k = jnp.clip(sL_ref_k, smin, smax)

    sF_ref = sF_ref_k
    sL_ref = sL_ref_k

    optimizer = optax.adam(lr)
    betas = logit(alpha_init)
    opt_state = optimizer.init(betas)

    step = make_step_fn_cladefgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, sigma, omega, Omega, sF_ref, sL_ref, T_sinkhorn, J_alt, train_mask, alpha_init)

    gamma_uv = (gamma0, uv0)
    loss_hist, alphas_hist = [], []

    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            betas, opt_state, gamma_uv, loss_value, alphas = step(betas, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alphas_hist.append(alphas)
            pbar.set_postfix({'loss': float(loss_value)})

    alpha_final = jax.nn.sigmoid(betas)
    # ensure returned alphas respect fixed ones
    alpha_final = jnp.where(train_mask > 0.5, alpha_final, alpha_init)
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

    # Do we need this?
    # C_feat_sc, C_tree_sc, C_space_sc, info = preprocess_costs(
    #     C_feature, C_tree, C_space, a, b, mode="gw_balanced"
    # )

    return C_feature, C_tree, C_space, a, b

def run_spotr_single(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix, 
            cell_type_assignments, cell_type_signatures, clade_column='clade_level2', sigma=0.01,
            eps=0.01, T_sinkhorn=100, J_alt=20, K_outer=100, lr=1e-2, clade_to_ignore='NA', alpha=0.5):
    
    C_feature, C_tree, C_space, a, b = prepare_ot_inputs(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix)

    Y = jnp.asarray(spatial_simulated_adata.X)            
    n_cells = ss_simulated_adata.shape[0]
    cell_clades = ss_simulated_adata.obs[clade_column].values
    n_clades = len(ss_simulated_adata.obs[clade_column].unique()) # includes clades that we shouldn't learn alphas for

    alpha_init = np.full((n_clades,), alpha)
    train_mask = np.ones((n_clades,), dtype=np.float32) # train all alphas by default
    deactivated_clades = np.where(cell_clades == clade_to_ignore)[0]
    train_mask[deactivated_clades] = 0.0
    alpha_init[deactivated_clades] = 0.0
    train_mask = jnp.array(train_mask)
    alpha_init = jnp.array(alpha_init)

    # omega: (n_cells, n_clades), one-hot encoding each cell's clade
    clade_to_idx = {clade: i for i, clade in enumerate(np.unique(cell_clades))}
    omega = np.zeros((n_cells, n_clades), dtype=np.float32)
    for i, clade in enumerate(cell_clades):
        omega[i, clade_to_idx[clade]] = 1.0

    # Omega is (n_cells, n_cells), 1 if same clade, else 0. Now rescale so within each clade, values are 1/(size of clade).
    Omega = (omega @ omega.T).astype(np.float32)
    row_sums = Omega.sum(axis=1, keepdims=True)
    Omega = Omega / np.maximum(row_sums, 1e-8)
    Omega = Omega * n_cells
    omega = jnp.array(omega)
    Omega = jnp.array(Omega)

    # Initialize with features only OT
    gamma0, uv0 = sinkhorn_fgw(C_feature, C_tree, C_space, a, b, .01, 
                                T_sinkhorn=50, J_alt=1, alpha=0., gamma0=None, uv0=None)

    alphas, alphas_hist, loss_hist, coupling = learn_alpha_gamma_cladefgw(C_feature, Y, C_tree, C_space, a, b,
                                        jnp.array(cell_type_assignments), jnp.array(cell_type_signatures), sigma,
                                        omega, Omega,
                                        eps=eps, T_sinkhorn=T_sinkhorn, J_alt=J_alt, 
                                        K_outer=K_outer, lr=lr, alpha_init=alpha_init, train_mask=train_mask, gamma0=gamma0, uv0=uv0)    

    return alphas, alphas_hist, loss_hist, coupling


def run_spotr(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix, 
            cell_type_assignments, cell_type_signatures, clade_columns=['clade_level0', 'clade_level1', 'clade_level2'], sigma=0.01,
            eps=0.01, T_sinkhorn=100, J_alt=20, K_outer=100, lr=1e-2, clade_to_ignore='NA', alpha=0.5):
    
    C_feature, C_tree, C_space, a, b = prepare_ot_inputs(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix)

    # Initialize with features only OT
    gamma0, uv0 = sinkhorn_fgw(C_feature, C_tree, C_space, a, b, eps, 
                                T_sinkhorn=50, J_alt=1, alpha=0., gamma0=None, uv0=None)

    Y = jnp.asarray(spatial_simulated_adata.X)            
    n_cells = ss_simulated_adata.shape[0]

    depth_results = {}
    losses = []
    for clade_column in clade_columns:
        cell_clades = ss_simulated_adata.obs[clade_column].values
        clades = np.unique(cell_clades)
        n_clades = len(clades) # includes clades that we shouldn't learn alphas for

        # Turn off alpha learning for NA clades
        alpha_init = np.full((n_clades,), alpha)
        train_mask = np.ones((n_clades,), dtype=np.float32) # train all alphas by default
        deactivated_clades = np.where(cell_clades == clade_to_ignore)[0]
        train_mask[deactivated_clades] = 0.0
        alpha_init[deactivated_clades] = 0.0
        train_mask = jnp.array(train_mask)
        alpha_init = jnp.array(alpha_init)

        # omega: (n_cells, n_clades), one-hot encoding each cell's clade
        clade_to_idx = {clade: i for i, clade in enumerate(clades)}
        omega = np.zeros((n_cells, n_clades), dtype=np.float32)
        for i, clade in enumerate(cell_clades):
            omega[i, clade_to_idx[clade]] = 1.0

        # Omega is (n_cells, n_cells), 1 if same clade, else 0. Now rescale so within each clade, values are 1/(size of clade).
        Omega = (omega @ omega.T).astype(np.float32)
        row_sums = Omega.sum(axis=1, keepdims=True)
        Omega = Omega / np.maximum(row_sums, 1e-8)
        Omega = Omega * n_cells
        omega = jnp.array(omega)
        Omega = jnp.array(Omega)

        alphas, alphas_hist, loss_hist, coupling = learn_alpha_gamma_cladefgw(C_feature, Y, C_tree, C_space, a, b,
                                            jnp.array(cell_type_assignments), jnp.array(cell_type_signatures), sigma,
                                            omega, Omega,
                                            eps=eps, T_sinkhorn=T_sinkhorn, J_alt=J_alt, 
                                            K_outer=K_outer, lr=lr, alpha_init=alpha_init, train_mask=train_mask, gamma0=gamma0, uv0=uv0)    
        losses.append(loss_hist[-1])

        depth_results[clade_column] = {
            'alphas': dict(zip(clades, alphas)),
            'alphas_hist': alphas_hist,
            'loss_hist': loss_hist,
            'coupling': coupling
        }

    best_coupling_level = clade_columns[np.argmin(losses)]

    return best_coupling_level, depth_results