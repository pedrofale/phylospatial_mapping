import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import optax
from tqdm import tqdm
import numpy as np
import ot

def coupling_from_binary(Z, a=None, b=None, iters=1000, eps=1e-12):
    n, m = Z.shape
    Z = jnp.asarray(Z, dtype=jnp.float32)
    if a is None:
        a = jnp.ones((n,), dtype=jnp.float32) / n
    if b is None:
        # Case A: derive b from assignments (equal split per row)
        k = jnp.maximum(Z.sum(axis=1), 1.0)              # [n]
        G = (Z / k[:, None]) * a[:, None]                # rows sum to a
        b_out = G.sum(axis=0)
        return G, a, b_out

    # Case B: project to target b via IPFP on support of Z
    # start v=ones; scale rows then cols
    u = jnp.ones((n,), dtype=jnp.float32)
    v = jnp.ones((m,), dtype=jnp.float32)
    for _ in range(iters):
        Zv = Z @ v + eps
        u = a / Zv
        ZTu = Z.T @ u + eps
        v = b / ZTu
    G = (u[:, None] * Z) * v[None, :]
    return G, a, b

def alpha_penalty(alpha, mF, mL, tau=1e-2, lam=1e-2):
    S = jnp.clip(mL / jnp.maximum(mF, 1e-12), 0.0, 1e6)
    phi = jax.lax.stop_gradient(jnp.maximum((tau - S) / tau, 0.0))
    return lam * (alpha**2) * phi

def _deconvolution_loss(Y, gamma, cell_type_assignments, cell_type_signatures, sigma):
    """
    Outer loss: scalar function of the transport plan gamma [n, m].
    """
    spot_cell_type_proportions = jnp.einsum('ji,ik->jk', gamma.T * gamma.shape[1], cell_type_assignments)
    spot_mean = jnp.einsum('jk,kg->jg', spot_cell_type_proportions, cell_type_signatures)
    return -tfp.distributions.Normal(spot_mean, sigma).log_prob(Y).sum()

def deconvolution_loss(Y, spot_scales, gamma, cell_type_assignments, cell_type_signatures):
    """
    Outer loss: scalar function of the transport plan gamma [n, m].
    """
    spot_cell_type_proportions = jnp.einsum('ji,ik->jk', gamma.T * gamma.shape[1], cell_type_assignments)
    spot_mean = jnp.einsum('jk,kg->jg', spot_cell_type_proportions, cell_type_signatures)
    ll = tfp.distributions.Poisson(spot_mean * spot_scales).log_prob(Y).sum()    
    lp = tfp.distributions.Gamma(1., 1.).log_prob(spot_scales).sum()
    return -(ll + lp) # loss

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
    return jnp.median(jnp.abs(M))# + eps

# def mad_abs(x, axis=None, eps=1e-12):
#     q50 = jax.lax.quantile(jnp.abs(x), 0.5, axis=axis, interpolation='nearest')
    # return jnp.clip(q50, eps, 1e12)

def build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, sF_ref, sL_ref, alpha_ref=0.5, zero_thresh=1e-6, *args, **kwargs):
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
    no_struct = (mL < .001 * mF)

    # 4) normalize and mix (if no_struct → drop L̂)
    F_hat = F_c / mF
    L_hat = jnp.where(no_struct, 0.0, L_c / mL)
    C = (1-alpha) * F_hat + alpha * L_hat
    Z = jnp.median(jnp.abs(F_hat)) + jnp.median(jnp.abs(L_hat))
    den = ((1-alpha)*jnp.median(jnp.abs(F_hat)) + alpha*jnp.median(jnp.abs(L_hat)))
    den = jnp.clip(den, 1e-2, 1e2) # if too clipped, large alpahs start modelling the epsilon. if not enough clipped, gets NaN gradients
    s = Z / den # problematic when L is small and alpha is large
    s = jax.lax.stop_gradient(s)    
    C = C * s
    return C

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
    *,
    gw_balance=True,          # step 3: balance CT/CS for L pieces
    match_L_to_F=True,        # step 4: make median(|L|) ~= median(|F|) at gamma_ref
    gamma_ref=None,           # if None uses a b^T
    eps=1e-6
):
    """
    Returns:
      C_feat_sc, C_tree_sc, C_space_sc, info
    Scales are computed once (outside gradients) and should be treated as constants downstream.
    """
    # ---- 1) Double-center each input w.r.t. (a,b) ----
    F_dc = _dc_rect(C_feature, a, b)   # [n,m]
    T_dc = _dc_square(C_tree,  a)      # [n,n]
    S_dc = _dc_square(C_space, b)      # [m,m]

    # ---- 2) Robust per-matrix scales (post-DC) ----
    sF = jnp.clip(jnp.median(jnp.abs(F_dc)), eps, 1e12)
    sT = jnp.clip(jnp.median(jnp.abs(T_dc)), eps, 1e12)
    sS = jnp.clip(jnp.median(jnp.abs(S_dc)), eps, 1e12)

    # Normalize originals so later computations use these rescaled inputs
    C_feat_sc = C_feature / sF
    C_tree_sc = C_tree   / sT
    C_space_sc= C_space  / sS

    info = {"sF_dc": sF, "sT_dc": sT, "sS_dc": sS}

    # Reference coupling for GW calibration
    if gamma_ref is None:
        gamma_ref = a[:, None] * b[None, :]

    # ---- 3) (optional) GW-balance CT/CS so L pieces have comparable scale ----
    if gw_balance:
        CT, CS = C_tree_sc, C_space_sc
        mT2 = jnp.clip(jnp.median(jnp.abs((CT * CT) @ a)), eps, 1e12)
        mS2 = jnp.clip(jnp.median(jnp.abs((CS * CS) @ b)), eps, 1e12)
        mTS = jnp.clip(jnp.median(jnp.abs(CT @ gamma_ref @ CS.T)), eps, 1e12)
        cT  = jnp.sqrt(mTS / mT2)
        cS  = jnp.sqrt(mTS / mS2)
        C_tree_sc  = CT * cT
        C_space_sc = CS * cS
        info.update({"cT": cT, "cS": cS, "mT2": mT2, "mS2": mS2, "mTS": mTS})

    # ---- 4) (optional) Match median(|L|) to median(|F|) at gamma_ref ----
    if match_L_to_F:
        # Build GW linearization L at gamma_ref with current (balanced) CT/CS
        def _dc_rect_outer(M, a, b):  # DC for n×m (same as F_dc)
            r = M @ b; c = M.T @ a; mu = jnp.dot(a, r)
            return M - r[:, None] - c[None, :] + mu

        # L = (CT^2 a) 1^T + 1 (CS^2 b)^T - 2 CT gamma_ref CS^T
        CT, CS = C_tree_sc, C_space_sc
        L_ref = ( (CT * CT) @ a )[:, None] + ( (CS * CS) @ b )[None, :] \
                - 2.0 * (CT @ gamma_ref @ CS.T)
        L_ref_dc = _dc_rect_outer(L_ref, a, b)
        mF = jnp.clip(jnp.median(jnp.abs(_dc_rect(C_feat_sc, a, b))), eps, 1e12)
        mL = jnp.clip(jnp.median(jnp.abs(L_ref_dc)), eps, 1e12)

        # Scale CT, CS together so median(|L|) ≈ median(|F|)
        lam = jnp.sqrt(mF / mL)
        C_tree_sc  = C_tree_sc  * lam
        C_space_sc = C_space_sc * lam
        info.update({"lam_match_L_to_F": lam, "mF_ref": mF, "mL_ref": mL})

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
    zero_rel=1e-10,
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
    F_c = _dc_rect(C_feature, a, b)  # [n,m]
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


# Safe real-domain Sinkhorn (keeps gradients)
def exp_smooth_clipped(x, M=60.0):
    return jnp.exp(M * jnp.tanh(x / M))

def sinkhorn_unrolled_safe(C, a, b, eps, T, uv0=None, tiny=1e-300, M=40.0):
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

    F_c = _dc_rect(C_feature, a, b)                              # [n,m]
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
        # alpha = jnp.clip(alpha, 0.0, 0.95)   # gradients are 0 when saturated

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
    print(f"Starting optimization with beta0={beta0}")
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

def make_step_fn_cladefgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, omega, Omega, sF_ref, sL_ref, T_sinkhorn=50, J_alt=3, train_mask=None, alpha_init=None):
    if train_mask is None:
        # default: train all
        # infer K from omega
        K = omega.shape[1]
        train_mask = jnp.ones((K,), dtype=jnp.float32)
    if alpha_init is None:
        alpha_init = jnp.full((K,), 0.5)

    def loss_fn(params, gamma_uv):
        gamma0, uv0 = gamma_uv
        betas = params['betas']
        spot_scales = jax.nn.softplus(params['s_raw']) + 1e-6 # ensure positive
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

        loss = deconvolution_loss(Y, spot_scales, gamma_star, cell_type_assignments, cell_type_signatures)
        return loss, ((gamma_star, uv_star), alphas, spot_scales)

    @jax.jit
    def step(params, opt_state, gamma_uv):
        (loss_value, (gamma_uv_new, alphas, spot_scales)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, gamma_uv)
        # zero gradients on fixed alphas
        g['betas'] = g['betas'] * train_mask
        updates, opt_state = optimizer.update(g, opt_state, params=params)
        # also zero updates on fixed alphas (belt & suspenders)
        updates['betas'] = updates['betas'] * train_mask        
        params = optax.apply_updates(params, updates)
        return params, opt_state, gamma_uv_new, loss_value, alphas, spot_scales

    return step



def make_step_fn_cladefgw_proximal(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, omega, Omega, sF_ref, sL_ref, 
                                    T_sinkhorn=50, J_alt=3, train_mask=None, alpha_init=None, tau=5.0, learn_scales=True):
    if train_mask is None:
        # default: train all
        # infer K from omega
        K = omega.shape[1]
        train_mask = jnp.ones((K,), dtype=jnp.float32)
    if alpha_init is None:
        alpha_init = jnp.full((K,), 0.5)

    if learn_scales:
        s_raw_mask = jnp.ones((Y.shape[0],1))
    else:
        s_raw_mask = jnp.zeros((Y.shape[0],1))

    def loss_fn(params, gamma_uv, gamma_ref):
        gamma0, uv0 = gamma_uv
        betas = params['betas']
        spot_scales = jax.nn.softplus(params['s_raw']) + 1e-6 # ensure positive
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

        data_loss = deconvolution_loss(Y, spot_scales, gamma_star, cell_type_assignments, cell_type_signatures)
        kl_loss = tau * gamma_skl(gamma_star, gamma_ref)
        loss = data_loss + kl_loss
        return loss, ((gamma_star, uv_star), alphas, spot_scales, kl_loss)

    @jax.jit
    def step(params, opt_state, gamma_uv, gamma_ref):
        (loss_value, (gamma_uv_new, alphas, spot_scales, kl_loss)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, gamma_uv, gamma_ref)
        # zero gradients on fixed alphas
        g['betas'] = g['betas'] * train_mask
        g['s_raw'] = g['s_raw'] * s_raw_mask
        updates, opt_state = optimizer.update(g, opt_state, params=params)
        # also zero updates on fixed alphas (belt & suspenders)
        updates['betas'] = updates['betas'] * train_mask        
        updates['s_raw'] = updates['s_raw'] * s_raw_mask
        params = optax.apply_updates(params, updates)
        return params, opt_state, gamma_uv_new, loss_value, alphas, spot_scales, kl_loss

    return step

def row_mad_abs(X):                     # [n,m] -> [n]
    return jnp.median(jnp.abs(X), axis=1) + 1e-12

def softplus_inv(x):
    return x + jnp.log(-jnp.expm1(-x))

def learn_alpha_gamma_cladefgw_proximal(
    C_feature, Y, C_tree, C_space, a, b,
    cell_type_assignments, cell_type_signatures,
    omega, Omega,
    eps=0.05, T_sinkhorn=50, J_alt=3,
    K_outer=200, lr=1e-2, alpha_init=None, train_mask=None,
    gamma0=None, uv0=None,
    gamma_ref=None,
    spot_scales=None,
    tau=5.0,
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

    if gamma_ref is None:
        gamma_ref = gamma0

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

    s_raw = jnp.ones((Y.shape[0],1))
    learn_scales = True
    if spot_scales is not None:
        s_raw = softplus_inv(spot_scales)
        learn_scales = False
    params = {"betas": betas, "s_raw": s_raw}
    opt_state = optimizer.init(params)

    step = make_step_fn_cladefgw_proximal(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, omega, Omega, sF_ref, sL_ref, T_sinkhorn, J_alt, train_mask, alpha_init, learn_scales=learn_scales, tau=tau)

    gamma_uv = (gamma0, uv0)
    loss_hist, alphas_hist, spot_scales_hist, kl_loss_hist = [], [], [], []

    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            params, opt_state, gamma_uv, loss_value, alphas, spot_scales, kl_loss = step(params, opt_state, gamma_uv, gamma_ref)
            loss_hist.append(float(loss_value))
            alphas_hist.append(alphas)
            spot_scales_hist.append(spot_scales)
            kl_loss_hist.append(float(kl_loss))
            pbar.set_postfix({'loss': float(loss_value), 'kl_loss': float(kl_loss)})

    alpha_final = jax.nn.sigmoid(params['betas'])
    # ensure returned alphas respect fixed ones
    alpha_final = jnp.where(train_mask > 0.5, alpha_final, alpha_init)
    return alpha_final, jnp.stack(alphas_hist, axis=1), jnp.array(loss_hist), gamma_uv[0], spot_scales, jnp.stack(spot_scales_hist, axis=1), jnp.array(kl_loss_hist)


def learn_alpha_gamma_cladefgw(
    C_feature, Y, C_tree, C_space, a, b,
    cell_type_assignments, cell_type_signatures,
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
    s_raw = jnp.ones((Y.shape[0],1))
    params = {"betas": betas, "s_raw": s_raw}
    opt_state = optimizer.init(params)

    step = make_step_fn_cladefgw(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, omega, Omega, sF_ref, sL_ref, T_sinkhorn, J_alt, train_mask, alpha_init)

    gamma_uv = (gamma0, uv0)
    loss_hist, alphas_hist, spot_scales_hist = [], [], []

    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            params, opt_state, gamma_uv, loss_value, alphas, spot_scales = step(params, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alphas_hist.append(alphas)
            spot_scales_hist.append(spot_scales)
            pbar.set_postfix({'loss': float(loss_value)})

    alpha_final = jax.nn.sigmoid(params['betas'])
    # ensure returned alphas respect fixed ones
    alpha_final = jnp.where(train_mask > 0.5, alpha_final, alpha_init)
    return alpha_final, jnp.stack(alphas_hist, axis=1), jnp.array(loss_hist), gamma_uv[0], spot_scales, jnp.stack(spot_scales_hist, axis=1)


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

def run_spotr(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix, 
            cell_type_assignments, cell_type_signatures, clade_columns=['clade_level0', 'clade_level1', 'clade_level2'], gamma_ref=None,
            eps=0.01, T_sinkhorn=100, J_alt=20, K_outer=100, lr=1e-2, clade_to_ignore='NA', alpha=0.5, tau=5.0, spot_scales_ref=None, max_exp_ratio=25.0):
    
    C_feature, C_tree, C_space, a, b = prepare_ot_inputs(ss_simulated_adata, spatial_simulated_adata, tree_distance_matrix, spatial_distance_matrix)

    # Initialize with features only OT
    gamma0, uv0 = sinkhorn_fgw(C_feature, C_tree, C_space, a, b, eps, 
                                T_sinkhorn=50, J_alt=1, alpha=0., gamma0=None, uv0=None)

    Y = jnp.asarray(spatial_simulated_adata.X)            
    n_cells = ss_simulated_adata.shape[0]

    given_gamma_ref = gamma_ref is not None
    given_spot_scales_ref = spot_scales_ref is not None
    tau_ref = tau

    depth_results = {}
    losses = []
    for i, clade_column in enumerate(clade_columns):
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
        for c, clade in enumerate(cell_clades):
            omega[c, clade_to_idx[clade]] = 1.0

        # Omega is (n_cells, n_cells), 1 if same clade, else 0. Now rescale so within each clade, values are 1/(size of clade).
        Omega = (omega @ omega.T).astype(np.float32)
        # row_sums = Omega.sum(axis=1, keepdims=True)
        # Omega = Omega / np.maximum(row_sums, 1e-8)
        # Omega = Omega * n_cells
        omega = jnp.array(omega)
        Omega = jnp.array(Omega)

        # Set eps based on the initial cost matrix
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
        init_C = build_cladefgw_cost(alpha_init, C_feature, C_tree, C_space, a, b, gamma0, omega, Omega, sF_ref, sL_ref)
        q = jnp.quantile(jnp.abs(init_C), 0.99)   
        eps = q / max_exp_ratio                            # target max|-C/eps| ≈ 25
        print(eps)

        if i == 0 and not given_gamma_ref:
            tau = 0.0
        alphas, alphas_hist, loss_hist, coupling, spot_scales, spot_scales_hist, kl_loss_hist = learn_alpha_gamma_cladefgw_proximal(C_feature, Y, C_tree, C_space, a, b,
                                            jnp.array(cell_type_assignments), jnp.array(cell_type_signatures),
                                            omega, Omega,
                                            eps=eps, T_sinkhorn=T_sinkhorn, J_alt=J_alt, 
                                            K_outer=K_outer, lr=lr, alpha_init=alpha_init, train_mask=train_mask, gamma0=gamma0, uv0=uv0, gamma_ref=gamma_ref, 
                                            tau=tau, spot_scales=spot_scales_ref)
        losses.append(loss_hist[-1])

        depth_results[clade_column] = {
            'alphas': dict(zip(clades, alphas)),
            'alphas_hist': alphas_hist,
            'loss_hist': loss_hist,
            'coupling': coupling,
            'spot_scales': spot_scales,
            'spot_scales_hist': spot_scales_hist,
            'kl_loss_hist': kl_loss_hist
        }

        if i == 0:
            if not given_gamma_ref:
                gamma_ref = coupling
            if not given_spot_scales_ref:
                spot_scales_ref = spot_scales
            tau = tau_ref
            gamma0 = coupling
            J_alt = 3
        
    best_coupling_level = clade_columns[np.argmin(losses)]

    return best_coupling_level, depth_results


def gamma_skl(G, Gref, eps=1e-12):
    """
    Symmetric KL between normalized couplings.
    """
    P = G / (jnp.sum(G) + eps)
    Q = Gref / (jnp.sum(Gref) + eps)
    kl_pq = jnp.sum(P * (jnp.log(P + eps) - jnp.log(Q + eps)))
    kl_qp = jnp.sum(Q * (jnp.log(Q + eps) - jnp.log(P + eps)))
    return 0.5 * (kl_pq + kl_qp)

def _mad_abs(M, eps=1e-12):
    return jnp.clip(jnp.median(jnp.abs(M)), eps, 1e12)


@jax.jit
def make_cladexfgw_scales(C_feature, C_tree, C_space, a, b, gamma_ref):
    """
    Compute fixed (stop-grad) scales:
      sF_ref : from DC(C_feature)
      sL_ref : from DC(L_global) with Ω = all-ones
    """
    n = C_feature.shape[0]
    J = jnp.ones((n, n), dtype=C_feature.dtype)

    F_c = _dc_rect(C_feature, a, b)
    # L_global = L^{Ω=J}
    CT2, CS2 = C_tree*C_tree, C_space*C_space
    Lg = (CT2 @ a)[:,None] + (CS2 @ b)[None,:] - 2.0 * (C_tree @ gamma_ref @ C_space.T)
    # same as compute_Lcladegw with Ω=J
    # Lg = compute_Lcladegw(C_tree, C_space, a, b, gamma_ref, J)

    Lg_c = _dc_rect(Lg, a, b)

    sF_ref = jax.lax.stop_gradient(_mad_abs(F_c))
    sL_ref = jax.lax.stop_gradient(_mad_abs(Lg_c))
    return sF_ref, sL_ref

@jax.jit
def build_cladexfgw_cost(
    alpha,                  # scalar in [0,1]: features vs structure
    alpha_cross,            # scalar ≥ 0       : weight for cross-clade structural piece
    alpha_clades,           # [K]  ≥ 0         : per-clade within weights (row-weighted via clade_omega)
    C_feature, C_tree, C_space,
    a, b,                   # histograms (sum=1)
    gamma,                  # [n,m] current coupling
    celltype_omega,         # [n,K] row→celltype (one-hot or soft; rows sum≈1)
    clade_omega,            # [n,K] row→clade (one-hot or soft; rows sum≈1)
    Omega,                  # [n,n] within-clade mask (block diagonal)
    # --- normalization knobs ---
    sF_ref,            # scalar reference scale for features (post-DC); if None, computed on the fly
    sL_ref,            # scalar reference scale for structure (post-DC), shared by cross+within
    s_min=1e-6, s_max=1e6,
    s_lo=5e-1, s_hi=2.0,
    rel_floor=5e-2
):
    """
    Returns:
      C : [n,m] normalized mixed cost suitable for Sinkhorn
      info : dict with diagnostics
    Notes:
      • Both structural pieces share a single structural scale sL_ref to keep
        additivity (L_global_hat = L_cross_hat + L_within_hat).
      • All scales/equalizer use stop_gradient to decouple from parameter learning.
    """
    n, m = C_feature.shape
    J = jnp.ones((n, n), dtype=C_feature.dtype)

    # --- 1) Build structural pieces (linear in mask), then double-center ---
    L_within = compute_Lcladegw(C_tree, C_space, a, b, gamma, Omega)         # [n,m]
    L_cross  = compute_Lcladegw(C_tree, C_space, a, b, gamma, (1 - Omega))    # [n,m]

    F_c  = _dc_rect(C_feature, a, b)
    Lw_c = _dc_rect(L_within,  a, b)
    Lx_c = _dc_rect(L_cross,   a, b)

    # --- 2) Reference scales (stop-grad) ---
    sF = jax.lax.stop_gradient(jnp.clip(sF_ref, 1e-6, 1e6))   # scalar
    sL = jax.lax.stop_gradient(jnp.clip(sL_ref, 1e-6, 1e6))   # scalar

    # --- 3) Normalize pieces to comparable units ---
    F_hat  = F_c  / sF
    Lw_hat = Lw_c / sL
    Lx_hat = Lx_c / sL

    # --- 4) Structural mix: cross + row-weighted within ---
    # row weights for within = (clade_omega @ alpha_clades)  → shape [n]
    alpha_within_rows = clade_omega @ alpha_clades           # [n]
    S_hat = (alpha_cross * Lx_hat) + (alpha_within_rows[:, None] * Lw_hat)  # [n,m]


    # 4) per-clade medians from row medians (stop-grad)
    mF_i = jnp.median(jnp.abs(F_hat), axis=1)                        # [n]
    mL_i = jnp.median(jnp.abs(S_hat), axis=1)                        # [n]
    cnt_k = jnp.maximum(clade_omega.sum(axis=0), 1e-8)                     # [K]
    mF_k = (clade_omega.T @ mF_i) / cnt_k
    mL_k = (clade_omega.T @ mL_i) / cnt_k
    mF_k = jax.lax.stop_gradient(jnp.clip(mF_k, s_min, s_max))
    mL_k = jax.lax.stop_gradient(jnp.clip(mL_k, s_min, s_max))

    # --- 5) Global mix with α (features vs structure) ---\
    C = (1.0 - celltype_omega@alpha)[:, None] * F_hat + (celltype_omega@alpha)[:, None] * S_hat                # [n,m]

    # --- 6) Optional scale equalizer (stop-grad) to keep |C| stable across α ---
    mLk_safe = jnp.maximum(mL_k, rel_floor * mF_k)                   # [K]
    den_k = (1.0 - alpha) * mF_k + alpha * mLk_safe      # [K]
    den_k = jnp.clip(den_k, s_min, s_max)
    s_k   = (mF_k + mL_k) / den_k                                    # [K]
    s_k   = jnp.clip(s_k, s_lo, s_hi)
    s_k   = jax.lax.stop_gradient(s_k)

    # 8) row equalization and return
    s_i = clade_omega @ s_k                                                # [n]
    C   = C * s_i[:, None]

    return C

def make_step_fn_cladexfgw(C_feature, true_coupling, C_tree, C_space, a, b, eps, optimizer, celltype_omega, clade_omega, Omega, sF_ref, sL_ref, T_sinkhorn=50, J_alt=3, train_mask=None, alpha_init=None):
    if train_mask is None:
        # default: train all
        train_mask = jnp.ones((celltype_omega.shape[1],), dtype=jnp.float32)
    if alpha_init is None:
        alpha_init = jnp.full((celltype_omega.shape[1],), 0.5)

    def loss_fn(params, gamma_uv):
        gamma0, uv0 = gamma_uv
        betas = params['betas'] # in general more than 1 because of NA clades
        alphas = jax.nn.sigmoid(betas)  # α ∈ (0,1)
        # alphas = jnp.clip(alphas, 0.0, 0.95)   # gradients are 0 when saturated
        alphas = jnp.where(train_mask > 0.5, alphas, jax.lax.stop_gradient(alpha_init))

        beta_cross = params['beta_cross']
        alpha_cross = jax.nn.sigmoid(beta_cross) #* 0. + 1. # DEBUG

        beta_clades = params['beta_clades']
        alpha_clades = jax.nn.sigmoid(beta_clades)

        # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
        def one_round(gamma, uv):
            C = build_cladexfgw_cost(alphas, alpha_cross, alpha_clades, C_feature, C_tree, C_space, a, b, gamma, celltype_omega, clade_omega, Omega, sF_ref, sL_ref)
            return sinkhorn_unrolled_safe(C, a, b, eps, T_sinkhorn, uv)

        # Unroll J_alt rounds with carry
        def body(carry, _):
            gamma, uv = carry
            gamma, uv = one_round(gamma, uv)
            return (gamma, uv), None

        (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)

        loss = gamma_skl(gamma_star, true_coupling)
        return loss, ((gamma_star, uv_star), alphas, alpha_cross, alpha_clades)

    @jax.jit
    def step(params, opt_state, gamma_uv):
        (loss_value, (gamma_uv_new, alphas, alpha_cross, alpha_clades)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, gamma_uv)
        # zero gradients on fixed alphas
        g['betas'] = g['betas'] * train_mask
        updates, opt_state = optimizer.update(g, opt_state, params=params)
        # also zero updates on fixed alphas (belt & suspenders)
        updates['betas'] = updates['betas'] * train_mask        
        params = optax.apply_updates(params, updates)
        return params, opt_state, gamma_uv_new, loss_value, alphas, alpha_cross, alpha_clades

    return step

def learn_alpha_gamma_cladexfgw(
    C_feature, true_coupling, C_tree, C_space, a, b,
    celltype_omega, clade_omega, Omega,
    eps=0.05, T_sinkhorn=50, J_alt=3,
    K_outer=200, lr=1e-2, alpha_init=None, alpha_clades_init=None, alpha_cross_init=None, train_mask=None,
    gamma0=None, uv0=None,
):
    
    K = clade_omega.shape[1]
    if alpha_init is None:
        alpha_init = jnp.full((celltype_omega.shape[1],), 0.5)

    if alpha_clades_init is None:
        alpha_clades_init = jnp.full((clade_omega.shape[1],), 0.5)

    if alpha_cross_init is None:
        alpha_cross_init = jnp.array([1.])

    if train_mask is None:
        train_mask = jnp.ones((celltype_omega.shape[1],), dtype=jnp.float32)

    n = a.shape[0]; m = b.shape[0]
    if gamma0 is None:
        # uniform feasible plan as a warm start
        gamma0 = (a[:, None] * b[None, :])
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    sF_ref, sL_ref = make_cladexfgw_scales(C_feature, C_tree, C_space, a, b, gamma0)

    optimizer = optax.adam(lr)
    betas = logit(alpha_init)
    beta_cross = logit(alpha_cross_init)
    beta_clades = logit(alpha_clades_init)
    params = {"betas": betas, "beta_cross": beta_cross, "beta_clades": beta_clades}
    opt_state = optimizer.init(params)

    step = make_step_fn_cladexfgw(C_feature, true_coupling, C_tree, C_space, a, b, eps, optimizer, celltype_omega, clade_omega, Omega, sF_ref, sL_ref, T_sinkhorn, J_alt, train_mask)

    gamma_uv = (gamma0, uv0)
    loss_hist, alphas_hist, alpha_cross_hist, alpha_clades_hist = [], [], [], []

    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            params, opt_state, gamma_uv, loss_value, alphas, alpha_cross, alpha_clades = step(params, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alphas_hist.append(alphas)
            alpha_cross_hist.append(alpha_cross)
            alpha_clades_hist.append(alpha_clades)
            pbar.set_postfix({'loss': float(loss_value)})

    alpha_final = jax.nn.sigmoid(params['betas'])
    # ensure returned alphas respect fixed ones
    alpha_final = jnp.where(train_mask > 0.5, alpha_final, alpha_init)
    return alpha_final, jnp.stack(alphas_hist, axis=1), jnp.array(loss_hist), gamma_uv[0], alpha_cross, alpha_clades, jnp.stack(alpha_cross_hist, axis=1), jnp.stack(alpha_clades_hist, axis=1)

def make_step_fn_cladexfgw_data(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, celltype_omega, clade_omega, Omega, sF_ref, sL_ref, T_sinkhorn=50, J_alt=3, train_mask=None, alpha_init=None, learn_scales=True):
    if train_mask is None:
        # default: train all
        train_mask = jnp.ones((celltype_omega.shape[1],), dtype=jnp.float32)
    if alpha_init is None:
        alpha_init = jnp.full((celltype_omega.shape[1],), 0.5)

    def loss_fn(params, gamma_uv):
        gamma0, uv0 = gamma_uv
        betas = params['betas'] # in general more than 1 because of NA clades
        alphas = jax.nn.sigmoid(betas)  # α ∈ (0,1)
        # alphas = jnp.clip(alphas, 0.0, 0.95)   # gradients are 0 when saturated
        alphas = jnp.where(train_mask > 0.5, alphas, jax.lax.stop_gradient(alpha_init))

        beta_cross = params['beta_cross']
        alpha_cross = jax.nn.sigmoid(beta_cross) #* 0. + 1. # DEBUG

        beta_clades = params['beta_clades']
        alpha_clades = jax.nn.sigmoid(beta_clades)

        # First alternating round uses gamma0 inside C; subsequent rounds are unrolled
        def one_round(gamma, uv):
            C = build_cladexfgw_cost(alphas, alpha_cross, alpha_clades, C_feature, C_tree, C_space, a, b, gamma, celltype_omega, clade_omega, Omega, sF_ref, sL_ref)
            return sinkhorn_unrolled_safe(C, a, b, eps, T_sinkhorn, uv)

        # Unroll J_alt rounds with carry
        def body(carry, _):
            gamma, uv = carry
            gamma, uv = one_round(gamma, uv)
            return (gamma, uv), None

        (gamma_star, uv_star), _ = jax.lax.scan(body, (gamma0, uv0), xs=None, length=J_alt)

        spot_scales = jax.nn.softplus(params['s_raw']) + 1e-6
        loss = deconvolution_loss(Y, spot_scales, gamma_star, cell_type_assignments, cell_type_signatures)
        return loss, ((gamma_star, uv_star), alphas, alpha_cross, alpha_clades, spot_scales)

    @jax.jit
    def step(params, opt_state, gamma_uv):
        (loss_value, (gamma_uv_new, alphas, alpha_cross, alpha_clades, spot_scales)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, gamma_uv)
        # zero gradients on fixed alphas
        g['betas'] = g['betas'] * train_mask
        g['s_raw'] = g['s_raw'] * learn_scales
        updates, opt_state = optimizer.update(g, opt_state, params=params)
        # also zero updates on fixed alphas (belt & suspenders)
        updates['betas'] = updates['betas'] * train_mask        
        params = optax.apply_updates(params, updates)
        return params, opt_state, gamma_uv_new, loss_value, alphas, alpha_cross, alpha_clades, spot_scales

    return step

def learn_alpha_gamma_cladexfgw_data(
    C_feature, Y, C_tree, C_space, a, b,
    celltype_omega, clade_omega, Omega,
    cell_type_assignments, cell_type_signatures,
    eps=0.05, T_sinkhorn=50, J_alt=3,
    K_outer=200, lr=1e-2, alpha_init=None, alpha_clades_init=None, alpha_cross_init=None, train_mask=None,
    gamma0=None, uv0=None,
    spot_scales=None,
):
    
    K = clade_omega.shape[1]
    if alpha_init is None:
        alpha_init = jnp.full((celltype_omega.shape[1],), 0.5)

    if alpha_clades_init is None:
        alpha_clades_init = jnp.full((clade_omega.shape[1],), 0.5)

    if alpha_cross_init is None:
        alpha_cross_init = jnp.array([1.])

    if train_mask is None:
        train_mask = jnp.ones((celltype_omega.shape[1],), dtype=jnp.float32)

    n = a.shape[0]; m = b.shape[0]
    if gamma0 is None:
        # uniform feasible plan as a warm start
        gamma0 = (a[:, None] * b[None, :])
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    sF_ref, sL_ref = make_cladexfgw_scales(C_feature, C_tree, C_space, a, b, gamma0)

    optimizer = optax.adam(lr)
    betas = logit(alpha_init)
    beta_cross = logit(alpha_cross_init)
    beta_clades = logit(alpha_clades_init)
    s_raw = jnp.ones((Y.shape[0],1))
    learn_scales = True
    if spot_scales is not None:
        s_raw = softplus_inv(spot_scales)
        learn_scales = False
    params = {"betas": betas, "beta_cross": beta_cross, "beta_clades": beta_clades, "s_raw": s_raw}
    opt_state = optimizer.init(params)

    step = make_step_fn_cladexfgw_data(C_feature, Y, C_tree, C_space, a, b, eps, optimizer, cell_type_assignments, cell_type_signatures, celltype_omega, clade_omega, Omega, sF_ref, sL_ref, T_sinkhorn, J_alt, train_mask, alpha_init, learn_scales)

    gamma_uv = (gamma0, uv0)
    loss_hist, alphas_hist, alpha_cross_hist, alpha_clades_hist, spot_scales_hist = [], [], [], [], []
    with tqdm(range(K_outer)) as pbar:
        for _ in pbar:
            params, opt_state, gamma_uv, loss_value, alphas, alpha_cross, alpha_clades, spot_scales = step(params, opt_state, gamma_uv)
            loss_hist.append(float(loss_value))
            alphas_hist.append(alphas)
            alpha_cross_hist.append(alpha_cross)
            alpha_clades_hist.append(alpha_clades)
            spot_scales_hist.append(spot_scales)
            pbar.set_postfix({'loss': float(loss_value)})

    alpha_final = jax.nn.sigmoid(params['betas'])
    # ensure returned alphas respect fixed ones
    alpha_final = jnp.where(train_mask > 0.5, alpha_final, alpha_init)
    return alpha_final, jnp.stack(alphas_hist, axis=1), jnp.array(loss_hist), gamma_uv[0], alpha_cross, alpha_clades, jnp.stack(alpha_cross_hist, axis=1), jnp.stack(alpha_clades_hist, axis=1), jnp.stack(spot_scales_hist, axis=1)

