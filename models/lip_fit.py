"""lip_fit -- the LIP (M, V) moment-matching distillation in the AGaLU
feature space, vendored from paper/scripts/lib/lip.py (the exp_1 canonical
build) and extended in ONE way: the fit target is a WEIGHTED point set, so a
stored memory and a new task stack by the moment mixture rule

    M_target = w_old * M(memory) + w_new * M(new rows),      w = N_part/N_seen

(moments are linear in the distribution, so the convex combination of the
parts' moments IS the mixture's moments; realized here by giving every target
row a weight, sum w = 1).  No VILA imports: `python models/lip_fit.py` runs
the fp64 CPU selftest proving the math.

THE OBJECT (unchanged from exp_1).  A distilled set is m natural-scale pairs
(b_a, y_a) with the UNIFORM WEIGHT 1/m applied explicitly -- a plain dataset:

    Mhat = (1/m) sum_a Phi(b_a) Phi(b_a)'      M = sum_n w_n Phi(z_n) Phi(z_n)'
    Vhat = (1/m) sum_a Phi(b_a) y_a'           V = sum_n w_n Phi(z_n) y_n'

with Phi(x) = (mask(Bg x) (x) x)/sqrt(k), so <Phi, Phi'> is the agalu kernel
k(a, b) = (a.b)(g_a.g_b)/k.  The fit minimizes the ABSOLUTE objective

    J = lam * J_M + (1 - lam) * J_V,
    J_M = ||Mhat - M||_F^2,   J_V = ||Vhat - V||_F^2,

in kernel form (the order-4 tensors are never materialized), by plain Adam at
a constant RELATIVE lr (lr * rms(group at init)), full-batch: the target here
is at most m + one task's rows, so the exact gradient is affordable and the
exp_1 minibatch estimator is not needed.

TIED GATES (mandatory, exp_1 convention): a distilled datum is (x, y); its
gate is always re-derived as mask(Bg x).  The objective is discontinuous in
the atom; Adam sees the frozen-mask a.e. gradient.  Gate drift from the fit's
init masks is logged as g_flip_frac.

fp DISCIPLINE: fit steps run fp32 (mask Grams are exact in fp32: entries 0/1,
integer inner products <= k << 2^24); every REPORTED J is an exact fp64 eval
of the stored state, and a negative squared error aborts (PrecisionError).
"""
import math
import time

import torch

DIVERGE_ABS = 1e12       # |loss| above this (or non-finite) aborts the fit


class PrecisionError(RuntimeError):
    """A reported squared error came out negative: a precision bug, not data."""


def masks(X, Bg):
    """g = 1{Bg x > 0}, the frozen-gate mask, as a bool tensor.  Computed at
    the input's own dtype: only the SIGN matters."""
    return X @ Bg.T.to(X.dtype) > 0


# Wave-10 gate-activation suite: the step gate generalizes to any elementwise
# act on the preactivation Bg x.  Everything downstream (kernel, J, fit) is
# act-agnostic; step stays the bit-identical default (bool gates).
GATE_ACTS = {
    "step": lambda t: (t > 0).to(t.dtype),
    "sign": torch.sign,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "identity": lambda t: t,
    "gauss": lambda t: torch.exp(-t * t),
    "exp": torch.exp,
    "expm1": torch.expm1,
    "softplus": torch.nn.functional.softplus,
    # wave 81: input-anchored relu gating relu(Bg x) (o) (W1 x) -- the
    # magnitude-sensitive cousin of "step"; stays in the anchored family
    # (linear in W1 for fixed gates), so lip fits price it like any gate
    "relu": torch.nn.functional.relu,
}


def gates(X, Bg, act="step", scale=1.0, alpha=1.0):
    """g = act(alpha * Bg x)/scale.  act='step' returns the classic bool mask
    (alpha/scale ignored: step is scale-invariant and never rescaled); other
    acts return real-valued gates at X's dtype.  alpha is the preactivation
    gain (wave 11): it changes the gate SHAPE regime (small -> constant-like,
    large -> step-like)."""
    if act == "step":
        return masks(X, Bg)
    return GATE_ACTS[act](alpha * (X @ Bg.T.to(X.dtype))) / scale


def _gram(GA, GB, out_dtype):
    """Gate Gram.  Bool (step) gates: computed fp32 -- exact, 0/1 entries,
    integer inner products <= k << 2^24 -- then cast.  Real-valued gates:
    plain matmul at the caller's dtype (fp64 on the exact path)."""
    if GA.dtype == torch.bool:
        return (GA.float() @ GB.float().T).to(out_dtype)
    return (GA.to(out_dtype) @ GB.to(out_dtype).T)


def k_agalu(A, B, GA, GB):
    """The AGaLU kernel (a.b)(g_a.g_b)/k at A's dtype (pass fp64 for exact)."""
    kk = GA.shape[1]
    return (A @ B.to(A.dtype).T) * (_gram(GA, GB, A.dtype) / float(kk))


def _nonneg(J, parts, where, tol=1e-10):
    """A raw squared-Frobenius error cannot be negative.  J is a difference of
    cancelling terms, so the tripwire scales with the largest cancelled term."""
    sc = max(abs(float(x)) for x in parts)
    if J < -tol * sc:
        raise PrecisionError(f"[{where}] J = {J:.6e} < -{tol:g}*{sc:.6e}: the "
                             f"eval path is not pure fp64")
    return J


# ================================================================ target
def make_target(parts, Bg, gate=("step", 1.0)):
    """Weighted fit target from (Z, Y, w_total) parts: rows fp64, per-row
    weights w_part/n_part (sum over all parts = 1), gates derived once (the
    target rows never move).  parts = [(Z [n,d], Y [n,C], w_total), ...].
    gate = (act, scale[, alpha]): step -> bool masks (waves 1-9
    bit-identical), else real-valued fp64 gates act(alpha Bg z)/scale."""
    Zs, Ys, ws = [], [], []
    for Z, Y, w in parts:
        n = Z.shape[0]
        Zs.append(Z.double())
        Ys.append(Y.double())
        ws.append(torch.full((n,), float(w) / n, dtype=torch.float64,
                             device=Z.device))
    Z, Y, w = torch.cat(Zs), torch.cat(Ys), torch.cat(ws)
    assert abs(float(w.sum().item()) - 1.0) < 1e-8, "target weights must sum to 1"
    return dict(Z=Z, Y=Y, w=w, G=gates(Z, Bg.double(), *gate), gate=gate)


def wmoments(TAR, chunk=8192):
    """S_M = ||M||_F^2, S_V = ||V||_F^2 of the weighted target, exact fp64:
    S_M = sum_ab w_a w_b k(a,b)^2,  S_V = sum_ab w_a w_b (Y_a.Y_b) k(a,b).
    Row-chunked so a joint 50k-row target never materializes the n x n Gram;
    chunking only reorders exact fp64 partial sums."""
    Z, Y, w, G = TAR["Z"], TAR["Y"], TAR["w"], TAR["G"]
    S_M = S_V = 0.0
    for i in range(0, Z.shape[0], chunk):
        K = k_agalu(Z[i:i + chunk], Z, G[i:i + chunk], G)
        W = w[i:i + chunk, None] * w[None, :]
        S_M += float((W * K * K).sum().item())
        S_V += float((W * (Y[i:i + chunk] @ Y.T) * K).sum().item())
    return dict(S_M=S_M, S_V=S_V)


# ================================================================ objective
def _J_terms(B, Yat, G_B, Z, Y, w, G, S_M, S_V, lam):
    """J = lam*J_M + (1-lam)*J_V of uniform-1/m atoms vs the weighted target,
    at the inputs' common dtype (fp64 -> exact, fp32 -> the fit's gradient
    path; the S constants are kept in so the value IS J)."""
    m, kk = B.shape[0], G_B.shape[1]
    K1 = (B @ B.T) * (_gram(G_B, G_B, B.dtype) / float(kk))
    PL = (B @ Z.T) * (_gram(G_B, G, B.dtype) / float(kk))
    J_M = ((K1 * K1).sum() / m ** 2
           - 2.0 * ((PL * PL) @ w).sum() / m + S_M)
    h = PL @ (w[:, None] * Y)
    J_V = (torch.einsum("ac,ab,bc->", Yat, K1, Yat) / m ** 2
           - 2.0 * (Yat * h).sum() / m + S_V)
    return lam * J_M + (1.0 - lam) * J_V, J_M, J_V


def _J_terms_split(B, Yat, G_B, iL, iR, Z, Y, w, G, S_M, S_V, lam):
    """Atom-subsampled J with the quadratic term estimated on L x R index
    sets (iL=None -> the full left side, mode 'half'; two INDEPENDENT draws
    -> mode 'ind') while the target cross term stays FULL-atom.  Unbiased
    for J and its gradient: with one side full (or the sides independent)
    the estimator is linear in each sampled uniform measure, so E commutes
    -- unlike the naive subset-everywhere mode, which is quadratic in one
    sampled measure (diagonal/variance inflation -> the collapse pressure).
    Cost: the m x m atom Gram becomes mL x mR."""
    m, kk = B.shape[0], G_B.shape[1]
    BL, GL, YL = (B, G_B, Yat) if iL is None else (B[iL], G_B[iL], Yat[iL])
    BR, GR, YR = B[iR], G_B[iR], Yat[iR]
    mL, mR = BL.shape[0], BR.shape[0]
    KLR = (BL @ BR.T) * (_gram(GL, GR, B.dtype) / float(kk))
    PL = (B @ Z.T) * (_gram(G_B, G, B.dtype) / float(kk))
    J_M = ((KLR * KLR).sum() / (mL * mR)
           - 2.0 * ((PL * PL) @ w).sum() / m + S_M)
    h = PL @ (w[:, None] * Y)
    J_V = (torch.einsum("ac,ab,bc->", YL, KLR, YR) / (mL * mR)
           - 2.0 * (Yat * h).sum() / m + S_V)
    return lam * J_M + (1.0 - lam) * J_V


def exact_state(B, Yat, TAR, mom, lam, Bg, G=None, chunk=0):
    """THE exact fp64 (M, V) evaluator: every reported J goes through
    literally this expression.  Gates derived from the atoms (tied), with
    the target's own gate activation; G (wave 36) = UNTIED per-atom gates,
    used verbatim instead of deriving from the atoms.  chunk > 0 (wave 44):
    the m x n cross terms accumulate over target-row chunks -- pure fp64
    re-ordering, bitwise-negligible, and the only way m=20000 vs a 50k
    target fits in memory (the unchunked PL alone is 8 GB)."""
    Bd, Yd = B.detach().double(), Yat.detach().double()
    G_B = (G.detach().double() if G is not None
           else gates(Bd, Bg.double(), *TAR.get("gate", ("step", 1.0))))
    if chunk and (TAR["Z"].shape[0] > chunk or Bd.shape[0] > chunk):
        Z, Y, w, Gt = TAR["Z"], TAR["Y"], TAR["w"], TAR["G"]
        m, kk = Bd.shape[0], G_B.shape[1]
        # wave 45: the atom-kernel scalars accumulate over ROW BLOCKS of K1
        # -- the full m x m fp64 kernel (20 GB at m=50000, x3 temporaries)
        # is never materialized; pure fp64 re-ordering like the cross terms
        q_M = torch.zeros((), dtype=torch.float64, device=Bd.device)
        q_V = torch.zeros((), dtype=torch.float64, device=Bd.device)
        for a in range(0, m, chunk):
            K1a = ((Bd[a:a + chunk] @ Bd.T)
                   * (_gram(G_B[a:a + chunk], G_B, Bd.dtype) / float(kk)))
            q_M = q_M + (K1a * K1a).sum()
            q_V = q_V + torch.einsum("ac,ab,bc->", Yd[a:a + chunk], K1a, Yd)
        t_M = torch.zeros((), dtype=torch.float64, device=Bd.device)
        h = torch.zeros_like(Yd)
        for i in range(0, Z.shape[0], chunk):
            PL = ((Bd @ Z[i:i + chunk].T)
                  * (_gram(G_B, Gt[i:i + chunk], Bd.dtype) / float(kk)))
            t_M = t_M + ((PL * PL) @ w[i:i + chunk]).sum()
            h = h + PL @ (w[i:i + chunk, None] * Y[i:i + chunk])
        J_M = (q_M / m ** 2 - 2.0 * t_M / m + mom["S_M"])
        J_V = (q_V / m ** 2 - 2.0 * (Yd * h).sum() / m + mom["S_V"])
    else:
        _, J_M, J_V = _J_terms(Bd, Yd, G_B, TAR["Z"], TAR["Y"], TAR["w"],
                               TAR["G"], mom["S_M"], mom["S_V"], lam)
    J_M = _nonneg(float(J_M.item()), (mom["S_M"], 1.0), "J_M")
    J_V = _nonneg(float(J_V.item()), (mom["S_V"], 1.0), "J_V")
    return dict(J=lam * J_M + (1.0 - lam) * J_V, J_M=J_M, J_V=J_V,
                x_norm_mean=float(Bd.norm(dim=1).mean().item()),
                y_norm_mean=float(Yd.norm(dim=1).mean().item()))


# ================================================================ the fit
LIP_CURVE = ("step", "J", "J_M", "J_V", "x_norm_mean", "y_norm_mean",
             "g_flip_frac")


def _rms(t):
    td = t.detach().double()
    return float(td.pow(2).mean().sqrt().item()) if td.numel() else 0.0


def fit_lip(TAR, B0, Y0, Bg, steps, fit_lr, lam, eval_every, adam_eps,
            dtype=torch.float32, mb=0, jitter=0.0, snapshot_best=False,
            sched="const", snap_steps=(), wm_chunk=8192, ste=False,
            atom_mb=0, atom_mode="naive", untied_g=False, eval_TAR=None, es_chunk=0,
            verbose=print):
    """One LIP fit: Adam on free natural-scale (atoms, targets), init (B0, Y0)
    -- the warm start IS the init (no redraw, no Y* solve) -- minimizing the
    exact full-batch J against the weighted target.  RELATIVE lr: each param
    group steps at fit_lr * rms(group at this fit's init).

    mb > 0        exp_1-style stochastic estimator: each step draws mb target
                  rows WITH replacement, proportional to their weights, and
                  uses them at uniform weight 1/mb.  Unbiased: the target-
                  quadratic part of J is the constant S terms; every sampled
                  term is linear in the target measure.
    jitter > 0    per-step Gaussian jitter on the target rows, sigma = jitter
                  in per-dim (unweighted) std units of the target; gates of
                  the jittered rows are re-derived (tied).  NOTE this biases
                  the target (distills a smoothed distribution).
    snapshot_best return the atoms from the best exact-J snapshot (the
                  eval_every cadence) instead of the final step.
    sched         "const" (waves 1-6) | "cosine": lr multiplier
                  0.5*(1+cos(pi*t/steps)) -- the kip_vs_cpx fit schedule
                  WITHOUT its 100-step warmup.
    snap_steps    payload snapshots: at each listed step the fp32 atoms are
                  stored and returned in `snaps` (the fit continues; under
                  sched=cosine a mid-fit snapshot is NOT a shorter cosine fit
                  -- its anneal never finished).
    wm_chunk      row-chunk of the one-time wmoments pass (exact either way).
    ste           straight-through estimator (step gates only): forward J is
                  unchanged, but the atoms' gate factor backpropagates through
                  a sigmoid surrogate instead of the frozen a.e. mask.
    atom_mb > 0   wave-31 ATOM-side subsampling: each step draws atom_mb of
                  the m atoms uniformly WITHOUT replacement.  atom_mode:
                  'naive'  the subset IS the atom set that step (quadratic
                           both sides + cross, uniform 1/atom_mb) -- the
                           literal target trick, BIASED (quadratic in the
                           sampled measure; the modal-collapse control);
                  'half'   quadratic = full x subset, cross = full-atom --
                           UNBIASED for J and its gradient, cost linear in m;
                  'ind'    quadratic = two INDEPENDENT subsets (atom_mb x
                           atom_mb), cross = full-atom -- also unbiased.
    eval_TAR      wave 40 PROBE target: when given, EVERY exact fp64 eval
                  (curve, snaps, final, S constants for reported J values)
                  runs against eval_TAR instead of TAR; the step loop still
                  samples the (possibly huge) TAR.  Use with a LAZY TAR --
                  G=None (target gates derived per step for the sampled mb
                  rows, never precomputed) and Z stored fp16/fp32 (cast to
                  the fit dtype after indexing) -- so a k-draw union target
                  whose exact J is O(n^2)-infeasible still fits.  A lazy TAR
                  requires mb > 0 and excludes jitter.
    untied_g      wave 36: the per-atom gates become a FREE real parameter
                  (m x k), initialized at the tied value gates(B0) and
                  optimized jointly (own Adam group at fit_lr * its init
                  rms).  Atoms stop being points with derived gates and
                  become (x, g, y) triples; every exact eval uses the stored
                  g; g_flip_frac becomes the fraction of g crossing the 0.5
                  mask threshold vs init (sigmoid-centric).  The payload and
                  snaps gain a "G" entry.  Not combinable with ste/atom_mb.

    Gates follow TAR["gate"] = (act, scale); smooth acts are differentiable
    in the atoms, so their gradient flows through the gate term naturally.
    The reported curve is ALWAYS the exact fp64 J of the stored state against
    the true (un-jittered, full) target.
    -> dict(B fp32, Yat fp32, curve, snaps, status, best_step, final_*)."""
    assert not torch.backends.cuda.matmul.allow_tf32, \
        "TF32 breaks the fp32 mask-Gram exactness argument"
    ETAR = eval_TAR if eval_TAR is not None else TAR
    if TAR["G"] is None:                  # lazy target (wave 40)
        assert mb > 0 and jitter == 0.0 and eval_TAR is not None, \
            "a lazy TAR needs the mb estimator, no jitter, and a probe"
    mom = wmoments(ETAR, chunk=wm_chunk)
    Bp = B0.detach().to(dtype).clone().requires_grad_(True)
    Yp = Y0.detach().to(dtype).clone().requires_grad_(True)
    m = Bp.shape[0]
    groups = [dict(params=[Bp], lr=fit_lr * _rms(Bp)),
              dict(params=[Yp], lr=fit_lr * _rms(Yp))]
    Gp = G0u = None
    if untied_g:
        assert not (ste or atom_mb), "untied_g excludes ste/atom_mb"
        ga, gs, *gal = TAR.get("gate", ("step", 1.0))
        Gp = gates(B0.detach().to(dtype), Bg.to(dtype), ga, gs,
                   gal[0] if gal else 1.0)
        Gp = Gp.to(dtype).clone().requires_grad_(True)
        G0u = (Gp.detach() > 0.5).clone()
        groups.append(dict(params=[Gp], lr=fit_lr * _rms(Gp)))
    opt = torch.optim.Adam(groups, eps=adam_eps)
    assert sched in ("const", "cosine"), sched
    scheduler = (torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: 0.5 * (1.0 + math.cos(math.pi * min(t, steps)
                                             / float(steps))))
        if sched == "cosine" else None)
    assert atom_mode in ("naive", "half", "ind"), atom_mode
    assert 0 <= atom_mb <= m, (atom_mb, m)
    assert not (atom_mb and ste), "atom subsampling + STE not supported"
    snap_set = {int(s) for s in snap_steps}
    assert all(0 < s < steps for s in snap_set), \
        f"snap_steps must lie inside the fit: {sorted(snap_set)} vs {steps}"
    snaps = []
    # fp16-stored Z stays fp16 until indexed (a 5M-row union in fp32 would
    # not fit); sampled rows are cast to the fit dtype per step
    Z_f = (TAR["Z"] if TAR["Z"].dtype == torch.float16
           else TAR["Z"].to(dtype))
    Y_f = TAR["Y"].to(dtype)
    w_f, Bg_f = TAR["w"].to(dtype), Bg.to(dtype)
    g_act, g_scale, *g_al = TAR.get("gate", ("step", 1.0))
    g_alpha = g_al[0] if g_al else 1.0
    assert not ste or g_act == "step", "STE is a step-gate gradient surrogate"
    G0 = masks(Bp.detach(), Bg_f)      # init sign masks, for the flip metric
    Z_std = Z_f.std(dim=0) if jitter > 0 else None
    w_mb = (torch.full((mb,), 1.0 / mb, dtype=dtype, device=Z_f.device)
            if mb else None)

    curve = {k: [] for k in LIP_CURVE}
    status, diverged_at = "ok", None
    best = dict(J=float("inf"), B=None, Y=None, step=0)
    t0 = time.time()

    def snapshot(step):
        with torch.no_grad():
            sn = exact_state(Bp, Yp, ETAR, mom, lam, Bg, G=Gp, chunk=es_chunk)
            flip = (((Gp.detach() > 0.5) != G0u) if untied_g
                    else (masks(Bp.detach(), Bg_f) != G0))
            sn.update(step=int(step),
                      g_flip_frac=float(flip.double().mean().item()))
            for k in LIP_CURVE:
                curve[k].append(sn[k])
            if snapshot_best and sn["J"] < best["J"]:
                best.update(J=sn["J"], step=int(step),
                            B=Bp.detach().clone(), Y=Yp.detach().clone(),
                            G=None if Gp is None else Gp.detach().clone())
            if step in snap_set:
                snaps.append(dict(step=int(step), J=sn["J"],
                                  B=Bp.detach().float().clone(),
                                  Y=Yp.detach().float().clone(),
                                  **({} if Gp is None else
                                     dict(G=Gp.detach().float().clone()))))
        return sn

    snapshot(0)
    for t in range(steps):
        Z_t, Y_t, w_t, G_t = Z_f, Y_f, w_f, TAR["G"]
        if mb:
            idx = torch.multinomial(w_f, mb, replacement=True)
            Z_t, Y_t, w_t = Z_f[idx].to(dtype), Y_f[idx], w_mb
            G_t = (gates(Z_t, Bg_f, g_act, g_scale, g_alpha)
                   if TAR["G"] is None else TAR["G"][idx])
        if jitter > 0:
            Z_t = Z_t + jitter * Z_std * torch.randn_like(Z_t)
            G_t = gates(Z_t, Bg_f, g_act, g_scale, g_alpha)  # tied jitter
        if atom_mb and atom_mode == "naive":
            ia = torch.randperm(m, device=Bp.device)[:atom_mb]
            B_s, Y_s = Bp[ia], Yp[ia]         # the subset IS the atom set
            G_s = gates(B_s, Bg_f, g_act, g_scale, g_alpha)
            loss, _, _ = _J_terms(B_s, Y_s, G_s, Z_t, Y_t, w_t, G_t,
                                  mom["S_M"], mom["S_V"], lam)
        elif atom_mb:
            iR = torch.randperm(m, device=Bp.device)[:atom_mb]
            iL = (None if atom_mode == "half"           # full left side
                  else torch.randperm(m, device=Bp.device)[:atom_mb])
            G_B = gates(Bp, Bg_f, g_act, g_scale, g_alpha)
            loss = _J_terms_split(Bp, Yp, G_B, iL, iR, Z_t, Y_t, w_t, G_t,
                                  mom["S_M"], mom["S_V"], lam)
        else:
            if untied_g:
                G_B = Gp                          # free gates, grad flows
            elif ste:
                z_pre = Bp @ Bg_f.T
                s = torch.sigmoid(z_pre)          # forward = step, backward =
                G_B = ((z_pre > 0).to(dtype) - s).detach() + s   # d(sigmoid)
            else:
                G_B = gates(Bp, Bg_f, g_act, g_scale, g_alpha)  # step: a.e.
            #                                    gradient; smooth: grad flows
            loss, _, _ = _J_terms(Bp, Yp, G_B, Z_t, Y_t, w_t, G_t,
                                  mom["S_M"], mom["S_V"], lam)
        lv = float(loss.detach().item())
        if not (lv == lv) or abs(lv) > DIVERGE_ABS:   # nan-safe
            status, diverged_at = "diverged", int(t)
            verbose(f"  [fit lip m={m}] DIVERGED at step {t}: J = {lv}")
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if (t + 1) % eval_every == 0 or t == steps - 1 or (t + 1) in snap_set:
            sn = snapshot(t + 1)
            if (t + 1) % (10 * eval_every) == 0 or t == steps - 1:
                verbose(f"  [fit lip m={m}] step {t + 1}/{steps} "
                        f"J={sn['J']:.6e} J_M={sn['J_M']:.3e} "
                        f"J_V={sn['J_V']:.3e} flip={sn['g_flip_frac']:.4f}")

    B_out, Y_out, best_step = Bp.detach(), Yp.detach(), None
    G_out = None if Gp is None else Gp.detach()
    if snapshot_best and best["B"] is not None:
        B_out, Y_out, best_step = best["B"], best["Y"], best["step"]
        G_out = best.get("G")
    fin = exact_state(B_out, Y_out, ETAR, mom, lam, Bg, G=G_out, chunk=es_chunk)
    B_out, Y_out = B_out.float().clone(), Y_out.float().clone()
    G_out = None if G_out is None else G_out.float().clone()
    verbose(f"  [fit lip m={m}] {status} J={fin['J']:.6e} J_M={fin['J_M']:.3e} "
            f"J_V={fin['J_V']:.3e} J0={curve['J'][0]:.3e} "
            + (f"(best snapshot step {best_step}) " if best_step is not None
               else "") + f"({time.time() - t0:.0f}s)")
    return dict(status=status, diverged_at=diverged_at, curve=curve,
                B=B_out, Yat=Y_out, G=G_out, best_step=best_step, snaps=snaps,
                **{f"final_{k}": v for k, v in fin.items()}, **mom)


# ================================================================ selftest
def _phi(Z, G, kk):
    """Explicit Phi(x) = (g (x) x)/sqrt(k), materialized (selftest only)."""
    n = Z.shape[0]
    return (G.double().unsqueeze(2) * Z.double().unsqueeze(1)
            ).reshape(n, -1) / kk ** 0.5


def selftest():
    """Prove the weighted math on tiny CPU fp64 problems against explicitly
    materialized moment tensors."""
    torch.manual_seed(0)
    d, kk, C, m = 5, 7, 3, 12
    Bg = torch.randn(kk, d, dtype=torch.float64) / d ** 0.5
    ok = lambda a, b, tag, tol=1e-9: (  # noqa: E731
        abs(a - b) <= tol * max(1.0, abs(a), abs(b))
        or (_ for _ in ()).throw(AssertionError(f"{tag}: {a} != {b}")))

    def explicit(Z, Y, w):
        Phi = _phi(Z, masks(Z, Bg), kk)
        M = Phi.T @ (w[:, None] * Phi)
        V = Phi.T @ (w[:, None] * Y.double())
        return M, V

    # 1. weighted S constants == explicit ||M||^2, ||V||^2
    Z1, Y1 = torch.randn(40, d, dtype=torch.float64), torch.randn(40, C, dtype=torch.float64)
    Z2, Y2 = torch.randn(30, d, dtype=torch.float64), torch.randn(30, C, dtype=torch.float64)
    TAR = make_target([(Z1, Y1, 0.7), (Z2, Y2, 0.3)], Bg)
    mom = wmoments(TAR)
    M, V = explicit(TAR["Z"], TAR["Y"], TAR["w"])
    ok(mom["S_M"], float((M * M).sum().item()), "S_M explicit")
    ok(mom["S_V"], float((V * V).sum().item()), "S_V explicit")

    # 1b. chunked wmoments == unchunked (pure re-ordering of exact sums)
    mom13 = wmoments(TAR, chunk=13)
    ok(mom13["S_M"], mom["S_M"], "S_M chunked", tol=1e-12)
    ok(mom13["S_V"], mom["S_V"], "S_V chunked", tol=1e-12)

    # 2. moment stacking: mixture target == w1*M(part1) + w2*M(part2)
    M1, V1 = explicit(Z1, Y1, torch.full((40,), 1 / 40, dtype=torch.float64))
    M2, V2 = explicit(Z2, Y2, torch.full((30,), 1 / 30, dtype=torch.float64))
    ok(float(((0.7 * M1 + 0.3 * M2) - M).norm().item()), 0.0, "M stacking")
    ok(float(((0.7 * V1 + 0.3 * V2) - V).norm().item()), 0.0, "V stacking")

    # 3. exact_state == explicit Frobenius errors, for arbitrary atoms
    B, Yat = torch.randn(m, d, dtype=torch.float64), torch.randn(m, C, dtype=torch.float64)
    Mh, Vh = explicit(B, Yat, torch.full((m,), 1 / m, dtype=torch.float64))
    st = exact_state(B, Yat, TAR, mom, 0.4, Bg)
    ok(st["J_M"], float(((Mh - M) ** 2).sum().item()), "J_M explicit")
    ok(st["J_V"], float(((Vh - V) ** 2).sum().item()), "J_V explicit")

    # 4. exactness: atoms == the rows of a UNIFORM target -> J = 0
    TU = make_target([(Z1, Y1, 1.0)], Bg)
    z = exact_state(Z1, Y1, TU, wmoments(TU), 0.5, Bg)
    ok(z["J"], 0.0, "J(self) = 0", tol=1e-12)

    # 5. the fit improves J from a random init (fp64, tiny budget)
    f = fit_lip(TAR, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                verbose=lambda *a: None)
    assert f["status"] == "ok" and f["final_J"] < 0.5 * st["J"], \
        f"fit did not improve: {f['final_J']} vs {st['J']}"

    # 6. mb estimator is unbiased: E[minibatch loss] == exact J (the target-
    #    quadratic part is the S constants; sampled terms are linear in the
    #    target measure).  Monte-Carlo mean vs 4-sigma band.
    lam = 0.4
    Jx, _, _ = _J_terms(B, Yat, masks(B, Bg), TAR["Z"], TAR["Y"], TAR["w"],
                        TAR["G"], mom["S_M"], mom["S_V"], lam)
    draws = []
    w_mb = torch.full((8,), 1.0 / 8, dtype=torch.float64)
    for _ in range(4000):
        idx = torch.multinomial(TAR["w"], 8, replacement=True)
        Jd, _, _ = _J_terms(B, Yat, masks(B, Bg), TAR["Z"][idx],
                            TAR["Y"][idx], w_mb, TAR["G"][idx],
                            mom["S_M"], mom["S_V"], lam)
        draws.append(float(Jd.item()))
    dr = torch.tensor(draws)
    se = float(dr.std().item()) / len(draws) ** 0.5
    assert abs(float(dr.mean().item()) - float(Jx.item())) < 4 * se, \
        f"mb estimator biased: {dr.mean().item()} vs {Jx.item()} (se {se})"

    # 7. jitter fit still improves the TRUE exact J (reported curve is always
    #    against the un-jittered full target)
    fj = fit_lip(TAR, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                 eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                 jitter=0.1, verbose=lambda *a: None)
    assert fj["status"] == "ok" and fj["final_J"] < 0.5 * st["J"], \
        f"jitter fit did not improve: {fj['final_J']} vs {st['J']}"

    # 8. snapshot_best returns the min of the exact-J curve
    fs = fit_lip(TAR, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                 eval_every=50, adam_eps=1e-8, dtype=torch.float64,
                 snapshot_best=True, verbose=lambda *a: None)
    ok(fs["final_J"], min(fs["curve"]["J"]), "snapshot_best == curve min",
       tol=1e-12)
    st_b = exact_state(fs["B"].double(), fs["Yat"].double(), TAR, mom, 0.5, Bg)
    ok(st_b["J"], fs["final_J"], "snapshot_best atoms match reported J",
       tol=1e-6)   # fp32 round-trip of the returned atoms

    # 9. cosine schedule: runs clean and still improves J (lr multiplier is 0
    #    at the horizon by construction)
    fc = fit_lip(TAR, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                 eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                 sched="cosine", verbose=lambda *a: None)
    assert fc["status"] == "ok" and fc["final_J"] < 0.5 * st["J"], \
        f"cosine fit did not improve: {fc['final_J']} vs {st['J']}"

    # 10. snap payloads: steps recorded in order; stored atoms reproduce the
    #     curve J at that step (fp32 round-trip tolerance)
    fp_ = fit_lip(TAR, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                  eval_every=50, adam_eps=1e-8, dtype=torch.float64,
                  snap_steps=(120, 200), verbose=lambda *a: None)
    assert [s["step"] for s in fp_["snaps"]] == [120, 200], fp_["snaps"]
    for s in fp_["snaps"]:
        i = fp_["curve"]["step"].index(s["step"])
        ok(s["J"], fp_["curve"]["J"][i], f"snap J curve@{s['step']}", tol=1e-12)
        st_s = exact_state(s["B"].double(), s["Y"].double(), TAR, mom, 0.5, Bg)
        ok(st_s["J"], s["J"], f"snap atoms J@{s['step']}", tol=1e-6)

    # 11. gate-activation generality: explicit moments == kernel machinery
    #     for a smooth scaled gate, and the fit (grad THROUGH the gates)
    #     improves the exact J
    gt = ("tanh", 0.7, 2.0)          # scaled AND alpha-gained gate
    TG = make_target([(Z1, Y1, 0.6), (Z2, Y2, 0.4)], Bg, gate=gt)
    momg = wmoments(TG)
    Phi = _phi(TG["Z"], TG["G"], kk)
    Mg = Phi.T @ (TG["w"][:, None] * Phi)
    Vg = Phi.T @ (TG["w"][:, None] * TG["Y"])
    ok(momg["S_M"], float((Mg * Mg).sum().item()), "S_M tanh")
    ok(momg["S_V"], float((Vg * Vg).sum().item()), "S_V tanh")
    G_Bg = gates(B, Bg, *gt)
    PhiB = _phi(B, G_Bg, kk)
    Mh = PhiB.T @ PhiB / m
    Vh = PhiB.T @ Yat.double() / m
    stg = exact_state(B, Yat, TG, momg, 0.4, Bg)
    ok(stg["J_M"], float(((Mh - Mg) ** 2).sum().item()), "J_M tanh explicit")
    ok(stg["J_V"], float(((Vh - Vg) ** 2).sum().item()), "J_V tanh explicit")
    fg = fit_lip(TG, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                 eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                 verbose=lambda *a: None)
    assert fg["status"] == "ok" and fg["final_J"] < 0.5 * stg["J"], \
        f"tanh fit did not improve: {fg['final_J']} vs {stg['J']}"

    # 12. STE: forward J identical to the frozen-mask path at init, but the
    #     gradient differs -- after a few steps the atoms have diverged
    f_frz = fit_lip(TAR, B, Yat, Bg, steps=100, fit_lr=1e-2, lam=0.5,
                    eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                    verbose=lambda *a: None)
    f_ste = fit_lip(TAR, B, Yat, Bg, steps=100, fit_lr=1e-2, lam=0.5,
                    eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                    ste=True, verbose=lambda *a: None)
    ok(f_ste["curve"]["J"][0], f_frz["curve"]["J"][0], "STE J0 == frozen J0",
       tol=1e-12)
    d_ste = float((f_ste["B"] - f_frz["B"]).abs().max().item())
    assert d_ste > 1e-8, "STE gradient did not diverge from the frozen mask"

    # 13. atom subsampling (wave 31).  naive at p=m is the full J exactly
    #     (a permutation of the atom set); 'half' and 'ind' are unbiased
    #     Monte-Carlo estimators of J for FIXED atoms (and hence of its
    #     gradient: the estimator is linear in the selection); a short fit
    #     under each mode still improves the exact J.
    G_full = masks(B, Bg)
    perm = torch.randperm(m)
    Jp, _, _ = _J_terms(B[perm], Yat[perm], G_full[perm], TAR["Z"], TAR["Y"],
                        TAR["w"], TAR["G"], mom["S_M"], mom["S_V"], lam)
    ok(float(Jp.item()), float(Jx.item()), "naive p=m == full J", tol=1e-12)
    for mode, use_iL in (("half", False), ("ind", True)):
        draws = []
        for _ in range(4000):
            iR = torch.randperm(m)[:4]
            iL = torch.randperm(m)[:4] if use_iL else None
            draws.append(float(_J_terms_split(
                B, Yat, G_full, iL, iR, TAR["Z"], TAR["Y"], TAR["w"],
                TAR["G"], mom["S_M"], mom["S_V"], lam).item()))
        dr = torch.tensor(draws)
        se_a = float(dr.std().item()) / len(draws) ** 0.5
        assert abs(float(dr.mean().item()) - float(Jx.item())) < 4 * se_a, \
            f"{mode} estimator biased: {dr.mean().item()} vs {Jx.item()}"
    for mode in ("naive", "half", "ind"):
        fa = fit_lip(TAR, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                     eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                     atom_mb=4, atom_mode=mode, verbose=lambda *a: None)
        assert fa["status"] == "ok" and fa["final_J"] < 0.7 * st["J"], \
            f"atom_mb {mode} fit did not improve: {fa['final_J']} vs {st['J']}"

    # 14. untied gates (wave 36).  At init the free gates ARE the tied ones,
    #     so J0 matches the tied fit's J0 exactly; the gradient reaches G
    #     (the gates move); exact_state with the returned G reproduces the
    #     reported final J; a short fit beats the TIED fit's J (a superset
    #     parametrization from the same init); tied fit is unaffected.
    f_tie = fit_lip(TG, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                    eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                    verbose=lambda *a: None)
    f_un = fit_lip(TG, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                   eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                   untied_g=True, verbose=lambda *a: None)
    ok(f_un["curve"]["J"][0], f_tie["curve"]["J"][0], "untied J0 == tied J0",
       tol=1e-12)
    G_init = gates(B, Bg, *gt)
    d_g = float((f_un["G"].double() - G_init).abs().max().item())
    assert d_g > 1e-8, "untied gates did not move"
    st_un = exact_state(f_un["B"].double(), f_un["Yat"].double(), TG, momg,
                        0.5, Bg, G=f_un["G"].double())
    ok(st_un["J"], f_un["final_J"], "untied exact_state(G) == reported",
       tol=1e-6)   # fp32 round-trip of the returned (B, Y, G)
    assert f_un["final_J"] < f_tie["final_J"], \
        f"untied did not beat tied: {f_un['final_J']} vs {f_tie['final_J']}"
    assert f_tie["G"] is None, "tied fit must not return gates"

    # 15. lazy target + probe (wave 40).  A lazy TAR (G=None, fp32 rows,
    #     uniform weights) with eval_TAR = the full precomputed target:
    #     J0 equals the tied fit's J0 exactly (the probe IS the target
    #     here), the mb fit improves the probe J, and the final reported J
    #     matches an independent exact_state of the returned atoms.
    # 16. row-chunked exact_state (wave 44) == unchunked, to fp64
    #     re-ordering tolerance, tied and untied
    st_c = exact_state(B, Yat, TG, momg, 0.4, Bg, chunk=13)
    ok(st_c["J_M"], stg["J_M"], "chunked J_M", tol=1e-12)
    ok(st_c["J_V"], stg["J_V"], "chunked J_V", tol=1e-12)
    G_rand = torch.rand(m, kk, dtype=torch.float64)
    st_u1 = exact_state(B, Yat, TG, momg, 0.4, Bg, G=G_rand)
    st_u2 = exact_state(B, Yat, TG, momg, 0.4, Bg, G=G_rand, chunk=7)
    ok(st_u2["J"], st_u1["J"], "chunked untied J", tol=1e-12)

    # TG has non-uniform part weights; use a uniform-weight twin so the
    # lazy uniform-w TAR matches its probe exactly
    ZU = torch.cat([Z1, Z2])
    YU = torch.cat([Y1, Y2])
    TU2 = make_target([(ZU, YU, 1.0)], Bg, gate=gt)
    TL = dict(Z=ZU.float(), Y=YU.float(),
              w=torch.full((ZU.shape[0],), 1.0 / ZU.shape[0],
                           dtype=torch.float32), G=None, gate=gt)
    f_lz = fit_lip(TL, B, Yat, Bg, steps=300, fit_lr=1e-2, lam=0.5,
                   eval_every=100, adam_eps=1e-8, dtype=torch.float64,
                   mb=16, eval_TAR=TU2, verbose=lambda *a: None)
    st_u2 = exact_state(B, Yat, TU2, wmoments(TU2), 0.5, Bg)
    ok(f_lz["curve"]["J"][0], st_u2["J"], "lazy J0 == exact vs probe",
       tol=1e-9)
    assert f_lz["status"] == "ok" and f_lz["final_J"] < 0.7 * st_u2["J"], \
        f"lazy fit did not improve: {f_lz['final_J']} vs {st_u2['J']}"
    st_lz = exact_state(f_lz["B"].double(), f_lz["Yat"].double(), TU2,
                        wmoments(TU2), 0.5, Bg)
    ok(st_lz["J"], f_lz["final_J"], "lazy final == exact_state", tol=1e-6)

    print(f"[lip_fit selftest] ALL OK (fit J {st['J']:.3e} -> "
          f"{f['final_J']:.3e}; mb unbiased within {se:.1e}; jitter fit "
          f"{fj['final_J']:.3e}; snapshot best step {fs['best_step']}; "
          f"cosine {fc['final_J']:.3e}; {len(fp_['snaps'])} snap payloads; "
          f"tanh fit {fg['final_J']:.3e}; STE atom delta {d_ste:.2e}; "
          f"untied {f_un['final_J']:.3e} < tied {f_tie['final_J']:.3e})")


if __name__ == "__main__":
    selftest()
