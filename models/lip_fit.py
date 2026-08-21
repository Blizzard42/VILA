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
import time

import torch

DIVERGE_ABS = 1e12       # |loss| above this (or non-finite) aborts the fit


class PrecisionError(RuntimeError):
    """A reported squared error came out negative: a precision bug, not data."""


def masks(X, Bg):
    """g = 1{Bg x > 0}, the frozen-gate mask, as a bool tensor.  Computed at
    the input's own dtype: only the SIGN matters."""
    return X @ Bg.T.to(X.dtype) > 0


def _gram(GA, GB, out_dtype):
    """Binary-mask Gram, computed fp32 (exact: 0/1 entries, integer inner
    products <= k << 2^24), cast to the caller's dtype."""
    return (GA.float() @ GB.float().T).to(out_dtype)


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
def make_target(parts, Bg):
    """Weighted fit target from (Z, Y, w_total) parts: rows fp64, per-row
    weights w_part/n_part (sum over all parts = 1), gates derived once (the
    target rows never move).  parts = [(Z [n,d], Y [n,C], w_total), ...]."""
    Zs, Ys, ws = [], [], []
    for Z, Y, w in parts:
        n = Z.shape[0]
        Zs.append(Z.double())
        Ys.append(Y.double())
        ws.append(torch.full((n,), float(w) / n, dtype=torch.float64,
                             device=Z.device))
    Z, Y, w = torch.cat(Zs), torch.cat(Ys), torch.cat(ws)
    assert abs(float(w.sum().item()) - 1.0) < 1e-8, "target weights must sum to 1"
    return dict(Z=Z, Y=Y, w=w, G=masks(Z, Bg.double()))


def wmoments(TAR):
    """S_M = ||M||_F^2, S_V = ||V||_F^2 of the weighted target, exact fp64:
    S_M = sum_ab w_a w_b k(a,b)^2,  S_V = sum_ab w_a w_b (Y_a.Y_b) k(a,b)."""
    K = k_agalu(TAR["Z"], TAR["Z"], TAR["G"], TAR["G"])
    W = TAR["w"][:, None] * TAR["w"][None, :]
    return dict(S_M=float((W * K * K).sum().item()),
                S_V=float((W * (TAR["Y"] @ TAR["Y"].T) * K).sum().item()))


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


def exact_state(B, Yat, TAR, mom, lam, Bg):
    """THE exact fp64 (M, V) evaluator: every reported J goes through
    literally this expression.  Gates derived from the atoms (tied)."""
    Bd, Yd = B.detach().double(), Yat.detach().double()
    G_B = masks(Bd, Bg.double())
    J, J_M, J_V = _J_terms(Bd, Yd, G_B, TAR["Z"], TAR["Y"], TAR["w"],
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
            dtype=torch.float32, verbose=print):
    """One LIP fit: Adam on free natural-scale (atoms, targets), init (B0, Y0)
    -- the warm start IS the init (no redraw, no Y* solve) -- minimizing the
    exact full-batch J against the weighted target.  RELATIVE lr: each param
    group steps at fit_lr * rms(group at this fit's init).
    -> dict(B fp32, Yat fp32, curve, status, final_*)."""
    assert not torch.backends.cuda.matmul.allow_tf32, \
        "TF32 breaks the fp32 mask-Gram exactness argument"
    mom = wmoments(TAR)
    Bp = B0.detach().to(dtype).clone().requires_grad_(True)
    Yp = Y0.detach().to(dtype).clone().requires_grad_(True)
    m = Bp.shape[0]
    opt = torch.optim.Adam(
        [dict(params=[Bp], lr=fit_lr * _rms(Bp)),
         dict(params=[Yp], lr=fit_lr * _rms(Yp))], eps=adam_eps)
    Z_f, Y_f = TAR["Z"].to(dtype), TAR["Y"].to(dtype)
    w_f, Bg_f = TAR["w"].to(dtype), Bg.to(dtype)
    G0 = masks(Bp.detach(), Bg_f)                 # init masks, for flip drift

    curve = {k: [] for k in LIP_CURVE}
    status, diverged_at = "ok", None
    t0 = time.time()

    def snapshot(step):
        with torch.no_grad():
            sn = exact_state(Bp, Yp, TAR, mom, lam, Bg)
            sn.update(step=int(step),
                      g_flip_frac=float((masks(Bp.detach(), Bg_f) != G0)
                                        .double().mean().item()))
            for k in LIP_CURVE:
                curve[k].append(sn[k])
        return sn

    snapshot(0)
    for t in range(steps):
        G_B = masks(Bp, Bg_f)                     # tied: a.e. gradient
        loss, _, _ = _J_terms(Bp, Yp, G_B, Z_f, Y_f, w_f, TAR["G"],
                              mom["S_M"], mom["S_V"], lam)
        lv = float(loss.detach().item())
        if not (lv == lv) or abs(lv) > DIVERGE_ABS:   # nan-safe
            status, diverged_at = "diverged", int(t)
            verbose(f"  [fit lip m={m}] DIVERGED at step {t}: J = {lv}")
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (t + 1) % eval_every == 0 or t == steps - 1:
            sn = snapshot(t + 1)
            if (t + 1) % (10 * eval_every) == 0 or t == steps - 1:
                verbose(f"  [fit lip m={m}] step {t + 1}/{steps} "
                        f"J={sn['J']:.6e} J_M={sn['J_M']:.3e} "
                        f"J_V={sn['J_V']:.3e} flip={sn['g_flip_frac']:.4f}")

    fin = exact_state(Bp, Yp, TAR, mom, lam, Bg)
    verbose(f"  [fit lip m={m}] {status} J={fin['J']:.6e} J_M={fin['J_M']:.3e} "
            f"J_V={fin['J_V']:.3e} J0={curve['J'][0]:.3e} "
            f"({time.time() - t0:.0f}s)")
    return dict(status=status, diverged_at=diverged_at, curve=curve,
                B=Bp.detach().float().clone(), Yat=Yp.detach().float().clone(),
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
    print(f"[lip_fit selftest] ALL OK (fit J {st['J']:.3e} -> "
          f"{f['final_J']:.3e} in 300 steps)")


if __name__ == "__main__":
    selftest()
