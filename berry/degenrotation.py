"""
  This program finds all degenerate subspaces in the Bloch wavefunctions,
  groups neighboring degenerate k-points into connected zones, selects a
  boundary reference k-point for each zone, and propagates an SVD-based
  (Procrustes) basis rotation outward through each zone via BFS.

  Run after wfcgen and before dot.

  For colinear calculations:   wavefunctions are complex arrays of shape (nr,).
  For non-colinear calculations: each wavefunction is a spinor with two
  components stored as separate files (band-0.wfc, band-1.wfc). The same
  N×N unitary rotation is applied to both spinor components simultaneously.
"""
import os
import time
import logging
from collections import deque, defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import combinations

import numpy as np
from numpy.linalg import svd
import scipy.linalg as sp_linalg

from berry import log

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except Exception:
    pass


ENERGY_THRESHOLD = 0.0001  # eigenvalue units (eV for standard QE XML output)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_wfc(nk: int, band: int, noncolin: bool, wfcdir: str):
    """Return wavefunction(s) for k-point nk, band band.

    Colinear    → ndarray of shape (nr,)
    Non-colinear → tuple (spin-up array, spin-down array), each shape (nr,)
    """
    def _read(path):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr["arr_0"]
        return arr

    if noncolin:
        f0 = _read(os.path.join(wfcdir, f"k0{nk}b0{band}-0.wfc"))
        f1 = _read(os.path.join(wfcdir, f"k0{nk}b0{band}-1.wfc"))
        return (f0, f1)
    else:
        return _read(os.path.join(wfcdir, f"k0{nk}b0{band}.wfc"))


def _save_wfc(wfc, nk: int, band: int, noncolin: bool, wfcdir: str, compress: bool) -> None:
    """Overwrite wavefunction file(s) for k-point nk, band band."""
    def _write(path, arr):
        with open(path, "wb") as fh:
            if compress:
                np.savez_compressed(fh, arr)
            else:
                np.save(fh, arr)

    if noncolin:
        _write(os.path.join(wfcdir, f"k0{nk}b0{band}-0.wfc"), wfc[0])
        _write(os.path.join(wfcdir, f"k0{nk}b0{band}-1.wfc"), wfc[1])
    else:
        _write(os.path.join(wfcdir, f"k0{nk}b0{band}.wfc"), wfc)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _available_memory_bytes() -> int:
    """Return available system RAM in bytes, or 0 if it cannot be determined.

    Tries psutil first (cross-platform), then /proc/meminfo (Linux fallback).
    Returns 0 when neither is available so callers can apply a conservative
    hard limit rather than crashing.
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except OSError:
        pass
    return 0


# ---------------------------------------------------------------------------
# Core linear-algebra helpers
# ---------------------------------------------------------------------------

def _overlap_matrix(ref_wfcs, cur_wfcs, dphase: np.ndarray, nr: int, noncolin: bool) -> np.ndarray:
    """Compute N×N Bloch-phase-corrected overlap matrix.

    M[i, j] = <ref_i | cur_j>
             = Σ_r dphase[r] * conj(ref_i[r]) * cur_j[r] / nr

    dphase must equal  phase(nk) * conj(phase(ref_nk))
    so that it accounts for the exp(i(k_nk − k_ref)·r) phase difference.

    For non-colinear wavefunctions the inner product sums both spinor components.
    """
    N = len(ref_wfcs)
    M = np.zeros((N, N), dtype=complex)
    ph = dphase  # alias for readability
    for i, ref in enumerate(ref_wfcs):
        for j, cur in enumerate(cur_wfcs):
            if noncolin:
                r0, r1 = ref
                c0, c1 = cur
                M[i, j] = (np.dot(ph * r0.conj(), c0) +
                            np.dot(ph * r1.conj(), c1)) / nr
            else:
                M[i, j] = np.dot(ph * ref.conj(), cur) / nr
    return M


def _apply_rotation(wfcs, R: np.ndarray, noncolin: bool):
    """Apply N×N unitary R to a list of N wavefunctions (Procrustes convention).

    New wavefunction k: psi_new_k = Σ_j R[j, k] * wfcs[j]

    This is column k of the matrix product  Ψ @ R  where Ψ[:, j] = wfcs[j].
    The same R is applied to both spinor components in the non-colinear case.
    """
    N = len(wfcs)
    result = []
    for k in range(N):
        if noncolin:
            acc0 = np.zeros_like(wfcs[0][0])
            acc1 = np.zeros_like(wfcs[0][1])
            for j in range(N):
                acc0 += R[j, k] * wfcs[j][0]
                acc1 += R[j, k] * wfcs[j][1]
            result.append((acc0, acc1))
        else:
            acc = np.zeros_like(wfcs[0])
            for j in range(N):
                acc += R[j, k] * wfcs[j]
            result.append(acc)
    return result


def _procrustes_rotation(M: np.ndarray) -> np.ndarray:
    """Return the N×N unitary R that maximises Re(Tr(M R)) via SVD.

    Solution to the orthogonal Procrustes problem:
        M = U Σ Vh  →  R = V U†  =  Vh.conj().T @ U.conj().T
    """
    U, _, Vh = svd(M)
    return Vh.conj().T @ U.conj().T


# ---------------------------------------------------------------------------
# Gauge correction helpers
# ---------------------------------------------------------------------------

def _correct_zone_holonomy(
    zone: set,
    bands: list,
    n_dir: int,
    neighbors: np.ndarray,
    d_phase: np.ndarray,
    noncolin: bool,
    wfcdir: str,
    nr: int,
    compress: bool,
    logger,
    max_iter: int = 20,
    tol: float = 1e-4,
    wfc_cache: dict | None = None,
) -> int:
    """Reduce plaquette holonomy defects before iterative gauge refinement.

    For each elementary 4-cycle (plaquette) nk→na→nc←nb←nk fully inside the
    zone, computes H = R01 @ R12 @ R23† @ R30†.  When ||H−I||_F > tol,
    distributes fractional corrections H^(−1/4), H^(−1/2), H^(−3/4) to
    vertices na, nc, nb respectively, reducing the gauge inconsistency
    introduced by the BFS path ordering.  Iterates until max ||H−I||_F < tol
    or max_iter sweeps done.  Returns the number of sweeps performed.
    """
    N        = len(bands)
    zone_set = set(zone)
    n_half   = n_dir // 2  # directions 0 .. n_half−1 are the forward directions

    # Enumerate all elementary plaquettes fully inside the zone.
    # Plaquette: nk --(d0)--> na --(d1)--> nc   and   nk --(d1)--> nb --(d0)--> nc.
    # Using d0 < d1 < n_half enumerates each plaquette exactly once on a periodic grid.
    plaquettes: list = []
    seen:       set  = set()
    for nk in sorted(zone):
        for d0 in range(n_half):
            na = neighbors[nk, d0]
            if na == -1 or na not in zone_set:
                continue
            for d1 in range(d0 + 1, n_half):
                nb = neighbors[nk, d1]
                if nb == -1 or nb not in zone_set:
                    continue
                nc_via_a = neighbors[na, d1]
                nc_via_b = neighbors[nb, d0]
                if (nc_via_a == -1 or nc_via_b == -1
                        or nc_via_a != nc_via_b
                        or nc_via_a not in zone_set):
                    continue
                nc  = nc_via_a
                key = tuple(sorted((nk, na, nc, nb)))
                if key not in seen:
                    seen.add(key)
                    plaquettes.append((nk, na, nc, nb))

    if not plaquettes:
        logger.info("\t    [Holonomy] No internal plaquettes found — skipping.")
        return 0

    logger.info(f"\t    [Holonomy] {len(plaquettes)} plaquette(s),  tol={tol:.1e}")

    def _w(nk_):
        if wfc_cache is not None:
            return [wfc_cache[(nk_, b)] for b in bands]
        return [_load_wfc(nk_, b, noncolin, wfcdir) for b in bands]

    def _proj_u(A: np.ndarray) -> np.ndarray:
        U, _, Vh = np.linalg.svd(A)
        return U @ Vh

    def _R(ref_k: int, cur_k: int, ref_wfcs, cur_wfcs) -> np.ndarray:
        dph = d_phase[:, cur_k] * d_phase[:, ref_k].conj()
        return _procrustes_rotation(_overlap_matrix(ref_wfcs, cur_wfcs, dph, nr, noncolin))

    max_defect = float("inf")
    for sweep in range(1, max_iter + 1):
        max_defect = 0.0

        for nk, na, nc, nb in plaquettes:
            w_nk = _w(nk);  w_na = _w(na);  w_nc = _w(nc);  w_nb = _w(nb)

            R01 = _R(nk, na, w_nk, w_na)  # rotate na to align with nk
            R12 = _R(na, nc, w_na, w_nc)  # rotate nc to align with na
            R23 = _R(nb, nc, w_nb, w_nc)  # rotate nc to align with nb (reversed edge)
            R30 = _R(nk, nb, w_nk, w_nb)  # rotate nb to align with nk (reversed edge)

            # Holonomy going nk → na → nc ← nb ← nk
            H      = R01 @ R12 @ R23.conj().T @ R30.conj().T
            defect = float(np.linalg.norm(H - np.eye(N, dtype=complex)))
            max_defect = max(max_defect, defect)
            if defect < tol:
                continue

            # H† ≈ H^{-1} since H is a product of near-unitary Procrustes rotations.
            # C1 = H^{-1/4} via two matrix square roots; project each onto U(N).
            try:
                C1 = _proj_u(sp_linalg.sqrtm(sp_linalg.sqrtm(H.conj().T)))  # H^{-1/4}
                C2 = _proj_u(C1 @ C1)                                         # H^{-1/2}
                C3 = _proj_u(C2 @ C1)                                         # H^{-3/4}
            except Exception:
                continue

            # Rotating wfcs at p by G changes R(ref, p) → G† @ R(ref, p),
            # distributing the holonomy accumulation evenly around the cycle.
            for p, G in ((na, C1), (nc, C2), (nb, C3)):
                new_w = _apply_rotation(_w(p), G, noncolin)
                for i, b in enumerate(bands):
                    _save_wfc(new_w[i], p, b, noncolin, wfcdir, compress)
                    if wfc_cache is not None:
                        wfc_cache[(p, b)] = new_w[i]

        logger.info(
            f"\t    [Holonomy] sweep {sweep:>3}/{max_iter}  "
            f"max ||H−I||_F = {max_defect:.4e}"
        )
        if max_defect < tol:
            logger.info(f"\t    [Holonomy] Converged in {sweep} sweep(s).")
            return sweep

    logger.warning(
        f"\t    [Holonomy] Did not converge in {max_iter} sweep(s) "
        f"(final max ||H−I||_F = {max_defect:.4e})."
    )
    return max_iter


# ---------------------------------------------------------------------------
# Degeneracy detection
# ---------------------------------------------------------------------------

def _degenerate_groups(energies: np.ndarray, band_indices: list, ethr: float) -> list:
    """Group band indices whose energies are within ethr of each other.

    Uses union-find; returns a list of frozensets, one per degenerate
    subspace, skipping singletons.
    """
    n = len(band_indices)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if abs(energies[band_indices[i]] - energies[band_indices[j]]) < ethr:
                union(i, j)

    groups: dict = defaultdict(list)
    for i, b in enumerate(band_indices):
        groups[find(i)].append(b)

    return [frozenset(g) for g in groups.values() if len(g) >= 2]


def _detect_kpt(args: tuple) -> tuple:
    """Phase 1 worker: detect degenerate groups at one k-point.

    Packed-tuple interface is required for ProcessPoolExecutor.map pickling.
    Returns (nk, groups) where groups is [] when the k-point is non-degenerate.
    """
    nk, energies_nk, band_range, ethr = args
    return nk, _degenerate_groups(energies_nk, band_range, ethr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _refine_zone_gauge(
    zone: set,
    bands: list,
    n_dir: int,
    neighbors: np.ndarray,
    d_phase: np.ndarray,
    eigenvalues: np.ndarray,
    ethr: float,
    noncolin: bool,
    wfcdir: str,
    nr: int,
    compress: bool,
    max_iter: int,
    tol: float,
    logger,
    anderson_m: int = 10,
    wfc_cache: dict | None = None,
    n_workers: int = 1,
) -> tuple:
    """Iterative multi-neighbour Procrustes gauge refinement with Anderson mixing.

    Each sweep (Gauss-Seidel pass) re-rotates every zone k-point to best align
    with ALL its valid neighbours simultaneously: zone-internal k-points (also
    being refined) and external non-degenerate k-points (fixed anchors that
    prevent global gauge drift).  The overlap matrices from every direction are
    summed before the Procrustes step so all neighbours vote on the rotation.

    After each Gauss-Seidel sweep, Anderson mixing (AA(m)) is applied to
    accelerate convergence.  Anderson mixing fits a least-squares linear
    combination of the last `anderson_m` residuals that best predicts the next
    fixed-point, then extrapolates the rotation matrices in that direction and
    reprojects each N×N block onto U(N) via SVD.  This converts the slow
    (linear, ~0.97/iter) Gauss-Seidel convergence into effective superlinear
    convergence, typically reducing required iterations by 5–10×.

    Set `anderson_m=0` to disable Anderson mixing (pure Gauss-Seidel).

    Convergence criterion: max Frobenius norm of (R − I) over all k-points in
    the Gauss-Seidel sweep drops below `tol`, or `max_iter` sweeps exhausted.

    Quality metric: Re(Tr(M_total @ R)) / N averaged over active k-points.
    Gauge-dependent; rises as neighbours agree on a common rotation.

    Returns (n_iter_done, mean_quality).
    """
    N           = len(bands)
    sorted_zone = sorted(zone)       # fixed ordering for flat state vectors
    NN          = N * N              # elements per k-point in the flat vector

    # ------------------------------------------------------------------
    # Pre-compute valid neighbours for each zone k-point.
    # ------------------------------------------------------------------
    valid_nbrs: dict = {}
    for nk in zone:
        nbrs = []
        for j in range(n_dir):
            nb = neighbors[nk, j]
            if nb == -1:
                continue
            if nb in zone:
                nbrs.append(nb)
            else:
                gap = min(
                    abs(eigenvalues[nb, bands[a]] - eigenvalues[nb, bands[b]])
                    for a in range(N) for b in range(a + 1, N)
                )
                if gap > ethr:
                    nbrs.append(nb)
        valid_nbrs[nk] = nbrs

    # ------------------------------------------------------------------
    # Wfc I/O helpers — route reads through the cache when available so
    # the sweep loop does not hit the disk on every neighbour access.
    # Writes always go to both the cache and disk to preserve GS semantics
    # (later k-points in the same sweep see the freshly updated state) and
    # to ensure the files on disk are always consistent.
    # ------------------------------------------------------------------
    def _get(nk_: int) -> list:
        if wfc_cache is not None:
            return [wfc_cache[(nk_, b)] for b in bands]
        return [_load_wfc(nk_, b, noncolin, wfcdir) for b in bands]

    def _put(nk_: int, new_wfcs: list) -> None:
        for i, b in enumerate(bands):
            _save_wfc(new_wfcs[i], nk_, b, noncolin, wfcdir, compress)
            if wfc_cache is not None:
                wfc_cache[(nk_, b)] = new_wfcs[i]

    # ------------------------------------------------------------------
    # Anderson mixing state.
    #   Q_acc[nk]  — accumulated rotation at k-point nk relative to the
    #                state at the very start of refinement:
    #                wfc_current[nk] = Q_acc[nk] @ wfc_initial[nk]
    #   hist_x     — list of flat state vectors  x_k (before each sweep)
    #   hist_f     — list of GS residuals        f_k = x_{k+1,GS} − x_k
    # ------------------------------------------------------------------
    Q_acc:  dict = {nk: np.eye(N, dtype=complex) for nk in zone}
    hist_x: list = []
    hist_f: list = []

    t0             = time.time()
    max_rot_change_0: float = 0.0
    mean_quality_0:   float = 0.0
    mean_quality:     float = 0.0

    # Jacobi mode requires the wfc cache so all workers read the same
    # pre-sweep state from RAM.  Fall back to Gauss-Seidel if unavailable.
    use_jacobi = n_workers > 1 and wfc_cache is not None
    if n_workers > 1 and wfc_cache is None:
        logger.warning(
            f"\t    Jacobi parallel sweep (n_workers={n_workers}) requires "
            "wfc_cache; falling back to Gauss-Seidel "
            "(pass use_wfc_cache=True to enable)."
        )

    for it in range(1, max_iter + 1):

        # Snapshot accumulated rotations before this sweep.
        x_before = np.concatenate([Q_acc[nk].ravel() for nk in sorted_zone])

        # --------------------------------------------------------------
        # Sweep — Jacobi (parallel) or Gauss-Seidel (serial)
        # --------------------------------------------------------------
        max_rot_change = 0.0
        total_quality  = 0.0
        n_active       = 0

        if use_jacobi:
            # Freeze the current wfc state so every worker reads the same
            # pre-sweep snapshot.  Shallow copy is sufficient because _put
            # replaces cache values (dict[key] = new_array) rather than
            # mutating arrays in-place, so snap entries stay unchanged.
            snap = dict(wfc_cache)

            def _jacobi_worker(nk):
                cur_w   = [snap[(nk, b)] for b in bands]
                M_total = None
                for nb in valid_nbrs[nk]:
                    ref_w   = [snap[(nb, b)] for b in bands]
                    dph     = d_phase[:, nk] * d_phase[:, nb].conj()
                    M_nb    = _overlap_matrix(ref_w, cur_w, dph, nr, noncolin)
                    M_total = M_nb if M_total is None else M_total + M_nb
                if M_total is None:
                    return nk, None, None, 0.0, 0.0
                R       = _procrustes_rotation(M_total)
                new_w   = _apply_rotation(cur_w, R, noncolin)
                quality = float(np.real(np.trace(M_total @ R))) / N
                rot_chg = float(np.linalg.norm(R - np.eye(N, dtype=complex)))
                return nk, R, new_w, quality, rot_chg

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                updates = list(ex.map(_jacobi_worker, sorted_zone))

            for nk, R, new_w, q, rc in updates:
                if R is None:
                    continue
                total_quality  += q
                n_active       += 1
                max_rot_change  = max(max_rot_change, rc)
                _put(nk, new_w)
                Q_acc[nk] = R @ Q_acc[nk]

        else:
            # Gauss-Seidel: each nk sees the freshly written state of
            # neighbours that were updated earlier in the same sweep.
            for nk in sorted_zone:
                cur_w   = _get(nk)
                M_total = None

                for nb in valid_nbrs[nk]:
                    ref_w   = _get(nb)
                    dph     = d_phase[:, nk] * d_phase[:, nb].conj()
                    M_nb    = _overlap_matrix(ref_w, cur_w, dph, nr, noncolin)
                    M_total = M_nb if M_total is None else M_total + M_nb

                if M_total is None:
                    continue

                R = _procrustes_rotation(M_total)
                # Re(Tr(M @ R)) = sum of singular values of M
                total_quality += float(np.real(np.trace(M_total @ R))) / N
                n_active      += 1
                max_rot_change = max(
                    max_rot_change,
                    float(np.linalg.norm(R - np.eye(N, dtype=complex))),
                )
                new_w = _apply_rotation(cur_w, R, noncolin)
                _put(nk, new_w)
                Q_acc[nk] = R @ Q_acc[nk]

        mean_quality = total_quality / n_active if n_active > 0 else 0.0

        # Logging and convergence check (uses pure GS residual).
        if it == 1:
            max_rot_change_0 = max_rot_change if max_rot_change > 0.0 else 1.0
            mean_quality_0   = mean_quality

        pct_iter = 100.0 * it / max_iter
        pct_conv = 100.0 * (1.0 - max_rot_change / max_rot_change_0)
        pct_qual = (
            100.0 * (mean_quality - mean_quality_0) / abs(mean_quality_0)
            if mean_quality_0 != 0.0 else 0.0
        )
        elapsed = time.time() - t0
        logger.info(
            f"\t    [{pct_iter:5.1f}%]  iter {it:>3}/{max_iter}  "
            f"max_ΔR={max_rot_change:.2e}  ΔR_reduced={pct_conv:+.1f}%  "
            f"quality={mean_quality:.6f} ({pct_qual:+.1f}%)  elapsed={elapsed:.1f}s"
        )
        if max_rot_change < tol:
            logger.info(
                f"\t    Converged: max_ΔR={max_rot_change:.2e} < tol={tol:.1e}"
            )
            return it, mean_quality

        # --------------------------------------------------------------
        # Anderson acceleration (AA(m))
        #
        # Fixed-point map:  g(x) = GS_sweep(x)
        # Residual:         f(x) = g(x) − x   (want f → 0)
        #
        # Given last m residuals, solve the least-squares problem
        #   min ‖f_k + ΔF θ‖   where ΔF_j = f_j − f_{j-1}
        # then extrapolate:
        #   x_{k+1}^AA = x_{k+1,GS} + (ΔX + ΔF) θ
        #              = x_k + f_k   + (ΔX + ΔF) θ
        # Finally project each N×N block back onto U(N) via SVD.
        #
        # IMPORTANT: f_gs must be captured from Q_acc right after the GS sweep
        # and BEFORE the correction is applied, so the history always stores
        # the true GS fixed-point residuals (not GS + Anderson combined).
        # --------------------------------------------------------------
        if anderson_m > 0:
            x_after_gs = np.concatenate([Q_acc[nk].ravel() for nk in sorted_zone])
            f_gs       = x_after_gs - x_before   # pure GS residual, captured now

            if hist_f:                            # need ≥1 previous residual
                n_use  = min(len(hist_f), anderson_m)
                all_f  = hist_f[-n_use:] + [f_gs]
                all_x  = hist_x[-n_use:] + [x_before]

                dF = np.column_stack(
                    [all_f[j] - all_f[j - 1] for j in range(1, len(all_f))]
                )
                dX = np.column_stack(
                    [all_x[j] - all_x[j - 1] for j in range(1, len(all_x))]
                )

                # Least-squares: min ‖f_gs + dF θ‖²
                theta, *_ = np.linalg.lstsq(dF, -f_gs, rcond=None)

                x_aa = x_after_gs + (dX + dF) @ theta

                # Safety: reject if Anderson step is >5× the plain GS step.
                # This guards against divergence in the early iterations when
                # the history is short and the curvature estimate is unreliable.
                gs_norm = np.linalg.norm(f_gs)
                aa_norm = np.linalg.norm(x_aa - x_before)
                if gs_norm > 0.0 and aa_norm < 5.0 * gs_norm:
                    # Project each N×N block onto U(N) and apply incremental
                    # correction  dR = Q_mix @ Q_acc†  to the stored wfc.
                    for i, nk in enumerate(sorted_zone):
                        block    = x_aa[i * NN : (i + 1) * NN].reshape(N, N)
                        U, _, Vh = np.linalg.svd(block)
                        Q_mix    = U @ Vh
                        dR       = Q_mix @ Q_acc[nk].conj().T
                        # Skip trivial corrections to avoid unnecessary disk I/O.
                        if np.linalg.norm(dR - np.eye(N, dtype=complex)) > 1e-10:
                            new_w = _apply_rotation(_get(nk), dR, noncolin)
                            _put(nk, new_w)
                        Q_acc[nk] = Q_mix

            # Append the pure-GS residual captured before any correction.
            hist_x.append(x_before)
            hist_f.append(f_gs)
            if len(hist_x) > anderson_m:
                hist_x.pop(0)
                hist_f.pop(0)

    return max_iter, mean_quality


# ---------------------------------------------------------------------------
# Per-zone processing — extracted so parallel zone workers can call it
# ---------------------------------------------------------------------------

# Module-level shared context populated by run_degenrotation before forking.
# Workers read this dict (never write to it); large arrays are inherited via
# copy-on-write when using the fork start method on Linux / macOS.
_zone_ctx: dict = {}


def _process_zone(
    zi: int,
    sig: frozenset,
    zone: set,
    d_phase: np.ndarray,
    eigenvalues: np.ndarray,
    neighbors: np.ndarray,
    noncolin: bool,
    wfcdir: str,
    nr: int,
    n_dir: int,
    compress: bool,
    ethr: float,
    max_refinement_iter: int,
    refinement_iter_cap: int,
    refinement_anderson_m: int,
    refinement_tol: float,
    holonomy_correction: bool,
    holonomy_max_iter: int,
    holonomy_tol: float,
    use_wfc_cache: bool,
    n_workers: int,
    logger,
) -> tuple:
    """Run Phases 3–4 plus boundary check, holonomy, and refinement for one zone.

    Returns a summary_row tuple:
        (zi, bands, zone_size, n_rotated, best_k, ref_type, mean_sigma, n_holo_sweeps)
    """
    bands = sorted(sig)
    N     = len(bands)

    logger.info()
    logger.info(f"\t--- Zone {zi}: bands {bands},  {len(zone)} k-point(s),  subspace dim = {N} ---")

    # ------------------------------------------------------------------
    # Phase 3 — Find the boundary k-point with the best external reference
    # ------------------------------------------------------------------
    best_k, best_nb, best_gap = None, None, -1.0

    for nk in zone:
        for j in range(n_dir):
            nb = neighbors[nk, j]
            if nb == -1 or nb in zone:
                continue
            gap = min(
                abs(eigenvalues[nb, bands[a]] - eigenvalues[nb, bands[b]])
                for a in range(N)
                for b in range(a + 1, N)
            )
            if gap <= ethr:
                continue
            if gap > best_gap:
                best_gap, best_k, best_nb = gap, nk, nb

    if best_k is not None:
        ref_type = "boundary"
        logger.info(f"\t  Boundary reference k-point : {best_k}")
        logger.info(f"\t  External reference k-point : {best_nb}")
        logger.info(f"\t  Min energy gap at external reference : {best_gap:.6f}")
    else:
        ref_type = "isolated"
        best_k   = sorted(zone)[0]
        best_nb  = None
        logger.info(f"\t  WARNING: Zone {zi} is fully isolated — no non-degenerate external neighbor found.")
        logger.info(f"\t           Choosing k={best_k} as arbitrary root.")
        logger.info(f"\t           Internal consistency is enforced but the absolute basis is arbitrary.")

    # ------------------------------------------------------------------
    # Phase 4 — BFS propagation of the Procrustes rotation
    # ------------------------------------------------------------------
    bfs_parent: dict  = {best_k: best_nb}
    queue: deque      = deque([best_k])
    visited_zone: set = set()
    n_rotated         = 0
    sigma_log         = []

    while queue:
        nk = queue.popleft()
        if nk in visited_zone:
            continue
        visited_zone.add(nk)

        ref_nk = bfs_parent[nk]

        if ref_nk is None:
            logger.debug(f"\t  k={nk}: isolated zone root, wavefunctions kept as-is")
        else:
            ref_wfcs = [_load_wfc(ref_nk, b, noncolin, wfcdir) for b in bands]
            cur_wfcs = [_load_wfc(nk,     b, noncolin, wfcdir) for b in bands]

            dphase = d_phase[:, nk] * d_phase[:, ref_nk].conj()

            M = _overlap_matrix(ref_wfcs, cur_wfcs, dphase, nr, noncolin)

            _, sigma, _ = svd(M)
            sigma_log.append(sigma)
            R        = _procrustes_rotation(M)
            new_wfcs = _apply_rotation(cur_wfcs, R, noncolin)

            logger.debug(f"\t  k={nk} (ref={ref_nk}): singular values {np.round(sigma, 5)}")

            for i, b in enumerate(bands):
                _save_wfc(new_wfcs[i], nk, b, noncolin, wfcdir, compress)

            n_rotated += 1

        for j in range(n_dir):
            nb = neighbors[nk, j]
            if nb != -1 and nb in zone and nb not in bfs_parent:
                bfs_parent[nb] = nk
                queue.append(nb)

    mean_sigma = float(np.mean([s.mean() for s in sigma_log])) if sigma_log else 1.0
    min_sigma  = float(np.concatenate(sigma_log).min())         if sigma_log else 1.0

    logger.info(f"\t  Rotated {n_rotated} / {len(zone)} k-point(s)")
    logger.info(f"\t  Mean singular value : {mean_sigma:.6f}  (1.0 = perfect alignment)")
    logger.info(f"\t  Min singular value  : {min_sigma:.6f}")

    # ------------------------------------------------------------------
    # Per-zone wfc cache
    # ------------------------------------------------------------------
    wfc_cache = None
    if use_wfc_cache:
        all_cache_kpts: set = set(zone)
        for nk in zone:
            for j in range(n_dir):
                nb = neighbors[nk, j]
                if nb != -1:
                    all_cache_kpts.add(nb)
        n_arrays  = len(all_cache_kpts) * len(bands) * (2 if noncolin else 1)
        est_bytes = n_arrays * nr * 16
        avail     = _available_memory_bytes()
        over_limit = (
            (avail > 0 and est_bytes > avail * 0.8) or
            (avail == 0 and est_bytes > 4 * 1024 ** 3)
        )
        if over_limit:
            logger.warning(
                f"\t  Wfc cache disabled: {est_bytes / 1e9:.2f} GB needed"
                + (f", {avail / 1e9:.2f} GB available" if avail > 0
                   else ", available memory unknown")
            )
        else:
            avail_str = (f"{avail / 1e9:.1f} GB available"
                         if avail > 0 else "available memory unknown")
            logger.info(
                f"\t  Building wfc cache: {len(all_cache_kpts)} k-point(s), "
                f"{est_bytes / 1e6:.1f} MB ({avail_str})"
            )
            t_cache = time.time()
            wfc_cache = {}
            for nk_c in all_cache_kpts:
                for b in bands:
                    wfc_cache[(nk_c, b)] = _load_wfc(nk_c, b, noncolin, wfcdir)
            logger.info(f"\t  Cache ready in {time.time() - t_cache:.1f}s")

    # ------------------------------------------------------------------
    # Boundary consistency check
    # ------------------------------------------------------------------
    n_holo_sweeps = 0
    boundary_edges: list = []
    for nk in zone:
        for j in range(n_dir):
            nb = neighbors[nk, j]
            if nb == -1 or nb in zone:
                continue
            gap = min(
                abs(eigenvalues[nb, bands[a]] - eigenvalues[nb, bands[b]])
                for a in range(N) for b in range(a + 1, N)
            )
            if gap <= ethr:
                continue
            if wfc_cache is not None:
                ref_wfcs_b = [wfc_cache[(nb, b)] for b in bands]
                cur_wfcs_b = [wfc_cache[(nk, b)] for b in bands]
            else:
                ref_wfcs_b = [_load_wfc(nb, b, noncolin, wfcdir) for b in bands]
                cur_wfcs_b = [_load_wfc(nk, b, noncolin, wfcdir) for b in bands]
            dphase_b   = d_phase[:, nk] * d_phase[:, nb].conj()
            M_b        = _overlap_matrix(ref_wfcs_b, cur_wfcs_b, dphase_b, nr, noncolin)
            _, sigma_b, _ = svd(M_b)
            boundary_edges.append((nk, nb, float(sigma_b.min())))
            logger.debug(f"\t    boundary k={nk}→ext={nb}: min_sigma={sigma_b.min():.6f}")

    spread = 0.0
    if boundary_edges:
        b_sigmas = [s for _, _, s in boundary_edges]
        b_min, b_max = min(b_sigmas), max(b_sigmas)
        spread = b_max - b_min
        logger.info(
            f"\t  Boundary consistency : {len(boundary_edges)} edge(s), "
            f"min_sigma in [{b_min:.6f}, {b_max:.6f}], spread={spread:.6f}"
        )
        if spread > 0.1:
            worst_nk, worst_nb, worst_sigma = min(boundary_edges, key=lambda t: t[2])
            logger.warning(
                f"\t  WARNING: boundary spread {spread:.6f} > 0.1 — "
                f"zone gauge is not fully consistent across all borders "
                f"(worst edge: k={worst_nk}→ext={worst_nb}, "
                f"worst_sigma={worst_sigma:.6f})"
            )
            if holonomy_correction and not (spread > 0.5 and worst_sigma < 0.6):
                logger.info(
                    f"\t  [Option B] Applying holonomy correction "
                    f"(spread={spread:.3f}, worst_sigma={worst_sigma:.3f})"
                )
                n_holo_sweeps = _correct_zone_holonomy(
                    zone=zone,
                    bands=bands,
                    n_dir=n_dir,
                    neighbors=neighbors,
                    d_phase=d_phase,
                    noncolin=noncolin,
                    wfcdir=wfcdir,
                    nr=nr,
                    compress=compress,
                    logger=logger,
                    max_iter=holonomy_max_iter,
                    tol=holonomy_tol,
                    wfc_cache=wfc_cache,
                )
                logger.info(f"\t  [Option B] Done: {n_holo_sweeps} sweep(s)")
            elif spread > 0.5 and worst_sigma < 0.6:
                logger.warning(
                    f"\t  [Option C not implemented] Topological discontinuity likely "
                    f"(spread={spread:.3f} > 0.5, worst_sigma={worst_sigma:.3f} < 0.6). "
                    f"Zone split skipped; refinement may not converge."
                )

    # ------------------------------------------------------------------
    # Iterative multi-neighbour gauge refinement
    # ------------------------------------------------------------------
    if max_refinement_iter > 0 and len(zone) > 1:
        zone_max_iter = max(
            max_refinement_iter,
            min(refinement_iter_cap, max_refinement_iter + 3 * len(zone)),
        )
        sweep_label = (f"Jacobi n={n_workers}" if n_workers > 1 else "Gauss-Seidel")
        logger.info(
            f"\t  --- Iterative gauge refinement "
            f"(max {zone_max_iter} iter, tol={refinement_tol:.1e}, "
            f"Anderson m={refinement_anderson_m}, sweep={sweep_label}) ---"
        )
        n_ref_iter, final_quality = _refine_zone_gauge(
            zone=zone,
            bands=bands,
            n_dir=n_dir,
            neighbors=neighbors,
            d_phase=d_phase,
            eigenvalues=eigenvalues,
            ethr=ethr,
            noncolin=noncolin,
            wfcdir=wfcdir,
            nr=nr,
            compress=compress,
            max_iter=zone_max_iter,
            tol=refinement_tol,
            anderson_m=refinement_anderson_m,
            logger=logger,
            wfc_cache=wfc_cache,
            n_workers=n_workers,
        )
        logger.info(
            f"\t  Refinement complete: {n_ref_iter} iter(s), "
            f"final quality={final_quality:.6f}"
        )

    return (zi, bands, len(zone), n_rotated, best_k, ref_type, mean_sigma, n_holo_sweeps)


def _process_zone_worker(args: tuple) -> tuple:
    """Fork-based parallel zone worker.

    Reads all shared data from the module-level _zone_ctx dict, which is
    populated by run_degenrotation before the ProcessPoolExecutor is created.
    With the fork start method (Linux/macOS), child processes inherit the
    parent's address space via copy-on-write — large arrays (d_phase,
    eigenvalues, neighbors) are shared without pickling.

    Logging uses the inherited logger from the parent; log lines from
    concurrent zones may interleave in the output file.
    """
    zi, sig, zone = args
    ctx    = _zone_ctx
    import logging as _logging
    logger = _logging.getLogger(ctx['logger_name'])
    return _process_zone(
        zi=zi, sig=sig, zone=zone,
        d_phase         = ctx['d_phase'],
        eigenvalues     = ctx['eigenvalues'],
        neighbors       = ctx['neighbors'],
        noncolin        = ctx['noncolin'],
        wfcdir          = ctx['wfcdir'],
        nr              = ctx['nr'],
        n_dir           = ctx['n_dir'],
        compress        = ctx['compress'],
        ethr            = ctx['ethr'],
        max_refinement_iter   = ctx['max_refinement_iter'],
        refinement_iter_cap   = ctx['refinement_iter_cap'],
        refinement_anderson_m = ctx['refinement_anderson_m'],
        refinement_tol        = ctx['refinement_tol'],
        holonomy_correction   = ctx['holonomy_correction'],
        holonomy_max_iter     = ctx['holonomy_max_iter'],
        holonomy_tol          = ctx['holonomy_tol'],
        use_wfc_cache         = ctx['use_wfc_cache'],
        n_workers             = 1,  # avoid nested parallelism inside a worker
        logger                = logger,
    )


def run_degenrotation(
    ethr: float = ENERGY_THRESHOLD,
    logger_name: str = "degenrotation",
    logger_level: int = logging.INFO,
    compress: bool = False,
    flush: bool = False,
    max_refinement_iter: int = 50,
    refinement_iter_cap: int = 500,
    refinement_anderson_m: int = 10,
    refinement_tol: float = 1e-4,
    holonomy_correction: bool = True,
    holonomy_max_iter: int = 20,
    holonomy_tol: float = 1e-4,
    use_wfc_cache: bool = True,
    n_workers: int = 1,
) -> None:
    logger = log(logger_name, "DEGENERATE BASIS ROTATION", level=logger_level, flush=flush)
    logger.header()

    initial_band = m.initial_band
    final_band   = m.final_band
    band_range   = list(range(initial_band, final_band + 1))

    # ------------------------------------------------------------------
    # Log run parameters
    # ------------------------------------------------------------------
    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tDirectory where the wfc are: {m.wfcdirectory}")
    logger.info(f"\tNumber of k-points: {m.nkx} x {m.nky} x {m.nkz} = {m.nks}")
    logger.info(f"\tTotal number of points in real space: {m.nr}")
    logger.info(f"\tBands: [{initial_band}, {final_band}]")
    logger.info(f"\tDimensions: {m.dimensions}")
    logger.info(f"\tNoncolinear: {m.noncolin}")
    logger.info(f"\tLSDA: {m.lsda}")
    logger.info(f"\tEnergy degeneracy threshold: {ethr}")
    logger.info(
        f"\tHolonomy correction: {holonomy_correction}  "
        f"(max_iter={holonomy_max_iter}, tol={holonomy_tol:.1e})"
    )
    logger.info(f"\tIn-memory wfc cache: {use_wfc_cache}")
    logger.info(
        f"\tParallelism: n_workers={n_workers}"
        + (" (Jacobi sweep within a zone; zone-level parallelism when >1 same-sig zone exists)"
           if n_workers > 1 else " (serial Gauss-Seidel)")
    )
    logger.info()

    # ------------------------------------------------------------------
    # Load shared data
    # ------------------------------------------------------------------
    d_phase    = np.load(os.path.join(m.data_dir, "phase.npy"))   # shape (nr, nks)
    eigenvalues = d.eigenvalues                                     # shape (nks, nbnd)
    neighbors   = d.neighbors                                       # shape (nks, 2*dimensions)

    logger.info(f"\tPhases loaded, shape: {d_phase.shape}")
    logger.info(f"\tEigenvalues loaded, shape: {eigenvalues.shape}")
    logger.info()

    # ==================================================================
    # Phase 1 — Detect degenerate subspaces at every k-point
    # ==================================================================
    logger.info("\t****  Phase 1: Detecting degenerate subspaces  ****")
    logger.info()

    degen_at_k: dict = {}
    if n_workers > 1:
        # Each task is cheap (union-find on n_bands elements), so use a
        # generous chunksize to amortise inter-process communication cost.
        chunksize = max(1, m.nks // (n_workers * 4))
        _args = [(nk, eigenvalues[nk], band_range, ethr) for nk in range(m.nks)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for nk, groups in ex.map(_detect_kpt, _args, chunksize=chunksize):
                if groups:
                    degen_at_k[nk] = groups
    else:
        for nk in range(m.nks):
            groups = _degenerate_groups(eigenvalues[nk], band_range, ethr)
            if groups:
                degen_at_k[nk] = groups

    n_degen_kpts = len(degen_at_k)
    logger.info(f"\tFound {n_degen_kpts} k-point(s) with at least one degenerate subspace")

    if n_degen_kpts == 0:
        logger.info("\tNo degenerate k-points found. Nothing to do.")
        logger.footer()
        return

    sig_counts: dict = defaultdict(int)
    for groups in degen_at_k.values():
        for g in groups:
            sig_counts[tuple(sorted(g))] += 1
    logger.info("\tDegenerate band groups and their k-point counts:")
    for sig, count in sorted(sig_counts.items(), key=lambda x: (len(x[0]), x[0])):
        logger.info(f"\t  bands {list(sig):}: {count} k-point(s)")
    logger.info()

    # ==================================================================
    # Phase 2 — Form connected degenerate zones per band-group signature
    # ==================================================================
    logger.info("\t****  Phase 2: Forming degenerate zones  ****")
    logger.info()

    # Collect every signature that appears as the exact degenerate group at some k-point.
    actual_sigs: set = {grp for groups in degen_at_k.values() for grp in groups}

    # A k-point with signature T contributes to every sub-signature S ⊆ T that is
    # itself a real signature elsewhere in the BZ.  This bridges gaps where the
    # degeneracy dimension increases (e.g. a doublet region flanked by a triplet
    # region): the doublet zone now includes the triplet k-points, keeping the zone
    # connected rather than split into two disconnected pieces.
    sig_kpoints: dict = defaultdict(set)
    for nk, groups in degen_at_k.items():
        for grp in groups:
            for size in range(2, len(grp) + 1):
                for sub in combinations(sorted(grp), size):
                    sub_frozen = frozenset(sub)
                    if sub_frozen in actual_sigs:
                        sig_kpoints[sub_frozen].add(nk)

    # Process signatures in ascending size order so that a sub-zone (e.g. {0,1})
    # is fully rotated before the enclosing super-zone ({0,1,2}) runs on top of it.
    zones = []  # list of (frozenset_of_bands, set_of_kpoints)
    for sig, kpoints_set in sorted(sig_kpoints.items(), key=lambda kv: len(kv[0])):
        visited: set = set()
        for start in kpoints_set:
            if start in visited:
                continue
            zone: set = set()
            q: deque = deque([start])
            while q:
                nk = q.popleft()
                if nk in visited:
                    continue
                visited.add(nk)
                zone.add(nk)
                for j in range(2 * m.dimensions):
                    nb = neighbors[nk, j]
                    if nb != -1 and nb in kpoints_set and nb not in visited:
                        q.append(nb)
            zones.append((sig, zone))

    logger.info(f"\tFormed {len(zones)} degenerate zone(s) across all band groups")
    for zi, (sig, zone) in enumerate(zones):
        logger.info(f"\t  Zone {zi:>4}: bands {sorted(sig)}, {len(zone)} k-point(s)")
    logger.info()

    # ==================================================================
    # Phases 3 & 4 — Reference selection and BFS rotation propagation
    # ==================================================================
    logger.info("\t****  Phases 3 & 4: Reference selection and BFS rotation propagation  ****")

    # Populate the shared context so fork-based workers inherit it.
    _zone_ctx.update({
        'd_phase':               d_phase,
        'eigenvalues':           eigenvalues,
        'neighbors':             neighbors,
        'noncolin':              m.noncolin,
        'wfcdir':                m.wfcdirectory,
        'nr':                    m.nr,
        'n_dir':                 2 * m.dimensions,
        'compress':              compress,
        'ethr':                  ethr,
        'max_refinement_iter':   max_refinement_iter,
        'refinement_iter_cap':   refinement_iter_cap,
        'refinement_anderson_m': refinement_anderson_m,
        'refinement_tol':        refinement_tol,
        'holonomy_correction':   holonomy_correction,
        'holonomy_max_iter':     holonomy_max_iter,
        'holonomy_tol':          holonomy_tol,
        'use_wfc_cache':         use_wfc_cache,
        'logger_name':           logger_name,
        'logger_level':          logger_level,
    })

    # Group consecutive zones by signature.  Same-signature zones have
    # disjoint k-point sets (guaranteed by BFS in Phase 2) and can run
    # in parallel.  Different signatures at the same band-set size may
    # share k-points via sub-signature expansion and must stay sequential.
    zone_groups: list = []
    for zi, (sig, zone) in enumerate(zones):
        if zone_groups and zone_groups[-1][0] == sig:
            zone_groups[-1][1].append((zi, zone))
        else:
            zone_groups.append((sig, [(zi, zone)]))

    # Check for fork availability once (not on Windows).
    try:
        _fork_ctx = mp.get_context('fork')
    except ValueError:
        _fork_ctx = None
        if n_workers > 1:
            logger.warning(
                "\tParallel zone processing requires the 'fork' start method "
                "(Linux/macOS); falling back to serial processing."
            )

    summary_rows = []

    try:
        for _sig, _sig_zone_list in zone_groups:
            _n_zones = len(_sig_zone_list)
            if _n_zones > 1 and n_workers > 1 and _fork_ctx is not None:
                _n_pool = min(n_workers, _n_zones)
                logger.info(
                    f"\t  [Parallel zones] bands {sorted(_sig)}: "
                    f"{_n_zones} zone(s) → {_n_pool} worker(s)"
                )
                _args = [(_zi, _sig, _zone) for _zi, _zone in _sig_zone_list]
                with ProcessPoolExecutor(max_workers=_n_pool, mp_context=_fork_ctx) as _ex:
                    _rows = list(_ex.map(_process_zone_worker, _args))
                summary_rows.extend(_rows)
            else:
                for _zi, _zone in _sig_zone_list:
                    _row = _process_zone(
                        zi=_zi, sig=_sig, zone=_zone,
                        d_phase=d_phase, eigenvalues=eigenvalues, neighbors=neighbors,
                        noncolin=m.noncolin, wfcdir=m.wfcdirectory, nr=m.nr,
                        n_dir=2 * m.dimensions, compress=compress, ethr=ethr,
                        max_refinement_iter=max_refinement_iter,
                        refinement_iter_cap=refinement_iter_cap,
                        refinement_anderson_m=refinement_anderson_m,
                        refinement_tol=refinement_tol,
                        holonomy_correction=holonomy_correction,
                        holonomy_max_iter=holonomy_max_iter,
                        holonomy_tol=holonomy_tol,
                        use_wfc_cache=use_wfc_cache,
                        n_workers=n_workers,
                        logger=logger,
                    )
                    summary_rows.append(_row)
    finally:
        _zone_ctx.clear()

    # sort by zi so the final report appears in zone-list order
    summary_rows.sort(key=lambda r: r[0])

    # ==================================================================
    # Phase 5 — Final report and output file
    # ==================================================================
    logger.info()
    logger.info("\t****  Final Report  ****")
    logger.info()
    logger.info(f"\tTotal zones processed             : {len(zones)}")
    logger.info(f"\tTotal k-points with degenerate bands : {n_degen_kpts}")
    logger.info()

    hdr = (
        f"\t{'Zone':>5}  {'Bands':>24}  {'Type':>9}  "
        f"{'Size':>5}  {'Rotated':>7}  {'Root-k':>7}  {'<Sigma>':>9}  {'Holo':>5}"
    )
    logger.info(hdr)
    logger.info("\t" + "-" * (len(hdr) - 1))
    for zi, bands, zone_sz, n_rot, root_k, ref_type, ms, n_holo in summary_rows:
        logger.info(
            f"\t{zi:>5}  {str(bands):>24}  {ref_type:>9}  "
            f"{zone_sz:>5}  {n_rot:>7}  {root_k:>7}  {ms:>9.6f}  {n_holo:>5}"
        )

    # Save degenzones.npy — each row: [zone_id, b1 ... bN (−1 padded), k-point]
    if zones:
        max_n   = max(len(sig) for sig, _ in zones)
        records = []
        for zi, (sig, zone) in enumerate(zones):
            bs = sorted(sig) + [-1] * (max_n - len(sig))
            for nk in sorted(zone):
                records.append([zi] + bs + [nk])
        out_path = os.path.join(m.data_dir, "degenzones.npy")
        np.save(out_path, np.array(records, dtype=int))
        logger.info()
        logger.info(f"\tZone map saved to: {out_path}")
        logger.info(f"\t  Row format: [zone_id, b1, ..., b{max_n} (−1 if absent), k-point]")

    logger.info()
    logger.footer()


if __name__ == "__main__":
    run_degenrotation()
