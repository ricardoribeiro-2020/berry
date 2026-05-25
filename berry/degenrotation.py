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
import logging
from collections import deque, defaultdict

import numpy as np
from numpy.linalg import svd

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_degenrotation(
    ethr: float = ENERGY_THRESHOLD,
    logger_name: str = "degenrotation",
    logger_level: int = logging.INFO,
    compress: bool = False,
    flush: bool = False,
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

    sig_kpoints: dict = defaultdict(set)
    for nk, groups in degen_at_k.items():
        for grp in groups:
            sig_kpoints[grp].add(nk)

    zones = []  # list of (frozenset_of_bands, set_of_kpoints)
    for sig, kpoints_set in sig_kpoints.items():
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

    summary_rows = []

    for zi, (sig, zone) in enumerate(zones):
        bands = sorted(sig)
        N     = len(bands)

        logger.info()
        logger.info(f"\t--- Zone {zi}: bands {bands},  {len(zone)} k-point(s),  subspace dim = {N} ---")

        # ------------------------------------------------------------------
        # Phase 3 — Find the boundary k-point with the best external reference
        # ------------------------------------------------------------------
        best_k, best_nb, best_gap = None, None, -1.0

        for nk in zone:
            for j in range(2 * m.dimensions):
                nb = neighbors[nk, j]
                if nb == -1 or nb in zone:
                    continue
                # Minimum pairwise energy separation of the target bands at the external neighbor
                gap = min(
                    abs(eigenvalues[nb, bands[a]] - eigenvalues[nb, bands[b]])
                    for a in range(N)
                    for b in range(a + 1, N)
                )
                # Skip external neighbors that are themselves degenerate for these bands:
                # using such a point as reference would yield a near-singular overlap matrix.
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
        # bfs_parent[nk] = the already-rotated k-point used as reference for nk.
        #                   None means 'isolated zone root; no rotation applied'.
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
                # Root of an isolated zone — wavefunctions are kept as-is
                logger.debug(f"\t  k={nk}: isolated zone root, wavefunctions kept as-is")
            else:
                ref_wfcs = [_load_wfc(ref_nk, b, m.noncolin, m.wfcdirectory) for b in bands]
                cur_wfcs = [_load_wfc(nk,     b, m.noncolin, m.wfcdirectory) for b in bands]

                # Phase factor: exp(i(k_nk − k_ref)·r)
                dphase = d_phase[:, nk] * d_phase[:, ref_nk].conj()

                M = _overlap_matrix(ref_wfcs, cur_wfcs, dphase, m.nr, m.noncolin)

                _, sigma, _ = svd(M)
                sigma_log.append(sigma)
                R        = _procrustes_rotation(M)
                new_wfcs = _apply_rotation(cur_wfcs, R, m.noncolin)

                logger.debug(f"\t  k={nk} (ref={ref_nk}): singular values {np.round(sigma, 5)}")

                for i, b in enumerate(bands):
                    _save_wfc(new_wfcs[i], nk, b, m.noncolin, m.wfcdirectory, compress)

                n_rotated += 1

            # Enqueue unvisited zone neighbors
            for j in range(2 * m.dimensions):
                nb = neighbors[nk, j]
                if nb != -1 and nb in zone and nb not in bfs_parent:
                    bfs_parent[nb] = nk
                    queue.append(nb)

        mean_sigma = float(np.mean([s.mean() for s in sigma_log])) if sigma_log else 1.0
        min_sigma  = float(np.concatenate(sigma_log).min())         if sigma_log else 1.0

        logger.info(f"\t  Rotated {n_rotated} / {len(zone)} k-point(s)")
        logger.info(f"\t  Mean singular value : {mean_sigma:.6f}  (1.0 = perfect alignment)")
        logger.info(f"\t  Min singular value  : {min_sigma:.6f}")

        summary_rows.append((zi, bands, len(zone), n_rotated, best_k, ref_type, mean_sigma))

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
        f"{'Size':>5}  {'Rotated':>7}  {'Root-k':>7}  {'<Sigma>':>9}"
    )
    logger.info(hdr)
    logger.info("\t" + "-" * (len(hdr) - 1))
    for zi, bands, zone_sz, n_rot, root_k, ref_type, ms in summary_rows:
        logger.info(
            f"\t{zi:>5}  {str(bands):>24}  {ref_type:>9}  "
            f"{zone_sz:>5}  {n_rot:>7}  {root_k:>7}  {ms:>9.6f}"
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
