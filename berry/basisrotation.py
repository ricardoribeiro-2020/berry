"""
  Basis rotation of degenerate Bloch wavefunctions, driven by the degenerate
  k-points found during the clustering step (``cluster`` or ``cluster0``).

  This is the *a posteriori* twin of ``degenrotation``: it runs AFTER cluster
  and reuses the exact same rotation machinery (connected-zone construction,
  SVD/Procrustes alignment, holonomy correction, iterative gauge refinement,
  BFS propagation), but takes its list of degenerate points from the
  clustering's ``degeneratefinal.npy`` instead of detecting degeneracies by
  energy.  ``degenrotation`` remains the *a priori* tool (run before cluster),
  which detects degeneracies directly from the eigenvalues.

  Behaviour:
    * A flagged k-point with no flagged neighbour forms a single-point zone and
      is rotated against its best clean (non-degenerate) external neighbour.
    * Flagged k-points adjacent to each other form a connected zone; the
      rotation is propagated across the whole zone via BFS.
    * Only flagged k-points are ever written; clean neighbours are read as
      reference anchors but never modified.

  For colinear calculations each wavefunction is a complex array of shape (nr,);
  rotated wavefunctions overwrite ``k0{nk}b0{band}.wfc`` in place.
  For non-colinear calculations each wavefunction is a spinor pair stored as
  ``-0``/``-1`` files; the same N x N unitary is applied to both components.
"""
import os
import logging
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from berry import log
from berry._subroutines.rotation_core import (
    _process_zone, build_zones,
)

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except Exception:
    pass


ENERGY_THRESHOLD = 0.0001  # eigenvalue units (eV for standard QE XML output)

# Energy-coherence threshold for grouping cluster-flagged bands into one
# rotatable subspace.  Looser than ENERGY_THRESHOLD on purpose: it only splits
# flagged pairs whose bands are *clearly* non-degenerate (default ~10 meV),
# otherwise the cluster's flag is trusted.  Set to None to disable the guard.
GROUP_ENERGY_THRESHOLD = 0.01  # eigenvalue units (eV)


# Module-level shared context populated by run_basis_rotation before forking.
# Workers read this dict (never write to it); large arrays are inherited via
# copy-on-write when using the fork start method on Linux / macOS.
_zone_ctx: dict = {}


# ---------------------------------------------------------------------------
# Phase 1' — read degenerate points from the clustering output
# ---------------------------------------------------------------------------

def _union_pairs(pairs: list) -> list:
    """Merge degenerate band pairs that share a band into maximal groups.

    A triplet degeneracy is stored by the clustering as the overlapping pairs
    {a,b}, {a,c}, {b,c}; this returns the single group {a,b,c}.  Handles
    transitive merges (a pair can connect two previously-disjoint groups).
    """
    groups: list = []
    for p in pairs:
        p = set(p)
        overlap     = [g for g in groups if g & p]
        non_overlap = [g for g in groups if not (g & p)]
        for g in overlap:
            p |= g
        groups = non_overlap + [p]
    return [frozenset(g) for g in groups]


def _degen_at_k_from_cluster(degeneratefinal, eigenvalues, nbnd, initial_band,
                             bands_final, logger, group_ethr=None) -> dict:
    """Build ``degen_at_k = {nk: [frozenset(absolute_bands), ...]}`` from the
    clustering's ``degeneratefinal.npy``.

    Each row is ``[k, c1, c2]`` -- a degenerate band pair at k-point k.  The
    clustering works in bands RELATIVE to ``min_band = initial_band`` (it slices
    ``eigenvalues[:, min_band:]`` and stores ``bands_final`` values in that
    relative space), whereas the rotation core, the wfc files and
    ``d.eigenvalues`` are ABSOLUTE.  We therefore add ``initial_band`` to map
    every band to the absolute index space before returning.

    Robustness: ``c1, c2`` are band values for the main cluster path and for
    cluster0's primary block, but cluster0's secondary block can emit slot
    indices instead.  For each row we consider both interpretations --
    value: ``(c + initial_band)`` and slot: ``(bands_final[k, c] + initial_band)``
    -- and keep whichever is the more energy-degenerate (smallest |E_a - E_b|,
    measured on the absolute ``eigenvalues``).  Ties favour the value
    interpretation.

    Energy-coherent grouping (``group_ethr``): the cluster flags degeneracy by
    dot product, not energy, so two flagged bands can be far apart in energy
    (e.g. band 3 independently flagged with bands 2 and 4 at the same k, where
    E2 ~ E3 but E4 is distant).  Blindly unioning shared-band pairs would lump
    such non-degenerate bands into one subspace and let the Procrustes step mix
    states that are not eigenstates.  A flagged pair is therefore kept as a
    degenerate edge only if its energy gap is below ``group_ethr``; pairs whose
    bands are *clearly* separated are dropped (the surviving edges are then
    unioned, so a real triple degeneracy is preserved while a spurious one is
    split).  ``group_ethr=None`` disables the guard and trusts every flagged
    pair regardless of energy.
    """
    if degeneratefinal is None or len(degeneratefinal) == 0:
        return {}

    deg = np.atleast_2d(np.asarray(degeneratefinal))
    if deg.ndim != 2 or deg.shape[1] < 3:
        logger.warning(
            f"\tdegeneratefinal has unexpected shape {deg.shape}; "
            f"expected (N, >=3) rows [k, b1, b2]. Nothing to do."
        )
        return {}

    n_slots = bands_final.shape[1] if bands_final is not None else 0

    pairs_at_k: dict = defaultdict(list)
    n_used = n_skipped = n_filtered = 0
    for row in deg:
        k, c1, c2 = int(row[0]), int(row[1]), int(row[2])

        candidates = []
        # Candidate A -- columns are band values relative to initial_band.
        a1, a2 = c1 + initial_band, c2 + initial_band
        if initial_band <= a1 < nbnd and initial_band <= a2 < nbnd:
            candidates.append((a1, a2))
        # Candidate B -- columns are slots into bands_final (cluster0 edge path).
        if 0 <= c1 < n_slots and 0 <= c2 < n_slots:
            b1 = int(bands_final[k, c1]) + initial_band
            b2 = int(bands_final[k, c2]) + initial_band
            if initial_band <= b1 < nbnd and initial_band <= b2 < nbnd:
                candidates.append((b1, b2))

        best, best_gap = None, None
        for a, b in candidates:
            if a == b:
                continue
            gap = abs(float(eigenvalues[k, a]) - float(eigenvalues[k, b]))
            if best_gap is None or gap < best_gap:
                best, best_gap = (a, b), gap

        if best is None:
            n_skipped += 1
            continue
        # Energy-coherence guard: drop a flagged pair whose bands are clearly
        # non-degenerate, so it cannot pull a distant band into the subspace.
        if group_ethr is not None and best_gap >= group_ethr:
            n_filtered += 1
            continue
        pairs_at_k[k].append(frozenset(best))
        n_used += 1

    degen_at_k = {k: _union_pairs(pairs) for k, pairs in pairs_at_k.items()}

    logger.info(
        f"\tRead {n_used} degenerate pair(s) from cluster across "
        f"{len(degen_at_k)} k-point(s)"
        + (f"; dropped {n_filtered} pair(s) with energy gap >= {group_ethr} (non-degenerate)"
           if n_filtered else "")
        + (f"; skipped {n_skipped} unusable row(s)" if n_skipped else "")
    )
    return degen_at_k


# ---------------------------------------------------------------------------
# Parallel zone worker
# ---------------------------------------------------------------------------

def _process_zone_worker(args: tuple) -> list:
    """Fork-based parallel zone worker (reads shared data from _zone_ctx)."""
    zi, sig, zone = args
    ctx = _zone_ctx
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
        holonomy_correction      = ctx['holonomy_correction'],
        holonomy_max_iter        = ctx['holonomy_max_iter'],
        holonomy_tol             = ctx['holonomy_tol'],
        use_wfc_cache            = ctx['use_wfc_cache'],
        n_workers                = 1,  # avoid nested parallelism inside a worker
        logger                   = logger,
        holonomy_min_plaquettes  = ctx.get('holonomy_min_plaquettes', 2),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_basis_rotation(
    max_band: int = -1,
    npr: int = 1,
    logger_name: str = "basis",
    logger_level: int = logging.INFO,
    compress: bool = False,
    flush: bool = False,
    ethr: float = ENERGY_THRESHOLD,
    group_ethr: float = GROUP_ENERGY_THRESHOLD,
    max_refinement_iter: int = 50,
    refinement_iter_cap: int = 500,
    refinement_anderson_m: int = 10,
    refinement_tol: float = 1e-4,
    holonomy_correction: bool = True,
    holonomy_max_iter: int = 20,
    holonomy_tol: float = 1e-4,
    holonomy_min_plaquettes: int = 2,
    use_wfc_cache: bool = True,
) -> None:
    logger = log(logger_name, "BASIS ROTATION", level=logger_level, flush=flush)
    logger.header()

    n_workers = npr

    initial_band = m.initial_band if m.initial_band != "dummy" else 0
    final_band   = m.final_band

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
    logger.info(f"\tEnergy threshold (reference selection): {ethr}")
    logger.info(f"\tEnergy-coherence grouping threshold: {group_ethr}"
                + ("  (disabled — trusting all cluster flags)" if group_ethr is None else ""))
    logger.info(
        f"\tHolonomy correction: {holonomy_correction}  "
        f"(max_iter={holonomy_max_iter}, tol={holonomy_tol:.1e}, "
        f"min_plaquettes={holonomy_min_plaquettes})"
    )
    logger.info(f"\tIn-memory wfc cache: {use_wfc_cache}")
    logger.info(f"\tParallelism: n_workers={n_workers}")
    logger.info()

    # ------------------------------------------------------------------
    # Load shared data
    # ------------------------------------------------------------------
    def _require(path, hint):
        if not os.path.isfile(path):
            logger.error(
                f"\tRequired input '{os.path.basename(path)}' not found in {m.data_dir}.\n"
                f"\t{hint}"
            )
            logger.footer()
            raise SystemExit(1)
        return np.load(path)

    # u-convention: .wfc files hold the periodic parts u_nk and overlaps are
    # taken directly between them, so phase.npy is no longer loaded (see
    # docs/berry_geometry_physics.md §1.1).  d_phase=None is threaded through
    # to keep the rotation_core signatures unchanged.
    d_phase     = None
    eigenvalues = d.eigenvalues                                     # shape (nks, nbnd)
    neighbors   = d.neighbors                                       # shape (nks, 2*dimensions)

    logger.info(f"\tEigenvalues loaded, shape: {eigenvalues.shape}")

    bands_final     = _require(os.path.join(m.data_dir, "bandsfinal.npy"),
                               "Run 'berry cluster' (or 'cluster0') before basis rotation.")
    degeneratefinal = _require(os.path.join(m.data_dir, "degeneratefinal.npy"),
                               "Run 'berry cluster' (or 'cluster0') before basis rotation.")
    logger.info(f"\tRead bandsfinal.npy {bands_final.shape} and degeneratefinal.npy")
    logger.info()

    # ==================================================================
    # Phase 1' — Read degenerate k-points from the clustering output
    # ==================================================================
    logger.info("\t****  Phase 1: Reading degenerate k-points from cluster  ****")
    logger.info()

    degen_at_k = _degen_at_k_from_cluster(
        degeneratefinal, eigenvalues, m.nbnd, initial_band, bands_final, logger,
        group_ethr=group_ethr,
    )

    # Honour the "maximum band to consider" CLI argument: drop any degenerate
    # group that involves a band above max_band (max_band < 0 disables the cut).
    if max_band is not None and max_band >= 0:
        filtered = {}
        for nk, groups in degen_at_k.items():
            kept = [g for g in groups if max(g) <= max_band]
            if kept:
                filtered[nk] = kept
        n_dropped = len(degen_at_k) - len(filtered)
        degen_at_k = filtered
        if n_dropped:
            logger.info(f"\tBand cut <= {max_band}: dropped {n_dropped} k-point(s) above the range")

    n_degen_kpts = len(degen_at_k)

    if n_degen_kpts == 0:
        logger.info("\tNo degenerate k-points flagged by the clustering. Nothing to do.")
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
    # Phase 2 — Form connected degenerate zones (isolated points vs zones)
    # ==================================================================
    logger.info("\t****  Phase 2: Forming degenerate zones  ****")
    logger.info()

    zones = build_zones(degen_at_k, neighbors, 2 * m.dimensions)

    n_isolated = sum(1 for _, z in zones if len(z) == 1)
    logger.info(f"\tFormed {len(zones)} zone(s): "
                f"{n_isolated} isolated point(s), {len(zones) - n_isolated} multi-point zone(s)")
    for zi, (sig, zone) in enumerate(zones):
        kind = "isolated" if len(zone) == 1 else "zone"
        logger.info(f"\t  Zone {zi:>4} [{kind}]: bands {sorted(sig)}, {len(zone)} k-point(s)")
    logger.info()

    # ==================================================================
    # Phases 3 & 4 — Reference selection and BFS rotation propagation
    # ==================================================================
    logger.info("\t****  Phases 3 & 4: Reference selection and BFS rotation propagation  ****")

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
        'holonomy_correction':      holonomy_correction,
        'holonomy_max_iter':        holonomy_max_iter,
        'holonomy_tol':             holonomy_tol,
        'holonomy_min_plaquettes':  holonomy_min_plaquettes,
        'use_wfc_cache':            use_wfc_cache,
        'logger_name':           logger_name,
        'logger_level':          logger_level,
    })

    # Group consecutive zones by signature; same-signature zones have disjoint
    # k-point sets and may run in parallel.
    zone_groups: list = []
    for zi, (sig, zone) in enumerate(zones):
        if zone_groups and zone_groups[-1][0] == sig:
            zone_groups[-1][1].append((zi, zone))
        else:
            zone_groups.append((sig, [(zi, zone)]))

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
                    f"{_n_zones} zone(s) -> {_n_pool} worker(s)"
                )
                _args = [(_zi, _sig, _zone) for _zi, _zone in _sig_zone_list]
                with ProcessPoolExecutor(max_workers=_n_pool, mp_context=_fork_ctx) as _ex:
                    for _zone_rows in _ex.map(_process_zone_worker, _args):
                        summary_rows.extend(_zone_rows)
            else:
                for _zi, _zone in _sig_zone_list:
                    _rows = _process_zone(
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
                        holonomy_min_plaquettes=holonomy_min_plaquettes,
                    )
                    summary_rows.extend(_rows)
    finally:
        _zone_ctx.clear()

    summary_rows.sort(key=lambda r: r[0])

    # ==================================================================
    # Phase 5 — Final report and output file
    # ==================================================================
    logger.info()
    logger.info("\t****  Final Report  ****")
    logger.info()
    logger.info(f"\tTotal zones processed                : {len(summary_rows)}")
    logger.info(f"\tTotal k-points with degenerate bands : {n_degen_kpts}")
    logger.info()

    hdr = (
        f"\t{'Zone':>5}  {'Bands':>24}  {'Type':>9}  "
        f"{'Size':>5}  {'Rotated':>7}  {'Root-k':>7}  {'<Sigma>':>9}  {'Holo':>5}"
    )
    logger.info(hdr)
    logger.info("\t" + "-" * (len(hdr) - 1))
    for zi, bands, zone_sz, n_rot, root_k, ref_type, ms, n_holo, *_ in summary_rows:
        logger.info(
            f"\t{zi:>5}  {str(bands):>24}  {ref_type:>9}  "
            f"{zone_sz:>5}  {n_rot:>7}  {root_k:>7}  {ms:>9.6f}  {n_holo:>5}"
        )

    # Save basisrotation_zones.npy -- each row: [zone_id, b1 ... bN (-1 padded), k-point]
    if summary_rows:
        max_n   = max(len(bands) for _, bands, *_ in summary_rows)
        records = []
        for zi, bands, _, _, _, _, _, _, zone_set, *_ in summary_rows:
            bs = list(bands) + [-1] * (max_n - len(bands))
            for nk in sorted(zone_set):
                records.append([zi] + bs + [nk])
        out_path = os.path.join(m.data_dir, "basisrotation_zones.npy")
        np.save(out_path, np.array(records, dtype=int))
        logger.info()
        logger.info(f"\tZone map saved to: {out_path}")
        logger.info(f"\t  Row format: [zone_id, b1, ..., b{max_n} (-1 if absent), k-point]")

    logger.info()
    logger.footer()


if __name__ == "__main__":
    run_basis_rotation()
