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
import psutil

from berry import log

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except Exception:
    pass


ENERGY_THRESHOLD = 0.005  # eigenvalue units (eV for standard QE XML output); CLI -ethr overrides


from berry._subroutines.rotation_core import (
    _load_wfc, _save_wfc, _available_memory_bytes,
    _overlap_matrix, _apply_rotation, _procrustes_rotation,
    _correct_zone_holonomy, _degenerate_groups, _detect_kpt,
    _refine_zone_gauge, _run_bfs_rotation, _process_zone_suffix,
    _process_zone, build_zones,
)

# Module-level shared context populated by run_degenrotation before forking.
# Workers read this dict (never write to it); large arrays are inherited via
# copy-on-write when using the fork start method on Linux / macOS.
_zone_ctx: dict = {}


def _process_zone_worker(args: tuple) -> list:
    """Fork-based parallel zone worker.

    Reads all shared data from the module-level _zone_ctx dict, which is
    populated by run_degenrotation before the ProcessPoolExecutor is created.
    With the fork start method (Linux/macOS), child processes inherit the
    parent's address space via copy-on-write -- large arrays (d_phase,
    eigenvalues, neighbors) are shared without pickling.

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
        holonomy_correction      = ctx['holonomy_correction'],
        holonomy_max_iter        = ctx['holonomy_max_iter'],
        holonomy_tol             = ctx['holonomy_tol'],
        use_wfc_cache            = ctx['use_wfc_cache'],
        n_workers                = 1,  # avoid nested parallelism inside a worker
        logger                   = logger,
        holonomy_min_plaquettes  = ctx.get('holonomy_min_plaquettes', 2),
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
    holonomy_min_plaquettes: int = 2,
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
        f"(max_iter={holonomy_max_iter}, tol={holonomy_tol:.1e}, "
        f"min_plaquettes={holonomy_min_plaquettes})"
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

    zones = build_zones(degen_at_k, neighbors, 2 * m.dimensions)

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
        'holonomy_correction':      holonomy_correction,
        'holonomy_max_iter':        holonomy_max_iter,
        'holonomy_tol':             holonomy_tol,
        'holonomy_min_plaquettes':  holonomy_min_plaquettes,
        'use_wfc_cache':            use_wfc_cache,
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

    # sort by zi so the final report appears in zone-list order
    summary_rows.sort(key=lambda r: r[0])

    # ==================================================================
    # Phase 5 — Final report and output file
    # ==================================================================
    logger.info()
    logger.info("\t****  Final Report  ****")
    logger.info()
    n_zones_out = len(summary_rows)
    logger.info(f"\tTotal zones processed             : {n_zones_out}")
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

    # Save degenzones.npy — each row: [zone_id, b1 ... bN (−1 padded), k-point]
    # Uses zone_set from position 8 of each summary row so splits are reflected.
    if summary_rows:
        max_n   = max(len(bands) for _, bands, *_ in summary_rows)
        records = []
        for zi, bands, _, _, _, _, _, _, zone_set, *_ in summary_rows:
            bs = list(bands) + [-1] * (max_n - len(bands))
            for nk in sorted(zone_set):
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
