"""
Band Classification Algorithm

This module houses the band classification algorithm, which determines the assignment
of points to respective bands for subsequent calculations. The algorithm employs graph theory
to determine the connectivity of points in the k-space, and then uses unsupervised machine learning
to classify the points into bands.

TODO:
  - Implement functionality to allow the algorithm to save intermediate results.
    Also, consider adding the ability to resume the algorithm or select the best result,
    which may differ from the last one.
  - Implement the algorithm to resolve forbidden paths
  - Implement the algorithm to address points connected to more than one band
"""


from __future__ import annotations
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from collections import deque
from heapq import heappush, heappop
import random
import string
import textwrap
from typing import Tuple, Union, Callable

import logging

from scipy.ndimage import sobel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.optimize import curve_fit, linear_sum_assignment

import numpy as np
import networkx as nx
import time

from berry import log
from berry._subroutines.contatempo import tempo

###########################################################################
# Type Definition
###########################################################################
Kpoint = int
Connection = float
Band = int
###########################################################################
# Constant Definition
###########################################################################
FORCED = 6              # signal_final / evaluate_result scale (force-filled by completeness pass)
CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

FORCED_CONTINUITY = 5   # correct_signalfinal / evaluate_point (energy-continuity) scale

# A direction "supports" a slot's attribution when the dot product between the
# cell's assigned band and the band assigned to the SAME slot at that neighbour
# is at least this. The dp distribution is strongly bimodal (~0 vs ~0.9+), so the
# exact value is not critical; 0.5 sits in the empty middle. Used by the post-loop
# dp-evidence passes (_dp_support_counts / _repair_dp_swaps / _flag_dp_unsupported).
DP_SUP_TOL = 0.5

# CERTAIN link: dp >= DP_CERTAIN between a state at k and a state at a neighbouring
# k-point identifies the continuation with near-certainty, and normalization makes
# that a hard guarantee (Bessel: sum_b dp^2 <= 1, so two partners >= 0.9 are
# impossible -- any threshold above 1/sqrt(2) has a UNIQUE partner -- and a certain
# partner caps every rival continuation at sqrt(1 - 0.81) ~ 0.44). Certain links
# therefore form a partial matching per (k, direction) and act as must-keep
# constraints -- but only LOCALLY: character transport is path-dependent (around a
# conical intersection it returns on the other sheet), so the transitive closure is
# NOT single-slottable and global constructions over it are meaningless (see
# _repair_certain_links for the local enforcement and _certain_partner_map for the
# map). DP_FORBID marks the other mode of the bimodal distribution: a same-slot
# continuation this weak is evidence AGAINST the attribution whenever any evidence
# exists at all.
DP_CERTAIN = 0.9
# dp-seam relocation (tier 2): threshold of the LOOSER partner map used for
# its acceptance accounting. Still unique-partner-valid (> 1/sqrt(2), Bessel).
# Counting split links at 0.9 only lets a patch "improve" by relocating its
# damage into merely sub-certain (0.8-0.9) seams; counting at 0.8 closes that
# blind spot (observed: two (14,17) patches zeroed their 0.9-links while
# growing the finer-grained refuted-edge count).
DP_FLOOD_SUPPORT = 0.8
DP_FORBID = 0.1
# Post-loop min-cut seam placement (S2 of the dp-seam correction plan,
# docs/cluster0_dp_seam_correction_proposal.md): after the greedy dp-seam
# relocation, place the label seam of each remaining refuted (s,t) pattern
# region at its exact optimum via s-t min-cut (seams are free inside
# near-degenerate corridors and on the certain forest's refusal cuts -- the
# physical branch cuts -- and pay the dp-certainty weight where they split a
# certain link). Every move is still vetoed by the exact global
# refuted-census guard, so this can only reduce the strict census.
DP_SEAM_MINCUT = True
# Near-degenerate patch relocation (post-loop): the tier-2 dp-seam passes exempt
# near-degenerate / DEGENERATE cells (their per-band label is gauge -- basisrotation
# territory). A small patch that attached to the WRONG near-degenerate band is
# therefore invisible to them even though its wavefunction evidence cleanly names
# the correct band. _relocate_degenerate_patches block-exchanges exactly those
# patches into the right slot (so the seam becomes dp-continuous), leaving the
# intra-subspace gauge to basisrotation. When True, an accepted exchange must also
# not raise the strict (0.9-map) refuted census on its border -- a non-regression
# floor that can only reject, never wrongly accept (guards the run-6 429->474
# border-refuted failure mode). Flip to False to relax to pure "seam halves, no new
# wall" acceptance.
DEGEN_FREEZE_REFUTED = True

# Seeded region-growing of oversized (multi-band fused) graph components,
# replacing the inner Louvain split (which optimises edge density with no
# notion of energy, dot-product quality or the one-state-per-k constraint,
# so its cuts through ambiguous crossing regions are arbitrary):
RG_SEED_DP = 0.9        # min same-band dp for a sovereign (seed) node
RG_DEFER_MARGIN = 0.05  # cost margin within which two labels' claims are contested
RG_BAND_PRESENCE = 0.5  # fraction of nks a raw band must hold to be a growth target

###########################################################################
# Parallelization helpers (Pool workers)
###########################################################################
# The worker function and its extra arguments are published here just before
# the Pool is forked, so the workers inherit them (and the large read-only
# arrays captured by the worker closures) via copy-on-write, instead of
# pickling those arrays on every task.
_PARALLEL_FN: Union[Callable, None] = None
_PARALLEL_ARGS: tuple = ()

def _parallel_dispatch(chunk):
    '''
    Module-level entry point executed by the Pool workers.
    It forwards each chunk to the currently published worker function.
    Being module-level, it (unlike a closure) is picklable so it can be
    dispatched to the Pool, while the actual worker function is read from the
    fork-inherited module global.
    '''
    return _PARALLEL_FN(chunk, *_PARALLEL_ARGS)


###########################################################################
# Quadratic extrapolation weights (cached)
###########################################################################
# The continuity gate predicts a band's energy one grid step ahead by fitting a
# quadratic to its trajectory at fixed integer offsets. For a given set of
# offsets that prediction is a constant linear combination of the sampled
# energies, so the weight row is cached per offset-set instead of refitting with
# np.polyfit on every (k-point, band, direction) -- ~66x faster at scale, and
# bit-identical to polyfit. At most a handful of distinct offset-sets occur.
_EXTRAP_WEIGHT_CACHE: dict = {}

def _extrap_weights(offsets: tuple) -> np.ndarray:
    '''
    Weight row ``w`` such that the least-squares quadratic through points at the
    given integer ``offsets`` (x-values), evaluated at x=+1, equals ``w @ energies``.
    '''
    w = _EXTRAP_WEIGHT_CACHE.get(offsets)
    if w is None:
        X = np.asarray(offsets, dtype=float)
        V = np.vander(X, 3)                                   # columns [x^2, x, 1]
        w = (np.vander(np.array([1.0]), 3) @ np.linalg.pinv(V)).ravel()
        _EXTRAP_WEIGHT_CACHE[offsets] = w
    return w


###########################################################################
# In-loop f(E) diagnostics (fE_debug)
###########################################################################
# The in-loop energy term (fit_energy/predict_energy + difference_energy inside
# COMPONENT.get_cluster_score) decides how a sample attaches to a cluster. To
# find out WHY f(E) helps or hurts, the hot path accumulates a handful of
# counters into this module-global dict. Every field is an int/float so two
# dicts merge by summation -- this is what lets the per-chunk counts collected
# inside the forked Pool workers be returned and summed in the parent without a
# shared-memory Manager (same copy-on-write discipline as _PARALLEL_FN). The
# residual buckets are counts of |E_pred - E_neighbour| / disp_scale falling in
# fixed ranges, so the distribution shape is mergeable too. _FE_DEBUG gates the
# accumulation so a production run pays nothing when it is off.
_FE_DEBUG: bool = False
_FE_RESID_EDGES = (0.5, 1.0, 2.0, 5.0)      # residual / disp_scale bucket edges

def _fe_stats_new() -> dict:
    '''A zeroed f(E)-diagnostics accumulator.'''
    return {
        'calls': 0,          # predict_energy invocations
        'fit': 0,            # ... that produced a fitted (used) energy
        'fallback': 0,       # ... that returned None (no continuous segment) -> nearest-E default
        'clip_hit': 0,       # fitted predictions clamped by the disp_scale window
        'trunc': 0,          # trajectories shortened by the cliff-break
        'kept_len_sum': 0,   # sum of kept-trajectory lengths (for a mean)
        'brk_ejump': 0,      # cliff-break caused by an energy step > accept_E
        'brk_band': 0,       # cliff-break caused by a raw-band-index change
        'pair_n': 0,         # scored (k-edge, neighbour) pairs
        'conn_sum': 0.0,     # sum of alpha*connection contributions
        'eng_sum': 0.0,      # sum of (1-alpha)*energy_val contributions
        'resid_n': 0,        # predictions with a measurable residual
        'resid_buckets': [0, 0, 0, 0, 0],   # |resid|/disp_scale in [<0.5,<1,<2,<5,>=5]
    }

def _fe_stats_merge(dst: dict, src: dict) -> None:
    '''Sum-merge ``src`` into ``dst`` (both produced by _fe_stats_new).'''
    for key, val in src.items():
        if key == 'resid_buckets':
            for i in range(len(val)):
                dst[key][i] += val[i]
        else:
            dst[key] += val

# Per-worker live accumulator. Reset at the start of each evaluate_sample chunk,
# its delta returned to the parent at the end of the chunk.
_FE_STATS: dict = _fe_stats_new()

def _fe_resid_bucket(resid: float, disp_scale: float) -> int:
    '''Index of the residual/disp_scale bucket for ``resid``.'''
    if not disp_scale or disp_scale <= 0:
        return len(_FE_RESID_EDGES)
    r = abs(resid) / disp_scale
    for i, edge in enumerate(_FE_RESID_EDGES):
        if r < edge:
            return i
    return len(_FE_RESID_EDGES)


def _energy_continuity_score(Enew: float, Ecand: float,
                             accept_E: float = None, disp_scale: float = None) -> float:
    '''
    Directional energy-continuity score in [0, 1] between a predicted energy
    ``Enew`` (extrapolated along the line joining the reference and the new
    point) and a specific candidate energy ``Ecand`` (the neighbour's band).

    This REPLACES the old ``min(|Enew - E_all_bands|) / |Enew - Ecand|`` objective,
    which scored how close ``Ecand`` was to being the *globally nearest* band --
    a non-directional, scale-blind measure that (a) rewarded the energetically
    nearest band rather than the continuous one (wrong at crossings) and (b)
    could not forbid a gross inter-group jump, since it only ever looked at the
    nearest of all bands.

    New behaviour:
      * Gross-jump gate: if |Enew - Ecand| exceeds ``accept_E`` (a LOOSE
        gross-jump scale ~ 3*disp_scale, wider than any legitimate one-step
        dispersion but far below a genuine inter-band wall), return 0.0 -- the
        candidate is on the far side of an energy wall and must never be joined.
        This is deliberately NOT keyed to gap_scale (the SOC-pair splitting),
        which is what shattered the graph in the e90a33a regression.
      * Otherwise a smooth, scale-aware match: 1.0 at a perfect match, decaying
        on the ``disp_scale`` (local one-step dispersion) scale.

    With ``accept_E``/``disp_scale`` unset this reduces to a plain closeness in
    [0, 1] so callers that do not pass the scales keep working.
    '''
    resid = abs(Enew - Ecand)
    if accept_E is not None and resid > accept_E:
        return 0.0                                   # gross-jump veto (loose, accept_E)
    sigma = disp_scale if (disp_scale and disp_scale > 0) else accept_E
    if not sigma or sigma <= 0:
        return 1.0 if resid == 0 else 0.0
    return float(sigma / (sigma + resid))            # continuity-residual, scale-aware


def _local_scales(accept_E: float, disp_scale: float,
                  local_gap: float, band_disp: float) -> Tuple[float, float]:
    '''
    Sharpen the global energy scales to the LOCAL band neighbourhood, never
    loosening them. The global ``accept_E`` (3x the p99.9 dispersion of the
    whole spectrum) is wider than the inter-band gap of a crowded manifold, so
    a one-band jump passes the gross-jump gate and still scores "smooth"
    against the global ``disp_scale`` -- the energy machinery goes blind
    exactly where the dot product is ambiguous too (phosphorene bands 14-24:
    gaps 0.02-0.035 Ry vs accept_E 0.075 Ry).

    Where the local adjacent-band gap RESOLVES the band's own dispersion
    (``local_gap > 2*band_disp``), gate at half the local gap -- floored at
    one per-band p95 step so a legitimate prediction residual is never cut --
    and decay the score on the band's own dispersion. Where the gap does NOT
    resolve the dispersion (near-degenerate / unresolved crowding), return the
    global scales unchanged: energy cannot referee there and a sharp gate
    would only manufacture mistakes (the e90a33a lesson); dot-product evidence
    has to decide.
    '''
    if (accept_E is None or local_gap is None or band_disp is None
            or not np.isfinite(local_gap) or band_disp <= 0
            or local_gap <= 2.0 * band_disp):
        return accept_E, disp_scale
    gate = min(accept_E, max(0.5 * local_gap, band_disp))
    disp = min(disp_scale, band_disp) if (disp_scale and disp_scale > 0) else band_disp
    return gate, disp


# Evidence weights of the dp-aware repair / completeness assignment: same blend
# as obtain_output's conflict solver (0.6*dot-product + 0.4*energy). The energy
# term ranks candidates inside the local gate; the dp term (mean overlap with
# the slot's trusted neighbours) picks among energy-admissible candidates,
# which energy alone cannot tell apart in a crowded manifold.
REPAIR_W_DP, REPAIR_W_E = 0.6, 0.4


EVALUATE_RESULT_HEADER = ['NOT', 'MIS', 'DEG', 'PMI', 'PCO', 'COR', 'FOR']
VALIDATE_RESULT_HEADER = ['NOT', 'MIS', 'DEG', 'OTH', 'COR', 'FOR']

###########################################################################
# Column legends (shared by every report table)
###########################################################################
# Short, human-readable meaning of each table column. Keyed by the column
# abbreviation so a legend can be built that always matches the columns shown.
COLUMN_LEGEND = {
    'NOT':    'not solved (no band attributed)',
    'MIS':    'mistake (energy break and/or dot product <= 0.2)',
    'DEG':    'degenerate (needs basis rotation)',
    'PMI':    'potential mistake (dot product <= 0.8)',
    'PCO':    'potential correct (0.8 < dot product < 0.9)',
    'OTH':    'other (partial energy continuity)',
    'COR':    'correct (energy-continuous, dot product > 0.9)',
    'FOR':    'forced (force-filled by the completeness pass; not a genuine solve)',
    'Score':  'mean dot product of the band over its genuine points, excluding benign force-fills (1 is best)',
    'Failed': 'NOT + MIS points (energy break / low overlap)',
    'Degen':  'degenerate points (need basis rotation)',
    'Forced': 'force-filled points (not a genuine solve)',
    'Benign': 'force-fills inside a near-degenerate doublet (gauge-level relabeling; do not affect usability)',
    'Susp':   'force-fills across a real energy gap (suspect; downgrade the band)',
    'Silent': 'validated cells sitting on a gross energy jump (> accept_E) to a same-slot neighbour -- the cliffs visible in a band plot',
    'Cliff':  'raw-band switch edges surviving on a jump above the LOCAL continuity gate (sub-accept_E cliffs the Silent column is too loose to see); a band with any is downgraded',
    'dpBreak': 'raw-band switch edges surviving against a CERTAIN dp link -- the energy-quiet permutation cycles the rotation pass could not unwind (a wavefunction discontinuity invisible to every energy column); a band with any is downgraded',
    'Status': 'CLEAN (pristine, nothing flagged) / USABLE (no genuine errors; may carry benign force-fills or a sub-pristine score) / ROTATE (needs basis rotation) / CHECK (a suspect force-fill across a real gap, a silent cliff, a sub-gate cliff, or a dp break -> verify) / FAIL (genuine energy break, or many suspect/silent/cliff/dp-break cells)',
}

# All report tables and their surrounding content use these two indent levels:
#   TITLE_INDENT  for the '====== ... ======' section header,
#   BODY_INDENT   for legends, table headers and table rows.
TITLE_INDENT = '\t'
BODY_INDENT = '\t\t'

def _format_legend(columns: list, indent: str = BODY_INDENT) -> str:
    '''
    A wrapped legend block explaining every column in ``columns``, indented to
    line up with the table body. Columns with no known meaning are skipped.
    '''
    items = [f"{c} = {COLUMN_LEGEND[c]}" for c in columns if c in COLUMN_LEGEND]
    return '\n' + textwrap.fill(
        ' | '.join(items),
        width=100,
        initial_indent=indent + 'Legend: ',
        subsequent_indent=indent + '        ',
    ) + '\n'


def evaluate_result(values: Union[list[Connection], np.ndarray]) -> int:
    '''
    This function attributes the correspondent signal using
    the dot product between each neighbor.

    Parameters
        values: array_like
            It is an array that contains the dot product
            between the k point and all neighbors.

    Returns
        signal: int
            C -> Mean connection of each k point
            Value :                              Description
            0     :                        The point is not solved
            1     :  MISTAKE               C <= 0.2
            2     :  DEGENERATE            It is a degenerate point.
            3     :  POTENTIAL_MISTAKE     C <= 0.8
            4     :  POTENTIAL_CORRECT     0.8 < C < 0.9
            5     :  CORRECT               C > 0.9
            6     :  FORCED                (assigned by the completeness pass, not here)
    '''

    TOL = 0.9       # Tolerance for CORRECT output
    TOL_DEG = 0.8   # Tolerance for POTENTIAL_CORRECT output
    TOL_MIN = 0.2   # Tolerance for POTENTIAL_MISTAKE output

    value = np.mean(values) # Mean conection of each k point

    if value > TOL:
        return CORRECT

    if value > TOL_DEG:
        return POTENTIAL_CORRECT

    if value > TOL_MIN:
        return POTENTIAL_MISTAKE

    return MISTAKE

def evaluate_point(dimension:int, k: Kpoint, bn: Band, k_index: np.ndarray, k_matrix: np.ndarray,
                   bands: np.ndarray, energies: np.ndarray,
                   accept_E: float = None, disp_scale: float = None,
                   local_gap: np.ndarray = None, band_disp: np.ndarray = None) -> Tuple[int, list[int]]:
    '''
    Assign a signal value depending on energy continuity.

    Parameters
        dimension: int
            Dimension of the problem.
        k: Kpoint
            Integer that index the k point on analysis.
        bn: Band
            Integer that index the band number on analysis.
        k_index: array_like
            An array that contains the indices of each k point on the k-space matrix.
        k_matrix: array_like
            An array with the shape of the k-space.
            It contains the value of each k point in their corresponding position.
        bands: array_like
            An array with the information of current solution of band clustering.
        energies: array_like
            It contais the energy value for each k point.
    
    Returns
        (signal, scores): Tuple[int, list[int]]
            scores: list[int]
                Signals whether energy continuity holds along each of the
                2*dimension grid directions (e.g. [Down, Right, Up, Left] in 2D).
                    1 --- This direction preserves energy continuity.
                    0 --- This direction does not preserve energy continuity.
            N -> Number of directions with energy continuity (max = 2*dimension).
            signal: int
                Value :                              Description
                0     :                        The point is not solved
                1     :  MISTAKE               N = 0
                2     :  DEGENERATE            It is a degenerate point.
                3     :  OTHER                 0 < N < 2*dimension
                4     :  CORRECT               N = 2*dimension
    '''
    
    CORRECT = 4
    MISTAKE = 1
    OTHER = 3

    # Continuity threshold, re-tuned for the directional continuity-residual score
    # (difference_energy now returns disp_scale/(disp_scale+resid), not the old
    # min/delta ratio). A direction "preserves continuity" when its prediction
    # error is within ~one local dispersion step:
    #     score = disp_scale/(disp_scale + resid) >= 0.5  <=>  resid <= disp_scale.
    # The old TOL=0.9 was calibrated to the min/delta ratio and is far too strict
    # here (it would demand resid < disp_scale/9 and flag smooth bands / legitimate
    # anticrossings). Gross inter-group jumps are vetoed by the gate (resid >
    # accept_E -> score 0) regardless of this threshold. This is the primary tunable
    # knob: lower it toward 1/3 (resid <= 2*disp_scale) if smooth bands over-flag.
    # If no scales are passed (unwired path) keep the original 0.9 so behaviour is
    # unchanged for any caller that does not supply disp_scale.
    TOL = 0.5 if disp_scale is not None else 0.9
    N = 4                           # Number of points to fit the curve
    N_NEIGS = 2 * dimension         # Number of neighbors to consider the continuity

    mach_bn = bands[k, bn]          # original band
    if mach_bn < 0:
        # Unattributed cell: nothing to validate (and eigenvalues[k, -1] would
        # silently read the LAST band). Grade it a mistake with no continuous
        # direction, matching the guard in _extrapolate_energy.
        return MISTAKE, np.zeros(N_NEIGS, dtype=int)

    if dimension == 1:
        ik = k_index[k]             # k point index on k-space
    elif dimension == 2:
        ik, jk = k_index[k]         # k point indices on k-space
    else:
        ik, jk, kk = k_index[k]     # k point indices on k-space

    Ek = energies[k, mach_bn]       # k point's Energy value

    # Sharpen the gate/decay scales to the assigned band's local neighbourhood
    # (half the local adjacent gap where it resolves the band's dispersion,
    # unchanged global scales elsewhere -- see _local_scales).
    if local_gap is not None and band_disp is not None and mach_bn >= 0:
        accept_E, disp_scale = _local_scales(accept_E, disp_scale,
                                             float(local_gap[k, mach_bn]),
                                             float(band_disp[mach_bn]))

    def difference_energy(Ek: float, Enew: float) -> float:
        '''
        Directional energy-continuity score in [0, 1] between the assigned energy
        ``Ek`` and the energy ``Enew`` extrapolated along ONE direction (the line
        joining this k-point to that direction's neighbours). The direction is
        "continuous" when the band's own energy matches the prediction from that
        direction's trajectory, within the local dispersion ``disp_scale``, with a
        gross-jump veto at ``accept_E`` for inter-group walls.

        This replaces the old ``min(|Enew - E_all_bands|) / |Enew - Ek|`` ratio,
        which scored "is Ek the globally nearest band" -- non-directional and
        unreliable once the gap collapses below disp_scale (SOC/Kramers manifold),
        where the nearest band is no longer the continuous one. See the shared
        ``_energy_continuity_score`` for the gate/decay definition; ``accept_E``
        and ``disp_scale`` are threaded in from the caller.
        '''
        return _energy_continuity_score(float(Ek), float(Enew),
                                        accept_E=accept_E, disp_scale=disp_scale)


    if dimension == 1:
        directions = np.array([[1], [-1]])                     # Right, Left
        energy_vals = []

        ###########################################################################
        # Calculate the score for each direction
        ###########################################################################

        for direction in directions:
            # Iterates each direction and obtain N points to be used for fit the curve
            n = np.repeat(np.arange(1,N+1),2).reshape(N,2)
            kn_index = n*direction + np.array([ik])
            i = kn_index[:, 0]
            i = i[i >= 0]
            i = i[i < k_matrix.shape[0]]

            ks = k_matrix[i] if len(i) > 0 else []               # Identify the N k points
            if len(ks) > 0:
                # Skip unattributed (-1) cells: eigenvalues[k, -1] would silently
                # read the LAST band's energy into the fit (same trap guarded in
                # _extrapolate_energy). Order stays nearest-first.
                sel = bands[ks, bn] >= 0
                ks = ks[sel]
                i = i[sel]
            if len(ks) == 0:
                # The direction in analysis does not have points
                energy_vals.append(1)
                continue
            if len(ks) <= 3:
                # If there are not enough points to fit the curve it is used the Energy of the nearest neighbor
                Eneig = energies[ks[0], bands[ks[0], bn]]
                energy_vals.append(difference_energy(Ek, Eneig))
                continue

            k_bands = bands[ks, bn]
            Es = energies[ks, k_bands]
            X = i
            new_x = ik
            pol = lambda x, a, b, c: a*x**2 + b*x + c           # Second order polynomial
            popt, pcov = curve_fit(pol, X, Es)                  # Curve fitting
            Enew = pol(new_x, *popt)                            # Obtain Energy value
            energy_vals.append(difference_energy(Ek, Enew))     # Calculate score

    elif dimension == 2:

        directions = np.array([[1,0], [0,1], [-1,0], [0,-1]])       # Down, Right, Up, Left
        energy_vals = []

        ###########################################################################
        # Calculate the score for each direction
        ###########################################################################

        for direction in directions:
            # Iterates each direction and obtain N points to be used for fit the curve
            n = np.repeat(np.arange(1,N+1),2).reshape(N,2)
            kn_index = n*direction + np.array([ik, jk])
            i, j = kn_index[:, 0], kn_index[:, 1]   # Selects the indices of these N points
            flag = len(np.unique(i)) > 1            # Necessary to identify which will be the direction of the fit
            if flag:
                i = i[i >= 0]
                i = i[i < k_matrix.shape[0]]
                j = np.full(len(i), j[0])
            else:
                j = j[j >= 0]
                j = j[j < k_matrix.shape[1]]
                i = np.full(len(j), i[0])
            
            ks = k_matrix[i, j] if len(i) > 0 else []   # Identify the N k points
            if len(ks) > 0:
                # Skip unattributed (-1) cells (see the 1D branch).
                sel = bands[ks, bn] >= 0
                ks = ks[sel]
                i = i[sel]
                j = j[sel]
            if len(ks) == 0:
                # The direction in analysis does not have points
                energy_vals.append(1)
                continue
            if len(ks) <= 3:
                # If there are not enough points to fit the curve it is used the Energy of the nearest neighbor
                Eneig = energies[ks[0], bands[ks[0], bn]]
                energy_vals.append(difference_energy(Ek, Eneig))
                continue

            k_bands = bands[ks, bn]
            Es = energies[ks, k_bands]
            X = i if flag else j
            new_x = ik if flag else jk
            pol = lambda x, a, b, c: a*x**2 + b*x + c           # Second order polynomial
            popt, pcov = curve_fit(pol, X, Es)                  # Curve fitting
            Enew = pol(new_x, *popt)                            # Obtain Energy value
            energy_vals.append(difference_energy(Ek, Enew))     # Calculate score
    
    else:
            
        directions = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]) # Down, Right, Up, Left, Front, Back
        energy_vals = []

        ###########################################################################
        # Calculate the score for each direction
        ###########################################################################

        for direction in directions:
            # Iterates each direction and obtain N points to be used for fit the curve
            n = np.repeat(np.arange(1,N+1),3).reshape(N,3)
            kn_index = n*direction + np.array([ik, jk, kk])
            # k3 (not k): the k-point argument must not be shadowed
            i, j, k3 = kn_index[:, 0], kn_index[:, 1], kn_index[:, 2]
            flag_i = len(np.unique(i)) > 1
            flag_j = len(np.unique(j)) > 1

            if flag_i:
                i = i[i >= 0]
                i = i[i < k_matrix.shape[0]]
                j = np.full(len(i), j[0])
                k3 = np.full(len(i), k3[0])
            elif flag_j:
                j = j[j >= 0]
                j = j[j < k_matrix.shape[1]]
                i = np.full(len(j), i[0])
                k3 = np.full(len(j), k3[0])
            else:
                k3 = k3[k3 >= 0]
                k3 = k3[k3 < k_matrix.shape[2]]
                i = np.full(len(k3), i[0])
                j = np.full(len(k3), j[0])

            ks = k_matrix[i, j, k3] if len(i) > 0 else []   # Identify the N k points
            if len(ks) > 0:
                # Skip unattributed (-1) cells (see the 1D branch).
                sel = bands[ks, bn] >= 0
                ks = ks[sel]
                i = i[sel]
                j = j[sel]
                k3 = k3[sel]
            if len(ks) == 0:
                # The direction in analysis does not have points
                energy_vals.append(1)
                continue
            if len(ks) <= 3:
                # If there are not enough points to fit the curve it is used the Energy of the nearest neighbor
                Eneig = energies[ks[0], bands[ks[0], bn]]
                energy_vals.append(difference_energy(Ek, Eneig))
                continue

            k_bands = bands[ks, bn]
            Es = energies[ks, k_bands]
            X = i if flag_i else j if flag_j else k3
            new_x = ik if flag_i else jk if flag_j else kk
            pol = lambda x, a, b, c: a*x**2 + b*x + c           # Second order polynomial
            popt, pcov = curve_fit(pol, X, Es)                  # Curve fitting
            Enew = pol(new_x, *popt)                            # Obtain Energy value
            energy_vals.append(difference_energy(Ek, Enew))     # Calculate score

    
    energy_vals = np.array(energy_vals)
    scores = (energy_vals > TOL)*1  # Verification energy continuity on each direction
    score = np.sum(scores)          # Counting how many directions preserves energy continuity
    
    if score == N_NEIGS:
        return CORRECT, scores
    if score == 0:
        return MISTAKE, scores
    return OTHER, scores


class MATERIAL:
    '''
    This object contains all information about the material that
    will be used to solve their bands' problem.

    Atributes
        d : int
            Dimension of the problem.
        nk_i : list[int]
            It contains the number of k points on each direction. 
        nbnb : int
            Total number of bands.
        total_bands : int
            Total number of bands.
        nks : int
            Total number of k points.
        eigenvalues : array_like
            It contains the energy value for each k point.
        connections : array_like
            The dot product information between k points.
        neighbors : array_like
            An array with the information about which are the neighbors of each k point.
        vectors : array_like
            Each k point in the vector representation on k-space.
        n_process : int
            Number of processes to use.
        bands_final : array_like
            An array with final result of bands attribution.
        signal_final : array_like
            Contains the resulting signal for each k point.
        final_score : array_like
            It contains the result score for each band.

    Methods
        solve() : None
            This method is the main algorithm which iterates between solutions
                trying to find the best result for the material.
        make_vectors() : None
            It transforms the information into more convenient data structures.
        make_BandsEnergy() : array_like
            It sets the energy information in more convinient data structure
        make_kpointsIndex() : None
            It computes the indices of each k point in their correspondence in k-space.
        make_connections() : None
            This function evaluates the connection between each k point, and adds an edge
                to the graph if its connection is greater than a tolerance value (tol).
        get_neigs() : list[Kpoint]
            Obtain the i's neighbors.
        find_path() : bool
            Verify if exist a path between two k points inside the graph.
        parallelize() : array_like
            Create processes for some function f over an iterator.
        get_components() : None
            Tt detects components well constructed.
        obtain_output() : None
            This function prepares the final data structures
                that are essential to other programs.
        print_report() : (str, np.ndarray)
            Shows on screen the report for each band.
        correct_signal() : None
            This function evaluates the k-point signal calculated on previous analysis and attributes
                a new signal value depending only on energy continuity.
    '''     
    def __init__(self, dimensions:int, nk_i: list[int], nbnd: int, nks: int, eigenvalues: np.ndarray,
                 connections: np.ndarray, neighbors: np.ndarray, logger: log, min_band: int,
                 max_band: int = -1, n_process: int=1) -> None:
        '''
        Initialize the object.

        Parameters
            d : int
                Dimension of the problem.
            nk_i : list[int]
                It contains the number of k points on each direction. 
            nbnb : int
                Total number of bands.
            total_bands : int
                Total number of bands.
            nks : int
                Total number of k points.
            eigenvalues : array_like
                It contains the energy value for each k point.
            connections : array_like
                The dot product information between k points.
            neighbors : array_like
                An array with the information about which are the neighbors of each k point.
            vectors : array_like
                Each k point in the vector representation on k-space.
            min_band : int
                First band of the clustering window (the preprocessing cutoff:
                the dp array's band axis starts here).
            max_band : int
                Last band of the clustering window (absolute index; -1 = all
                bands up to nbnd-1).
            n_process : int
                Number of processes to use.
        '''
        self.dimensions = dimensions
        self.nkx, self.nky, self.nkz = nk_i
        # Band window [min_band, max_band] (absolute band indices; max_band == -1
        # means every band up to nbnd-1). Everything downstream -- eigenvalues,
        # the dp array (whose band axis already starts at min_band, the
        # preprocessing cutoff) and every band-indexed structure -- is sliced to
        # this window HERE, so the whole solver sees one consistent band count.
        nb = (max_band + 1 - min_band) if max_band != -1 else (nbnd - min_band)
        if nb <= 0:
            raise ValueError(f'empty band window: min_band={min_band}, max_band={max_band}')
        if connections.shape[2] < nb:
            raise ValueError(f'dp.npy holds {connections.shape[2]} band(s) but the requested '
                             f'window [{min_band}, {max_band}] needs {nb} -- re-run dot with '
                             f'more bands or lower max_band')
        self.nbnd = nb
        self.total_bands = nb
        self.nks = nks
        self.eigenvalues = eigenvalues[:, min_band:min_band + nb]
        self.connections = connections[:, :, :nb, :nb] if connections.shape[2] > nb else connections
        self.neighbors = neighbors
        self.number_neighbors = self.dimensions * 2
        self.vectors = None
        self.n_process = n_process
        self.logger = logger
        self.min_band = min_band

    def make_BandsEnergy(self) -> np.ndarray:
        '''
        It sets the energy information in more convenient data structure
        
        Parameters
            None
        
        Returns
            BandsEnergy : array_like
                An array with the information about each energy value on k-space.
        '''
        # BandsEnergy[bn, ikx(, iky(, ikz))] = eigenvalues[k, bn] with the k-points
        # laid out kx-fastest (k = ((ikz*nky) + iky)*nkx + ikx -- the same order
        # make_kpointsIndex uses), which is a plain reshape + axis reversal.
        if self.dimensions == 1:
            return self.eigenvalues.T.copy()

        if self.dimensions == 2:
            return self.eigenvalues.T.reshape(
                self.nbnd, self.nky, self.nkx).transpose(0, 2, 1).copy()

        return self.eigenvalues.T.reshape(
            self.nbnd, self.nkz, self.nky, self.nkx).transpose(0, 3, 2, 1).copy()

    def make_kpointsIndex(self) -> None:
        '''
        It computes the indices of each k point in their correspondence in k-space.
        '''
        if self.dimensions == 1:
            self.matrix = np.arange(self.nks)
            self.kpoints_index = np.arange(self.nks)
            return
    
        if self.dimensions == 2:
            My, Mx = np.meshgrid(np.arange(self.nky), np.arange(self.nkx))
            self.matrix = My * self.nkx + Mx
            counts = np.arange(self.nks)
            self.kpoints_index = np.stack([counts % self.nkx, counts//self.nkx],
                                            axis=1)
            return
        
        self.matrix = np.empty((self.nkx, self.nky, self.nkz), int)
        self.kpoints_index = np.empty((self.nks, 3), int)
        count = -1
        for k in range(self.nkz):
            for j in range(self.nky):
                for i in range(self.nkx):
                    count += 1
                    self.matrix[i, j, k] = count
                    self.kpoints_index[count] = [i, j, k]

    def make_vectors(self) -> None:
        '''
        It transforms the information into more convenient data structures.
        The band window (min_band/max_band) is fixed at construction time
        (see ``MATERIAL.__init__``); this method only derives structures.

        Result
            self.vectors: [kx_b, ky_b, kz_b, E_b]
                k = (kx, ky, k_z)_b: k point
                b: band number
            self.degenerados: It marks the degenerate points
            self.GRPAH: It is a graph in which each node represents a vector.
            self.energies: It contains the energy values for each band distributed
                        in a matrix.
        '''
        process_name = '\tMaking Vectors'
        self.logger.percent_complete(0, 100, title=process_name)

        ###########################################################################
        # Compute the auxiliar information
        ###########################################################################
        self.GRAPH = nx.Graph()     # Create the initail Graph
        nbnd = self.nbnd
        self.make_kpointsIndex()
        energies = self.make_BandsEnergy()
        self.logger.percent_complete(20, 100, title=process_name)

        ###########################################################################
        # Compute the vector representation of each k point
        ###########################################################################
        n_vectors = nbnd*self.nks
        ik = np.tile(self.kpoints_index[:, 0], nbnd) if self.dimensions > 1 else np.tile(self.kpoints_index, nbnd)

        stack_aux = [ik]

        if self.dimensions >= 2:
            jk = np.tile(self.kpoints_index[:, 1], nbnd)
            stack_aux.append(jk)
        if self.dimensions == 3:
            kk = np.tile(self.kpoints_index[:, 2], nbnd)
            stack_aux.append(kk)

        bands = np.arange(0 , nbnd)
        eigenvalues = self.eigenvalues[:, bands].T.reshape(n_vectors)
        stack_aux.append(eigenvalues)

        self.vectors = np.stack(stack_aux, axis=1)
        self.logger.percent_complete(100, 100, title=process_name)

        self.GRAPH.add_nodes_from(np.arange(n_vectors))     # Add the nodes, each node represent a k point
        
        ###########################################################################
        # Verify if any k point is a degenerate point
        ###########################################################################
        self.degenerados = []
        def obtain_degenerates(vectors: np.ndarray) -> list[Kpoint]:
            '''
            Find all degenerate k points present on vectors.

            Parameters
                vectors : array_like
                    An array with vector representation of k points.
            
            Returns
                degenerates : list[Kpoint]
                    It contains the degenerate points found.
            '''
            degenerates = []
            for i, v in vectors:
                degenerado = np.where(np.all(np.isclose(self.vectors[i+1:]-v, 0),
                                    axis=1))[0] # Verify which points have numerically the same value
                if len(degenerado) > 0:
                    self.logger.debug(f'Found degenerete point for {i}')
                    degenerates += [[i, d+i+1] for d in degenerado]
            return degenerates

        # Parallelize the verification process
        self.degenerados = self.parallelize('\tFinding degenerate points', obtain_degenerates, enumerate(self.vectors))

        if len(self.degenerados) > 0:
            self.logger.debug('\tDegenerate Points: ')
            for d in self.degenerados:
                self.logger.debug(f'\t\t{d}')

        self.ENERGIES = energies

        ###########################################################################
        # Unit-free energy scales for the post-solve repair pass, so no absolute
        # energy threshold has to be hand-picked:
        #   gap_scale  - median spacing between adjacent bands.
        #   disp_scale - 99.9th-percentile single-step dispersion |E(k+1)-E(k)|
        #                (a robust "max" that ignores a rare outlier step). The
        #                gross-jump guard is a multiple of this so a legitimate
        #                one-step continuation (even in steep bands) is never cut.
        # NOTE: gap_scale collapses to the SOC splitting in noncolinear systems
        # (Kramers pairs), so it must NEVER be used as a sharp penalty width —
        # only as a loose floor (see the e90a33a in-solve gate regression).
        ###########################################################################
        gaps = np.diff(np.sort(self.eigenvalues, axis=1), axis=1)
        gaps = gaps[gaps > 0]
        self.gap_scale = float(np.median(gaps)) if gaps.size else 1.0

        disp_values = []                                              # per-direction |dE| arrays, band axis kept
        neigh = np.asarray(self.neighbors)
        for j in range(neigh.shape[1]):
            nb = neigh[:, j]
            valid = nb != -1
            if np.any(valid):
                d = np.abs(self.eigenvalues[nb[valid]] - self.eigenvalues[valid])
                if d.size:
                    disp_values.append(d)
        disp_all = np.concatenate(disp_values, axis=0).ravel() if disp_values else np.array([])
        disp_scale = float(np.percentile(disp_all, 99.9)) if disp_all.size else 0.0
        self.disp_scale = disp_scale if disp_scale > 0 else self.gap_scale
        self.sigma_min = 0.25 * self.gap_scale                          # min local-spread scale for _extrapolate_energy
        self.accept_E = max(3.0 * self.disp_scale, 0.5 * self.gap_scale)  # gross-jump guard for repair/forced fills

        ###########################################################################
        # LOCAL energy scales (see _local_scales): the global accept_E/disp_scale
        # above are calibrated by the steepest bands / widest gaps of the whole
        # spectrum and are wider than the inter-band gaps of a crowded manifold,
        # where a one-band jump then passes every energy check. Per-band p95
        # single-step dispersion + per-(k, band) adjacent-band gap let the gates
        # sharpen exactly where the local gap resolves the local dispersion;
        # near-degenerate regions keep the loose global scales (dp referees).
        ###########################################################################
        if disp_values:
            band_steps = np.concatenate(disp_values, axis=0)            # (n_pairs, nbnd)
            self.band_disp = np.maximum(np.percentile(band_steps, 95, axis=0), 1e-12)
        else:
            self.band_disp = np.full(self.eigenvalues.shape[1], self.disp_scale)
        dE_adj = np.abs(np.diff(self.eigenvalues, axis=1))
        self.local_gap = np.full(self.eigenvalues.shape, np.inf)
        self.local_gap[:, :-1] = dE_adj
        self.local_gap[:, 1:] = np.minimum(self.local_gap[:, 1:], dE_adj)
        _gate = np.minimum(self.accept_E,
                           np.maximum(0.5 * self.local_gap, self.band_disp[None, :]))
        _resolved = self.local_gap > 2.0 * self.band_disp[None, :]
        # continuity_gate: per-(k, band) energy gate used ONLY by the cliff
        # machinery (_cliff_seam_edges detection, cliff-patch repair, the final
        # _verify_no_cliffs invariant, and the report's Cliff column). It is the
        # SAME sharpened value _local_scales() returns, promoted to an array:
        # where the local adjacent-band gap resolves the band's own dispersion
        # (local_gap > 2*band_disp) it gates at half the local gap, and where it
        # does not (near-degenerate crowding) it falls back to the LOOSE accept_E,
        # so genuine crossings are never flagged. accept_E, _energy_continuity_score
        # and the in-loop clustering score are deliberately left untouched (a tight
        # gate on graph edges is the e90a33a shatter trap); the cliff pass is
        # post-loop and confined to raw-band SWITCH edges.
        self.continuity_gate = np.where(_resolved, _gate, self.accept_E)
        n_sharp = int(np.sum(_resolved & (_gate < self.accept_E)))
        self.logger.info(f'{BODY_INDENT}Local energy gates: {n_sharp}/{self.local_gap.size} '
                         f'(k, band) cells sharpen below the global accept_E '
                         f'(median sharpened gate '
                         f'{np.median(_gate[_resolved & (_gate < self.accept_E)]):.4g} Ry)'
                         if n_sharp else
                         f'{BODY_INDENT}Local energy gates: no cell resolves its local gap '
                         f'-- global scales everywhere')

        ###########################################################################
        # f(E) diagnostics, Tier 3: report the energy scales f(E) depends on, once.
        # If gap_scale has collapsed relative to the typical single-step dispersion
        # (the SOC/Kramers-pair trap), accept_E's 0.5*gap_scale floor and the
        # disp_scale clip are mis-scaled -- flag it so a bad f(E) result can be
        # traced to the scales rather than the logic.
        ###########################################################################
        if disp_all.size:
            d_med = float(np.median(disp_all))
            d_p95 = float(np.percentile(disp_all, 95))
            d_max = float(disp_all.max())
        else:
            d_med = d_p95 = d_max = 0.0
        gap_disp_ratio = (self.gap_scale / self.disp_scale) if self.disp_scale else float('inf')
        self.logger.info(f'{BODY_INDENT}f(E) scales (Ry): gap_scale={self.gap_scale:.4g}  '
                         f'disp_scale(p99.9)={self.disp_scale:.4g}  accept_E={self.accept_E:.4g}  '
                         f'sigma_min={self.sigma_min:.4g}')
        self.logger.info(f'{BODY_INDENT}f(E) single-step dispersion |E(k+1)-E(k)| (Ry): '
                         f'median={d_med:.4g}  p95={d_p95:.4g}  max={d_max:.4g}  '
                         f'(gap/disp={gap_disp_ratio:.3g})')
        # fe_eweight: down-weight the f(E) score term when the gap has collapsed
        # below the single-step dispersion. In that regime (SOC/Kramers manifold,
        # gap_disp_ratio < 1) energy continuity simply CANNOT separate adjacent
        # bands -- the prediction error (disp_scale) is larger than the gap, so
        # f(E) is noise. We therefore lean on the dot product and keep f(E) only
        # as a gross-jump gate (see ENERGY_GATE below). Clipped to [0.1, 1.0] so
        # f(E) is never fully zeroed (it still provides the gate) and never
        # over-trusted. When bands are well separated (ratio >= 1) this is 1.0
        # and the original f(E) weight is recovered exactly.
        self.fe_eweight = float(np.clip(gap_disp_ratio, 0.1, 1.0)) if np.isfinite(gap_disp_ratio) else 1.0
        if gap_disp_ratio < 1.0:
            self.logger.warning(f'{BODY_INDENT}f(E) WARNING: gap_scale < disp_scale -- the band gap '
                                f'looks collapsed (SOC/Kramers degeneracy?); down-weighting the f(E) '
                                f'score term to fe_eweight={self.fe_eweight:.3g} and relying on the '
                                f'dot product, with f(E) kept as a gross-jump gate at accept_E={self.accept_E:.4g}.')

        self.bands_final = np.full((self.nks, self.total_bands), -1, dtype=int)

    def get_neigs(self, i: Kpoint) -> list[Kpoint]:
        '''
        Obtain the i's neighbors

        Parameters
            i : Kpoint
                The node index.
        
        Returns
            neighbors : list[Kpoint]
                List with the nodes that are neighbors of the node i.
        '''
        return list(self.GRAPH.neighbors(i))

    def find_path(self, i: Kpoint, j:Kpoint) -> bool:
        '''
        Verify if exist a path between two k points inside the graph

        Parameters
            i : Kpoint
            j : Kpoint
        
        Returns : bool
            If exists a path return True
        '''
        neighs = self.get_neigs(i)
        neigh = neighs.pop(0) if len(neighs) > 0 else None
        visited = [i] + [d for points in self.degenerados
                         for d in points if d not in [i, j]]
        while (neigh is not None and neigh != j and
               (neigh not in visited or len(neighs) > 0)):
            if neigh in visited:
                neigh = neighs.pop(0)
                continue
            visited.append(neigh)
            for k in self.get_neigs(neigh):
                if k not in visited:
                    neighs.append(k)
            neigh = neighs.pop(0) if len(neighs) > 0 else None
        return neigh == j if neigh is not None else False

    def _grid_step_unit(self, k_ref: int, k_neighbor: int):
        '''
        Unit step (per axis, on the k-space grid) pointing from ``k_ref`` toward
        ``k_neighbor``. Neighbours are adjacent grid points, so exactly one axis
        differs by +/-1. No periodic (BZ-wrapping) boundary is assumed: the
        k-mesh may be open (as for MoS2), so stepping past an edge leaves the
        mesh rather than wrapping around.

        Returns
            (idx_ref, unit, sizes) : tuple of np.ndarray
                idx_ref : grid index of k_ref
                unit    : integer unit step per axis (a single axis is non-zero)
                sizes   : number of k-points per axis
        '''
        idx_ref = np.atleast_1d(np.asarray(self.kpoints_index[k_ref])).astype(int)
        idx_nb = np.atleast_1d(np.asarray(self.kpoints_index[k_neighbor])).astype(int)
        sizes = np.array([self.nkx, self.nky, self.nkz][:self.dimensions], dtype=int)
        unit = np.sign(idx_nb - idx_ref).astype(int)
        return idx_ref, unit, sizes

    def _extrapolate_energy(self, k_ref: int, k_neighbor: int, slot: int, N: int = 4,
                            trusted: np.ndarray = None, same_band: bool = False):
        '''
        Predict the energy that band ``slot`` (in the current solution
        ``self.bands_final``) would have at ``k_neighbor`` by fitting a quadratic
        to its own trajectory at ``k_ref`` and the points behind it (away from the
        neighbor), then evaluating one step forward.

        Points not yet attributed (``bands_final == -1``) are skipped so the fit
        never reads a bogus energy (``eigenvalues[k, -1]`` would silently pick the
        last band). When a ``trusted`` mask is given, untrusted points are skipped
        too, so the fit never reads through a suspect attribution. With
        ``same_band=True`` the trajectory is restricted to the single raw band held
        at the anchor (first usable point): the walk stops at the first change of
        band index, so the fit is never taken across a crossing / mis-attribution
        (the discrete form of a cliff). The k-mesh is open (no BZ wrap): stepping
        past an edge stops the trajectory.

        Returns
            (E_pred, sigma) : (float, float)
                E_pred : extrapolated energy at ``k_neighbor``.
                sigma  : local energy spread along the trajectory.
        '''
        idx_ref, unit, sizes = self._grid_step_unit(k_ref, k_neighbor)

        Xs, Es = [], []
        b_anchor = None
        for m in range(N):                                   # m=0 -> k_ref, m>0 -> behind it
            idx_b = idx_ref - m * unit
            if np.any(idx_b < 0) or np.any(idx_b >= sizes):
                break                                        # stepped off the open k-mesh: stop
            kb = int(self.matrix[tuple(idx_b)]) if self.dimensions > 1 else int(self.matrix[idx_b[0]])
            b = self.bands_final[kb, slot]
            if b == -1:
                continue                                     # unattributed: skip
            if trusted is not None and not trusted[kb, slot]:
                continue                                     # suspect attribution: skip
            E_b = float(self.eigenvalues[kb, b])
            if Es:
                # Cliff between this point and the last kept one -> stop the walk so the
                # quadratic is never fitted ACROSS a discontinuity. Cliff detection uses
                # BOTH criteria, never the band index alone: an energy step exceeding the
                # gross-jump scale accept_E, OR -- with same_band -- a change of raw band
                # index. (A step can break continuity either by jumping in energy or by
                # swapping bands; both must be caught.) Fitting through a cliff throws the
                # one-step prediction by ~100 mRy (median) in the entangled manifold;
                # truncating to the continuous segment nearest k_ref brings it to ~1 mRy.
                if abs(E_b - Es[-1]) > self.accept_E or (same_band and b != b_anchor):
                    break
            else:
                b_anchor = b                                 # first kept point sets the reference band
            Xs.append(-m)
            Es.append(E_b)

        if len(Es) == 0:
            # Nothing usable along this direction: fall back to the slot's own
            # energy at k_ref (or the raw eigenvalue ordering if unattributed).
            b_ref = self.bands_final[k_ref, slot]
            E_ref = self.eigenvalues[k_ref, b_ref if b_ref != -1 else slot]
            return float(E_ref), self.sigma_min

        sigma = max(float(np.std(Es)), self.sigma_min) if len(Es) > 1 else self.sigma_min
        if len(Es) < 3:
            return float(Es[0]), sigma                       # nearest-neighbour energy fallback
        w = _extrap_weights(tuple(Xs))                       # cached quadratic-extrapolation weights
        E_pred = float(w @ np.asarray(Es))                   # predict one step forward (k_neighbor)
        # Clip the one-step prediction to a local band window. A band cannot move
        # more than ~one (99.9-pct) dispersion step between adjacent k, so a larger
        # excursion is the quadratic overshooting -- the heavy error tail seen in the
        # entangled manifold. Bound it to E(nearest trajectory point) +/- disp_scale.
        # Smooth bands are unaffected (their prediction is already inside the window).
        E0 = float(Es[0])
        E_pred = min(max(E_pred, E0 - self.disp_scale), E0 + self.disp_scale)
        return E_pred, sigma

    def _slot_reference_energy(self, k: int, slot: int) -> float:
        '''
        Reference energy for (k, slot): the MEDIAN of one-step extrapolations taken
        from every attributed neighbour of k, each constrained (``same_band=True``)
        to the single band held at that neighbour -- so no direction's fit is ever
        taken across a band swap, and the points it uses are well attributed. The
        median makes the reference robust to a single bad direction. Falls back to
        the raw eigenvalue-ordering energy when no neighbour is attributed.
        '''
        preds = []
        for k_nb in self.neighbors[k]:
            if k_nb == -1:
                continue
            if self.bands_final[k_nb, slot] == -1:
                continue
            E_pred, _ = self._extrapolate_energy(k_nb, k, slot, same_band=True)
            preds.append(E_pred)
        if preds:
            return float(np.median(preds))
        return float(self.eigenvalues[k, slot])

    def _local_accept(self, k: int, band: int) -> float:
        '''
        Local gross-jump gate for candidate raw band ``band`` at k-point ``k``:
        half the local adjacent-band gap where it resolves the band's own
        dispersion, the loose global ``accept_E`` elsewhere (see _local_scales).
        '''
        if band < 0 or not hasattr(self, 'local_gap'):
            return self.accept_E
        gate, _ = _local_scales(self.accept_E, self.disp_scale,
                                float(self.local_gap[k, band]),
                                float(self.band_disp[band]))
        return gate

    def _dp_affinity(self, k: int, slot: int, band: int,
                     trusted: np.ndarray = None) -> Union[float, None]:
        '''
        Mean dot product of raw band ``band`` at ``k`` to whatever band the SAME
        slot holds at the attributed (optionally trusted-only) neighbours -- the
        wavefunction evidence that ``band`` is this slot's continuation. Returns
        None when no neighbour provides evidence (caller treats dp as neutral).
        '''
        vals = []
        for i_neig, k_nb in enumerate(self.neighbors[k]):
            if k_nb == -1:
                continue
            if trusted is not None and not trusted[k_nb, slot]:
                continue
            b2 = self.bands_final[k_nb, slot]
            if b2 == -1:
                continue
            vals.append(float(self.connections[k, i_neig, band, b2]))
        return float(np.mean(vals)) if vals else None

    def _energy_outlier_mask(self, bad0: np.ndarray, min_support: int = 2,
                             sigma_mult: float = 3.0) -> np.ndarray:
        '''
        Flag trusted slots whose currently-assigned band is energy-discontinuous
        with the trajectory their TRUSTED neighbours predict, even though the solver
        never flagged them NOT/MIS.

        The NOT/MIS trigger used by the repair is overlap-based; it misses slots that
        hold a valid (bijective) but wrong band -- e.g. a multi-band permutation cycle
        at a near-degenerate k-point that survives as POTENTIAL_MISTAKE / FORCED /
        CORRECT and shows up only as an energy jump in the band plot. Such a slot sits
        far from the quadratic trajectory its trusted neighbours extrapolate.

        A slot is flagged only when (a) it is currently trusted and not degenerate,
        (b) it has at least ``min_support`` trusted neighbours in that slot (so the
        prediction is robust to a single contaminated neighbour, taken via median) and
        (c) ``|E_current - median(prediction)|`` exceeds both the gross-jump guard
        ``accept_E`` and ``sigma_mult`` times the local trajectory spread. Prediction
        reuses ``_extrapolate_energy``, so legitimately steep bands are predicted
        correctly and are not flagged.

        Returns
            outlier : np.ndarray[bool], shape (nks, total_bands)
                True where a trusted slot is energy-discontinuous with its neighbours.
        '''
        trusted0 = (~bad0) & (self.bands_final != -1)
        outlier = np.zeros_like(bad0)
        for k in np.where(np.any(trusted0, axis=1))[0]:
            for s in np.where(trusted0[k])[0]:
                if self.signal_final[k, s] == DEGENERATE:
                    continue
                preds, sigmas = [], []
                for k_nb in self.neighbors[k]:
                    if k_nb == -1 or not trusted0[k_nb, s]:
                        continue
                    E_pred, sigma = self._extrapolate_energy(int(k_nb), int(k),
                                                             int(s), trusted=trusted0)
                    preds.append(E_pred)
                    sigmas.append(sigma)
                if len(preds) < min_support:
                    continue                              # too few anchors to judge robustly
                pred = float(np.median(preds))
                # Local gate instead of the flat accept_E floor: with the global
                # floor a one-band jump inside a crowded manifold (gap < accept_E)
                # could never be flagged as an outlier, no matter how clean the
                # trusted trajectory. Unresolved (near-degenerate) regions keep
                # the global gate, so nothing tightens where energy cannot judge.
                gate = self._local_accept(k, int(self.bands_final[k, s]))
                tol = max(gate, sigma_mult * float(np.median(sigmas)))
                E_cur = float(self.eigenvalues[k, self.bands_final[k, s]])
                if abs(E_cur - pred) > tol:
                    outlier[k, s] = True
        return outlier

    def _repair_energy_discontinuities(self, max_rounds: int = 64) -> np.ndarray:
        '''
        A-posteriori energy-continuity repair of the solver's best solution.

        The clustering solution is taken as-is and ONLY the bad points are revisited;
        trusted points are never touched. Bad = points that failed the energy-continuity
        validation (NOT/MIS in ``correct_signalfinal_best``), were left unattributed
        (-1), or hold a bijective-but-energy-discontinuous band that escaped the NOT/MIS
        overlap trigger (``_energy_outlier_mask`` -- catches permutation cycles at
        near-degenerate k-points that surface only as an energy jump in the band plot).
        For each bad k-point, all its bad slots are freed and jointly reassigned:
        each slot's expected energy is predicted by quadratic extrapolation of that
        slot's own trajectory (slot-correct namespace: ``E[k', bands_final[k', s]]``)
        built from TRUSTED neighbours only, and the freed bands are redistributed by
        a joint (Hungarian) assignment blending the energy residual with the dot
        product to the slot's trusted neighbours, respecting the per-k-point
        permutation constraint (no band used twice). A reassignment is only accepted
        if it lands within the LOCAL gross-jump gate of the candidate band (half the
        local adjacent gap where it resolves the band's dispersion, ``self.accept_E``
        elsewhere -- see ``_local_scales``).

        The POTENTIAL_MISTAKE / FORCED_CONTINUITY cells of a bad k-point are
        CO-FREED into the same joint assignment: the residual failures are
        permutation cycles (bands rotated among 3+ slots at a crossing filament)
        where only the largest-jump member is graded MISTAKE -- its cycle partners
        sit at OTHER/FOR, count as trusted, and lock exactly the bands the repair
        needs, leaving no admissible candidate. Co-freeing puts the whole cycle on
        the table; CORRECT cells stay locked, and a co-freed cell that finds no
        admissible reassignment is restored, so the co-free only enlarges the
        search space.

        Repaired points become trusted, so the repair propagates from the boundary
        of a bad island inward, one layer per round, until nothing changes. Slots
        that cannot be repaired keep their original attribution (left for the final
        validation to flag honestly) or stay -1 for the FORCED pass.

        This replaces the in-solve energy gate (e90a33a), which regressed badly:
        gating edges on a half-built solution shattered the graph (47 -> ~10k
        components) and extrapolated the wrong trajectories near crossings.

        Returns
            repaired_mask : np.ndarray[bool], shape (nks, total_bands)
                True where bands_final was reassigned (changed or newly attributed)
                by this pass.
        '''
        ###########################################################################
        # Bad = unattributed or energy-discontinuous in the best validation.
        # Degenerate points are basisrotation's job: never freed, never reassigned.
        ###########################################################################
        bad = self.bands_final == -1
        validation = getattr(self, 'correct_signalfinal_best', None)
        if validation is not None:
            bad |= (validation == NOT_SOLVED) | (validation == MISTAKE)
            bad &= validation != DEGENERATE
        bad &= self.signal_final != DEGENERATE

        ###########################################################################
        # Widen 'bad' with energy-continuity outliers: bijective-but-wrong bands
        # (permutation cycles at near-degenerate k-points) that escaped the NOT/MIS
        # overlap trigger above but sit far from the trajectory their trusted
        # neighbours predict. Freeing them lets the joint reassignment re-sort them.
        ###########################################################################
        n_signal_bad = int(np.sum(bad))
        bad |= self._energy_outlier_mask(bad)
        bad &= self.signal_final != DEGENERATE
        n_outlier = int(np.sum(bad)) - n_signal_bad
        if n_outlier:
            self.logger.info(f'{BODY_INDENT}Energy-outlier widening: {n_outlier} '
                             f'additional energy-discontinuous slot(s) flagged for repair')

        trusted = (~bad) & (self.bands_final != -1)
        repaired_mask = np.zeros_like(bad)
        n_bad_initial = int(np.sum(bad))
        if n_bad_initial == 0:
            return repaired_mask

        all_bands = np.arange(self.total_bands)
        neigh = np.asarray(self.neighbors)
        n_repaired = 0
        n_co_moved = 0

        for round_ in range(max_rounds):
            ks_bad = np.where(np.any(bad, axis=1))[0]
            if len(ks_bad) == 0:
                break
            ###########################################################################
            # Rank bad k-points by support: how many (bad slot, direction) pairs have
            # a trusted neighbour to extrapolate from. Most-supported first, so each
            # round peels the best-anchored boundary layer of every bad island.
            ###########################################################################
            support_pts = np.zeros(bad.shape, int)
            for j in range(neigh.shape[1]):
                nb = neigh[:, j]
                valid = nb != -1
                support_pts[valid] += trusted[nb[valid]]
            support = np.sum(support_pts * bad, axis=1)[ks_bad]
            order = np.argsort(-support, kind='stable')
            progress = 0
            for idx in order:
                if support[idx] == 0:
                    break                                   # the rest have no anchor this round
                k = int(ks_bad[idx])
                bad_slots = np.where(bad[k])[0]
                ###########################################################################
                # Free all bad slots at k, CO-FREEING its OTHER/FOR cells (see the
                # docstring: they are the trusted cycle partners locking the bands the
                # bad slots need). CORRECT and DEGENERATE cells stay locked.
                ###########################################################################
                # Predicted (reference) energy per slot, from every direction whose
                # nearest neighbour is trusted in that slot; median for robustness.
                def _slot_ref(s):
                    preds = [self._extrapolate_energy(k_nb, k, int(s), trusted=trusted)[0]
                             for k_nb in self.neighbors[k]
                             if k_nb != -1 and trusted[k_nb, s]]
                    return float(np.median(preds)) if preds else None

                bad_refs = [_slot_ref(s) for s in bad_slots]

                co = np.zeros(self.total_bands, dtype=bool)
                if validation is not None:
                    co = (((validation[k] == POTENTIAL_MISTAKE) |
                           (validation[k] == FORCED_CONTINUITY)) &
                          ~bad[k] & (self.bands_final[k] != -1) &
                          (self.signal_final[k] != DEGENERATE))
                    if co.any():
                        # Energy window: a cycle partner's held band must lie within the
                        # gross-jump gate of a bad slot's held band (the band the partner
                        # would take over) or of a bad slot's reference energy (the band
                        # the bad slot needs from the partner). Cells further away cannot
                        # take part in any admissible exchange, so freeing them only
                        # risks losing them to the Hungarian. Without this, valence OTHER
                        # cells co-freed at conduction-failure k-points ended as holes ->
                        # force-fill -> "suspect", costing a clean band (phosphorene
                        # band 5, Jul 10 run).
                        bad_bands = self.bands_final[k, bad_slots]
                        anchors = list(self.eigenvalues[k, bad_bands[bad_bands >= 0]])
                        anchors += [r for r in bad_refs if r is not None]
                        if anchors:
                            anchors = np.asarray(anchors, dtype=float)
                            co_idx = np.where(co)[0]
                            co_E = self.eigenvalues[k, self.bands_final[k, co_idx]]
                            near = np.min(np.abs(co_E[:, None] - anchors[None, :]),
                                          axis=1) <= self.accept_E
                            co[co_idx[~near]] = False
                        else:
                            co[:] = False
                free_slots = np.concatenate([bad_slots, np.where(co)[0]]).astype(int)
                n_genuine = len(bad_slots)
                locked = np.ones(self.total_bands, dtype=bool)
                locked[free_slots] = False
                kept = self.bands_final[k][locked]
                used = np.unique(kept[kept != -1])
                available = list(np.setdiff1d(all_bands, used))
                avail_E = [float(self.eigenvalues[k, b]) for b in available]

                refs = bad_refs + [_slot_ref(s) for s in free_slots[n_genuine:]]

                original = self.bands_final[k, free_slots].copy()
                self.bands_final[k, free_slots] = -1

                ###########################################################################
                # Jointly reassign the freed slots to the available bands (Hungarian),
                # scoring each (slot, band) pair by BOTH kinds of evidence:
                #     cost = REPAIR_W_E * (residual / local gate)
                #          + REPAIR_W_DP * (1 - mean dp to the slot's trusted neighbours)
                # A pair is admissible only when its residual is inside the LOCAL
                # gross-jump gate of the candidate band (half the local adjacent gap
                # in a resolved manifold, the loose global accept_E elsewhere). In a
                # crowded manifold the global gate admits 2-4 adjacent bands, and the
                # old energy-nearest choice was character-blind -- the phosphorene
                # failures all held a dp~0.5 band while a dp~0.99 continuation was
                # available. The dot product to the trusted same-slot neighbours now
                # picks among the energy-admissible candidates.
                ###########################################################################
                pred_idx = [i for i, r in enumerate(refs) if r is not None]
                if pred_idx and available:
                    BIG = 1e6
                    cost = np.full((len(pred_idx), len(available)), BIG)
                    for row, i in enumerate(pred_idx):
                        s = int(free_slots[i])
                        for col, b in enumerate(available):
                            gate = self._local_accept(k, int(b))
                            resid = abs(avail_E[col] - refs[i])
                            if resid > gate:
                                continue                    # across a local energy wall
                            dp_aff = self._dp_affinity(k, s, int(b), trusted)
                            dp_term = (1.0 - dp_aff) if dp_aff is not None else 0.5
                            cost[row, col] = REPAIR_W_E * (resid / gate) + REPAIR_W_DP * dp_term
                    rows_, cols_ = linear_sum_assignment(cost)
                    taken = []
                    for r_, c_ in zip(rows_, cols_):
                        if cost[r_, c_] >= BIG:
                            continue                        # only inadmissible pairings left
                        self.bands_final[k, int(free_slots[pred_idx[r_]])] = available[c_]
                        taken.append(c_)
                    for c_ in sorted(taken, reverse=True):
                        available.pop(c_)
                        avail_E.pop(c_)

                for i, s in enumerate(free_slots):
                    s = int(s)
                    genuine = i < n_genuine
                    if self.bands_final[k, s] != -1:
                        if genuine and self.bands_final[k, s] != original[i]:
                            # Repaired: trust it so the next layer can extrapolate through it.
                            # Re-picking the ORIGINAL band is not a repair -- in a blob
                            # core the refs run through still-wrong (OTHER, trusted)
                            # trajectories and just confirm the standing cycle; promoting
                            # that to trusted locks the core before the rim has healed.
                            # Leave it flagged: every later round retries it with the
                            # refs the earlier repairs have improved.
                            bad[k, s] = False
                            trusted[k, s] = True
                            progress += 1
                        if self.bands_final[k, s] != original[i]:
                            repaired_mask[k, s] = True
                            n_repaired += 1
                            if not genuine:
                                # A cycle partner moved: the canvas changed, so give the
                                # next round a chance to build on it. An unchanged
                                # co-freed cell is NOT progress (it would spin the round
                                # loop on k-points the repair can no longer improve).
                                n_co_moved += 1
                                progress += 1
                            if self.logger.level <= logging.DEBUG:
                                nb_ = int(self.bands_final[k, s])
                                rE = refs[i]
                                self.logger.debug(
                                    f'{BODY_INDENT}  [repair] k={k} slot={s}'
                                    f'{"" if genuine else " (co-freed)"}: '
                                    f'band {int(original[i])}->{nb_}  '
                                    f'E={float(self.eigenvalues[k, nb_]):.4f}  '
                                    f'refE={("%.4f" % rE) if rE is not None else "n/a"}  round={round_ + 1}')
                    elif original[i] != -1 and original[i] in available:
                        # Failed: restore the original attribution (stays flagged) so
                        # the FORCED count is not inflated; it may be repaired later.
                        j = available.index(original[i])
                        available.pop(j)
                        avail_E.pop(j)
                        self.bands_final[k, s] = original[i]
                    elif not genuine:
                        # Co-freed cell whose band was claimed by the cycle unwind and
                        # which found no admissible replacement this round: hand it to
                        # the repair as a genuine hole (later rounds gain anchors as the
                        # neighbourhood heals; the completeness pass is the backstop).
                        bad[k, s] = True
                        trusted[k, s] = False

            self.logger.debug(f'\t\tRepair round {round_ + 1}: {progress} point(s) re-anchored')
            if progress == 0:
                break

        n_left = int(np.sum(bad))
        self.logger.info(f'{BODY_INDENT}Energy repair: {n_bad_initial} flagged point(s), '
                         f'{n_repaired} reassigned ({n_co_moved} co-freed OTHER/FOR '
                         f'cycle partner(s) among them), {n_left} not repairable '
                         f'(kept original attribution or left for the completeness pass)')

        ###########################################################################
        # Re-derive the dot-product signal of every re-anchored point (its band or
        # its neighbourhood changed), capped at POTENTIAL_CORRECT so the final
        # energy-continuity validation always re-examines it.
        ###########################################################################
        ks_r, slots_r = np.where(repaired_mask)
        for k, s in zip(ks_r, slots_r):
            bn1 = self.bands_final[k, s]
            values = []
            for i_neig, k_neig in enumerate(self.neighbors[k]):
                if k_neig == -1:
                    continue
                bn2 = self.bands_final[k_neig, s]
                if bn2 == -1:
                    continue
                values.append(self.connections[k, i_neig, bn1, bn2])
            signal = evaluate_result(values) if values else NOT_SOLVED
            self.signal_final[k, s] = min(signal, POTENTIAL_CORRECT)

        return repaired_mask

    def _force_complete_bands(self) -> np.ndarray:
        '''
        Force a genuine band attribution for every k-point/slot the clustering
        left unattributed (-1). For each k-point, the bands not used by any
        attributed slot ("available") are jointly assigned to the empty slots by
        a Hungarian assignment whose cost blends the energy residual to each
        slot's continuity reference (its own trajectory's prediction, normalised
        by the candidate band's local gate and capped) with the dot product to
        the slot's attributed neighbours -- the same evidence blend as the
        repair pass, but with no hard gate, since every slot must be filled.

        Returns
            forced_mask : np.ndarray[bool], shape (nks, total_bands)
                True where bands_final was force-filled by this pass.
        '''
        forced_mask = (self.bands_final == -1)
        all_bands = np.arange(self.total_bands)
        for k in range(self.nks):
            empty_slots = np.where(forced_mask[k])[0]
            if len(empty_slots) == 0:
                continue
            used = np.unique(self.bands_final[k][~forced_mask[k]])
            # Available bands = those not used by any attributed slot at this k.
            available = list(np.setdiff1d(all_bands, used))
            avail_E = [float(self.eigenvalues[k, b]) for b in available]
            # Continuity reference for each empty slot.
            refs = [self._slot_reference_energy(k, s) for s in empty_slots]
            # Joint (Hungarian) fill blending the energy residual with the dot
            # product to the slot's trusted neighbours -- the same evidence blend
            # as the repair pass, but with NO hard gate (every slot must be
            # filled). The residual is normalised by the candidate band's local
            # gate and capped, so energy still ranks candidates but can no longer
            # silently place a band across a resolved local wall when a
            # dp-supported alternative exists.
            if available:
                cost = np.empty((len(empty_slots), len(available)))
                for row, s in enumerate(empty_slots):
                    for col, b in enumerate(available):
                        gate = self._local_accept(k, int(b))
                        resid = abs(avail_E[col] - refs[row])
                        dp_aff = self._dp_affinity(k, int(s), int(b))
                        dp_term = (1.0 - dp_aff) if dp_aff is not None else 0.5
                        cost[row, col] = (REPAIR_W_E * min(resid / gate, 10.0)
                                          + REPAIR_W_DP * dp_term)
                rows_, cols_ = linear_sum_assignment(cost)
                for r_, c_ in zip(rows_, cols_):
                    s = int(empty_slots[r_])
                    chosen = int(available[c_])
                    self.bands_final[k, s] = chosen
                    if self.logger.level <= logging.DEBUG:
                        dE = abs(avail_E[c_] - refs[r_])
                        self.logger.debug(f'{BODY_INDENT}  [forced] k={k} slot={s}: '
                                          f'-1->band {chosen}  E={float(self.eigenvalues[k, chosen]):.4f}  '
                                          f'refE={float(refs[r_]):.4f}  |dE|={dE:.4f}')
            for s in empty_slots:
                s = int(s)
                if self.bands_final[k, s] == -1:
                    self.bands_final[k, s] = s          # last resort (permutation defect)
                    if self.logger.level <= logging.DEBUG:
                        self.logger.debug(f'{BODY_INDENT}  [forced] k={k} slot={s}: '
                                          f'no band available -> last-resort band {s}')
        return forced_mask

    def _nn_energy_jump_map(self) -> np.ndarray:
        '''
        Per (k, slot) the largest |E(k, slot) - E(neighbour, slot)| over the slot's
        attributed in-BZ neighbours (slot-correct namespace: the energy of whatever
        band each slot holds). NaN where the slot is unattributed or has no attributed
        neighbour. This is the diagnostic that exposes silent mis-attributions: a slot
        sitting on a gross energy wall relative to its own neighbours, even when the
        validation flag marks it solved. Used only by the verbose provenance log.
        '''
        nks, nb = self.bands_final.shape
        jumps = np.full((nks, nb), np.nan)
        neigh = np.asarray(self.neighbors)
        for k in range(nks):
            for s in range(nb):
                b = self.bands_final[k, s]
                if b < 0:
                    continue
                Ek = self.eigenvalues[k, b]
                best = np.nan
                for kn in neigh[k]:
                    if kn == -1:
                        continue
                    bn = self.bands_final[kn, s]
                    if bn < 0:
                        continue
                    d = abs(Ek - self.eigenvalues[kn, bn])
                    best = d if np.isnan(best) else max(best, d)
                jumps[k, s] = best
        return jumps

    def _log_solution_provenance(self, inloop_bands: np.ndarray, inloop_csf: np.ndarray) -> None:
        '''
        Verbose (-v / DEBUG) post-mortem of the final solution. For every slot it
        reports where its value came from -- solved IN-LOOP by the clustering, REPAIRED
        by the a-posteriori energy-continuity pass, or FORCE-FILLED by the completeness
        pass -- and surfaces what is likely wrong:

          1. SLOT PROVENANCE & HEALTH table: per slot, counts of in-loop / repaired /
             forced, plus how many points the final flag failed (csf<3) and -- the key
             column -- how many PASS the flag (csf>=3) yet sit on a gross energy jump
             (> accept_E). That last count is the silent mis-attribution the report hides.
          2. POST-LOOP SLOT SWAPS: every (k, slot) whose band changed from the in-loop
             best to the final solution (repair + bijection reassignments + gap fills),
             with old->new band and energies.
          3. SILENT DISCONTINUITIES: each (k, slot) that passes validation but jumps
             > accept_E vs its own neighbours, sorted worst first, with provenance.

        Skipped unless the logger is at DEBUG level, since it scans every (k, slot).
        Takes the in-loop best (bands + validation) snapshotted before the post-loop
        passes ran.
        '''
        if self.logger.level > logging.DEBUG:
            return
        CSF_SOLVED = 3          # correct_signalfinal: >=3 (OTHER/CORRECT/FORCED_CONT) == "solved/reported-OK"
        CSF_NAME = {0: 'NOT_SOLVED', 1: 'MISTAKE', 2: 'DEGENERATE',
                    3: 'OTHER', 4: 'CORRECT', 5: 'FORCED_CONT'}
        nks, nb = self.bands_final.shape
        csf = self.correct_signalfinal
        repaired = getattr(self, 'repaired_mask', None)
        if repaired is None:
            repaired = np.zeros((nks, nb), bool)
        forced = getattr(self, 'forced_mask', None)
        if forced is None:
            forced = np.zeros((nks, nb), bool)
        jumps = self._nn_energy_jump_map()
        gross = float(self.accept_E)

        def prov(k: int, s: int) -> str:
            if forced[k, s]:
                return 'FORCED'
            if repaired[k, s]:
                return 'REPAIRED'
            if inloop_csf[k, s] >= CSF_SOLVED:
                return 'IN-LOOP'
            return 'IN-LOOP(flag)'

        self.logger.debug('\n\t\t==================== SOLUTION PROVENANCE & HEALTH (verbose) ===================='
                          f'\n\t\tgross-jump threshold accept_E = {gross:.4f} Ry')
        self.logger.debug('\t\tslot | in-loop  repaired  forced | csf<3(fail) | SILENT-DISC(csf>=3 & jump>accept_E)')
        for s in range(nb):
            kept_flag = ((inloop_csf[:, s] < CSF_SOLVED) & ~repaired[:, s] & ~forced[:, s]).sum()
            n_inloop = ((inloop_csf[:, s] >= CSF_SOLVED) & ~repaired[:, s] & ~forced[:, s]).sum()
            n_rep = (repaired[:, s] & ~forced[:, s]).sum()
            n_forced = forced[:, s].sum()
            n_fail = (csf[:, s] < CSF_SOLVED).sum()
            col_jump = jumps[:, s]
            n_silent = ((csf[:, s] >= CSF_SOLVED) & (col_jump > gross)).sum()
            self.logger.debug(f'\t\t{s:4d} | {int(n_inloop):7d} {int(n_rep):9d} {int(n_forced):7d} | '
                              f'{int(n_fail):11d} | {int(n_silent):d}'
                              + (f'   (+{int(kept_flag)} in-loop slots kept though flagged)' if kept_flag else ''))

        # 2. Post-loop swaps (band changed vs the in-loop best solution).
        sk, ss = np.where(inloop_bands != self.bands_final)
        self.logger.debug(f'\n\t\t==================== POST-LOOP SLOT SWAPS: {len(sk)} ====================')
        if len(sk):
            self.logger.debug('\t\t    k  slot  band(in->out)   E(in->out) [Ry]      nn-jump  provenance     final-csf')
            for k, s in sorted(zip(sk.tolist(), ss.tolist())):
                ob, nbd = int(inloop_bands[k, s]), int(self.bands_final[k, s])
                oE = float(self.eigenvalues[k, ob]) if ob >= 0 else float('nan')
                nE = float(self.eigenvalues[k, nbd]) if nbd >= 0 else float('nan')
                jv = jumps[k, s]
                self.logger.debug(f'\t\t{k:5d} {s:4d}   {ob:3d}->{nbd:<3d}      {oE:8.4f}->{nE:<8.4f}   '
                                  f'{(jv if not np.isnan(jv) else -1):8.4f}  {prov(k, s):13s}  '
                                  f'{CSF_NAME.get(int(csf[k, s]), int(csf[k, s]))}')

        # 3. Silent discontinuities: pass validation but jump > accept_E, worst first.
        hk, hs = np.where((csf >= CSF_SOLVED) & (jumps > gross))
        self.logger.debug(f'\n\t\t==================== SILENT DISCONTINUITIES (pass validation, '
                          f'jump>accept_E): {len(hk)} ====================')
        if len(hk):
            self.logger.debug('\t\t    k  slot  band     E [Ry]    nn-jump  provenance     in-loop-band  final-csf')
            order = np.argsort(-jumps[hk, hs])
            for i in order:
                k, s = int(hk[i]), int(hs[i])
                b = int(self.bands_final[k, s])
                ib = int(inloop_bands[k, s])
                self.logger.debug(f'\t\t{k:5d} {s:4d}  {b:4d}  {float(self.eigenvalues[k, b]):8.4f}  '
                                  f'{jumps[k, s]:8.4f}  {prov(k, s):13s}  {ib:11d}   '
                                  f'{CSF_NAME.get(int(csf[k, s]), int(csf[k, s]))}')
        self.logger.debug('\t\t===============================================================================\n')

    def make_connections(self, tol:float=0.80, not_first_iteration:bool=False, node_subset=None) -> None:
        '''
        This function evaluates the connection between each k point,
        and adds an edge to the graph if its connection is greater
        than a tolerance value (tol).

        <i|j>: The dot product between i and j represents its connection

        Parameters
            tol : float
                It is the minimum connection value that will be accepted as an edge.
                default: 0.80
            node_subset : array_like or None
                If given, dot-product edges are computed ONLY from these node ids
                (the edges are undirected, so an edge to a node outside the subset
                is still created). Used by the error-region-only rebuild so the
                good bands keep just their continuity edges and are not re-clustered.
                default: None (every node, i.e. the full graph)
        '''
        ###########################################################################
        # Find the edges on the graph
        ###########################################################################
        self.tol = tol
        tol = 1 - 2/np.pi * np.arccos(tol) # Convert the tolerance value to arccos metric

        ###########################################################################
        # Edges are built from the dot product ALONE. Energy continuity is NOT
        # enforced here: gating edges on a half-built solution shattered the graph
        # and locked in mistakes (e90a33a regression). Energy continuity is applied
        # a posteriori instead, by _repair_energy_discontinuities at the end of
        # solve(), which only ever touches flagged points.
        ###########################################################################
        def connection_component(vectors:np.ndarray) -> list[list[Kpoint]]:
            '''
            Find the possible edges in the graph using the information of dot product.

            Parameters
                vectors : array_like
                    An array with vector representation of k points.

            Returns
                edges : list[list[Kpoint]]
                    List of all edges that was found.
            '''
            edges = []
            bands = np.repeat(np.arange(0, self.nbnd), self.number_neighbors)
            for i_ in vectors:
                bn1 = i_//self.nks  # bi
                k1 = i_ % self.nks
                neighs = np.tile(self.neighbors[k1], self.nbnd)
                ks = neighs + bands*self.nks
                ks = ks[neighs != -1]
                for j_ in ks:
                    k2 = j_ % self.nks
                    bn2 = j_//self.nks  # bj
                    i_neig = int(np.where(self.neighbors[k1] == k2)[0][0])
                    connection = float(self.connections[k1, i_neig,
                                                    bn1, bn2])  # <i|j>
                    '''
                    for each first neighbor
                    Edge(i,j) = 1 iff <i, j> ~ 1
                    '''
                    connection = 1 - 2/np.pi * np.arccos(connection)
                    if connection <= tol:
                        continue
                    # Cross-band guard (anti-bridge). A cross-band edge (bn1 != bn2)
                    # is only physical at a genuine crossing, where band bn1 has lost
                    # its own continuity at k2. If bn1 still connects strongly to
                    # itself (<bn1@k1|bn1@k2> > tol), this cross-band link is a
                    # spurious bridge: connected components are transitive, so a
                    # single such edge fuses two otherwise-separate bands (e.g. a
                    # Kramers-degenerate pair) into one component and collapses a band
                    # slot (the empty-band symptom). Drop it -- the same-band edge
                    # already carries the continuity. Dropping only the edges where
                    # the same band stays strong cannot disconnect a band from its own
                    # chain; genuine crossings (same-band overlap collapsed) are kept.
                    if bn1 != bn2:
                        same = float(self.connections[k1, i_neig, bn1, bn1])   # <bn1@k1|bn1@k2>
                        same = 1 - 2/np.pi * np.arccos(same)
                        if same > tol:
                            continue
                    edges.append([i_, j_, connection])  # Add the weighted edge
            return edges

        self.logger.info(f'\t\tTolerance: {tol}')
        # Parallelize the edges calculation (over every node, or just the subset
        # when an error-region rebuild restricts which nodes originate edges).
        nodes_iter = range(len(self.vectors)) if node_subset is None else list(node_subset)
        edges = self.parallelize('\t\tComputing Edges', connection_component, nodes_iter)
        # Establish the edges on the graph from edges array
        self.GRAPH.add_weighted_edges_from(edges)

        if not_first_iteration:
            return


        ###########################################################################
        # Solve problems that a degenerate point may cause
        ###########################################################################
        degnerates = []
        problems = []
        for d1, d2 in self.degenerados:
            '''
            The degenerate points may cause problems.
            The algorithm below finds its problems and solves them.
            '''
            flag = False
            if not self.find_path(d1, d2):
                # Verify if exist a path that connects two forbidden points
                # The points does not cause problems but are degenerated, then, they are signaled
                # The basis rotation program will solve them.
                degnerates.append([d1, d2])
                continue
            # Obtains the neighbors from each degenerate point that cause problems
            N1 = np.array(self.get_neigs(d1))
            N2 = np.array(self.get_neigs(d2))
            if len(N1) == 0 or len(N2) == 0:
                continue
            problem = {
                d1 : N1,
                d2 : N2,
            }
            self.logger.debug(f'\tProblem:\n\t{d1}: {N1}\n\t{d2}: {N2}\n')
            NKS = self.nks
            if len(N1) > 1 and len(N2) > 1:
                N = []
                for n1 in N1:
                    n2_idx = np.where(N2 % NKS == n1 % NKS)
                    if len(n2_idx[0]) == 0:
                        continue
                    n2 = N2[n2_idx[0][0]]
                    N.append([n1, n2])
                if len(N) == 0:
                    continue
            else:
                if len(N1) == len(N2):
                    N = list(zip(N1, N2))
                else:
                    Ns = [N1, N2]
                    N_1 = Ns[np.argmin([len(N1), len(N2)])]
                    N_2 = Ns[np.argmax([len(N1), len(N2)])]
                    n2_idx = np.where(N_2 % NKS == N_1[0] % NKS)
                    if len(n2_idx[0]) == 0:
                        continue
                    n2_index = n2_idx[0][0]
                    N = [[N_1[0], N_2[n2_index]]] \
                        + [[n] for n in N_2 if n != N_2[n2_index]]
                    flag = True
            # Assign to a specific band each point and establish the corresponding edges
            n1 = np.random.choice(N[0])
            if flag:
                N1_ = [n1]
                N2_ = [N[0][np.argmax(np.abs(N[0]-n1))]]
                n2 = N2_[0]
                Ns = [N1_, N2_]
                for n in N[1:]:
                    n = n[0]
                    Ns[np.argmin(np.abs(np.array([n1, n2]) - n))].append(n)
            else:
                N1_ = [n[np.argmin(np.abs(n-n1))] for n in N]
                N2_ = [n[np.argmax(np.abs(n-n1))] for n in N]
            solution = {
                d1 : N1_,
                d2 : N2_
            }
            problems.append({
                'points' : [d1, d2],
                'problem' : problem,
                'solution' : solution
            })
            self.logger.debug(f'\tSolution:\n\t{d1}: {N1_}\n\t{d2}: {N2_}\n')
            for k in N1:
                self.GRAPH.remove_edge(k, d1)
            for k in N2:
                self.GRAPH.remove_edge(k, d2)

            for k in N1_:
                self.GRAPH.add_edge(k, d1)
            for k in N2_:
                self.GRAPH.add_edge(k, d2)
        
        self.degenerates = np.array(degnerates)

        ###########################################################################
        # Show the degenerate points that causes problems
        ###########################################################################
        self.solved_problems_info : list[str, list] = ['', []]
        forbidden_points = '''\n\t\t  ''' + textwrap.fill(
            'A forbidden path indicates that a specific k-point was identified as degenerate. However, this k-point establishes a connection between two distinct bands, culminating in the k-point having a link with the same k-point in a different band—an arrangement that is not allowed.',
            width=110,
            subsequent_indent='\t\t  '
        ) + '\n'

        self.solved_problems_info[0] = '\n\t\tThe number of points with forbidden paths between them is: ' + str(len(problems))
        self.solved_problems_info[0] += '\n\t' + forbidden_points
        if len(problems) > 0:
            self.solved_problems_info[0] += '\n\t\t  Problems in points:'
            self.logger.info('\t*** Points with forbidden paths found and solved ***')
            self.logger.info(forbidden_points)
            self.logger.info('\t    The problems and their solutions are:')
        
        calc_k_bn = lambda p: (p % self.nks, p // self.nks )
        for problem_dic in problems:
            d1, d2 = problem_dic['points']
            problem = problem_dic['problem']
            solution = problem_dic['solution']
            self.logger.info(f'\t    * Problem:')
            k1, bn1 = calc_k_bn(d1)
            _, bn2 = calc_k_bn(d2)
            self.solved_problems_info[1].append([d1, d2])
            self.logger.info(f'\n\t\tK-point: {k1} in bands: {bn1}, {bn2}')
            self.logger.info(f'\t\t   k: {k1}, band: {bn1} has edges with:')
            for k, bn in map(calc_k_bn, problem[d1]):
                self.logger.info(f'\t\t    k: {k} bn: {bn}')
            self.logger.info(f'\t\t   k: {k1}, band: {bn2} has edges with:')
            for k, bn in map(calc_k_bn, problem[d2]):
                self.logger.info(f'\t\t    k: {k} bn: {bn}')
            self.logger.info(f'\n\t      Solution:')
            self.logger.info(f'\t\t   k: {k1}, band: {bn1} has edges with:')
            for k, bn in map(calc_k_bn, solution[d1]):
                self.logger.info(f'\t\t    k: {k} bn: {bn}')
            self.logger.info(f'\t\t   k: {k1}, band: {bn2} has edges with:')
            for k, bn in map(calc_k_bn, solution[d2]):
                self.logger.info(f'\t\t    k: {k} bn: {bn}')

        if len(problems) > 0:
            self.logger.info('\n\t    Note that this solution may be wrong \n\t    but next iterations will correct it.\n')

            

    def parallelize(self, process_name: str, f: Callable, iterator: Union[list, np.ndarray], per_actual:int=0, N_total:int=None,*args) -> np.ndarray:
        '''
        Apply the function f to an iterator in parallel using a process Pool.

        The iterator is split into one chunk per worker. Each chunk is processed
        by f in a separate process and the per-chunk results are returned
        directly by the Pool (no shared-memory Manager) and concatenated in
        input order.

        A fresh fork-context Pool is created on each call so that the workers
        inherit the current process state (self and its arrays) copy-on-write.
        This is required because the worker closures capture mutable state that
        changes between calls (e.g. signal_final, the samples/clusters lists),
        so a Pool reused across calls would operate on a stale snapshot.

        Parameters
            process_name : string
                The name of the process to be parallelized.
            f : Callable
                Function f(iterator: array_like, *args) to be applied.
                This function gets another iterator as a parameter and will compute the result for each element in the iterator.
            iterator : array_like
                The function f is applied to elements in the iterator array.
            per_actual : int
                Number of elements already processed (for a cumulative progress bar).
            N_total : int
                Total number of elements across several calls (for a cumulative progress bar).

        Return
            result : list
                It contains the result for each value inside the iterator,
                concatenated in input order.
        '''
        global _PARALLEL_FN, _PARALLEL_ARGS

        iterator = list(iterator)
        N = len(iterator) if N_total is None else N_total

        ###########################################################################
        # Debug information and Progress bar
        ###########################################################################
        self.logger.debug(f'Starting Parallelization for {process_name} with {len(iterator)} values. Number of processes: {self.n_process}. Starting at {per_actual}/{N_total}')
        self.logger.percent_complete(per_actual, N, title=process_name)

        def finish_progress():
            if N_total is None:
                self.logger.percent_complete(N, N, title=process_name)
            else:
                self.logger.percent_complete(per_actual, N, title=process_name)

        result = []
        if len(iterator) == 0:
            finish_progress()
            return result

        ###########################################################################
        # Split the work into one chunk per worker
        ###########################################################################
        n_proc = max(1, min(self.n_process, len(iterator)))
        chunk_size = (len(iterator) + n_proc - 1) // n_proc
        chunks = [iterator[i:i + chunk_size] for i in range(0, len(iterator), chunk_size)]

        # Publish the worker function so the forked Pool workers reach it (and
        # the large read-only arrays it closes over) via copy-on-write, without
        # pickling them on every task.
        _PARALLEL_FN = f
        _PARALLEL_ARGS = args

        done = per_actual
        try:
            if n_proc == 1:
                # Run serially, skipping the fork/IPC overhead entirely.
                for chunk in chunks:
                    value = _parallel_dispatch(chunk)
                    if value:
                        result += value
                    done += len(chunk)
                    self.logger.percent_complete(done, N, title=process_name)
            else:
                # A fork context lets the workers inherit the current process state
                # (self and its arrays) copy-on-write without re-pickling it.  We use
                # ProcessPoolExecutor instead of Pool.imap (H1): if a worker dies
                # mid-chunk, executor.map raises BrokenProcessPool rather than blocking
                # the parent forever.  The workers are forked after _PARALLEL_FN is
                # published above, so they still inherit it via copy-on-write.
                with ProcessPoolExecutor(max_workers=n_proc, mp_context=get_context('fork')) as executor:
                    try:
                        for chunk, value in zip(chunks, executor.map(_parallel_dispatch, chunks)):
                            if value:
                                result += value                                  # Store the output into result array
                            done += len(chunk)                                   # The counter is actualized
                            self.logger.percent_complete(done, N, title=process_name)
                    except BrokenProcessPool:
                        self.logger.error('A clustering worker process died without returning a result.')
                        raise
        finally:
            _PARALLEL_FN = None
            _PARALLEL_ARGS = ()

        finish_progress()
        return result

    def _log_fE_stats(self, stats: dict) -> None:
        '''
        f(E) diagnostics, Tiers 1-2: per-(solve-iteration) summary of the in-loop
        energy term. Reports how often the predictor fit vs fell back to the
        nearest-neighbour default, how aggressively the cliff-break truncated and
        the disp_scale window clipped, the residual distribution (|E_pred - E_neigh|
        in units of disp_scale), and how much weight the energy term carried in the
        blended score relative to the dot product. Together these say WHETHER f(E)
        influenced the clustering and whether its predictions were any good.
        '''
        if not _FE_DEBUG or stats['calls'] == 0:
            return
        calls = stats['calls']
        kept_mean = stats['kept_len_sum'] / calls if calls else 0.0
        pair_n = stats['pair_n']
        conn = stats['conn_sum'] / pair_n if pair_n else 0.0
        eng = stats['eng_sum'] / pair_n if pair_n else 0.0
        total_w = conn + eng
        eng_frac = (eng / total_w) if total_w else 0.0
        rn = stats['resid_n']
        rb = stats['resid_buckets']
        rb_pct = [100.0 * b / rn for b in rb] if rn else [0.0] * len(rb)
        self.logger.info(
            f'{BODY_INDENT}f(E) iter: calls={calls} fit={stats["fit"]} '
            f'fallback={stats["fallback"]} ({100.0*stats["fallback"]/calls:.0f}%) '
            f'clip_hit={stats["clip_hit"]} trunc={stats["trunc"]} kept_len={kept_mean:.2f}')
        self.logger.info(
            f'{BODY_INDENT}f(E) breaks: Ejump={stats["brk_ejump"]} bandchg={stats["brk_band"]} | '
            f'term weight conn={conn:.3f} eng={eng:.3f} (eng/total={eng_frac:.2f})')
        self.logger.info(
            f'{BODY_INDENT}f(E) resid/disp [<0.5,<1,<2,<5,>=5] %: '
            f'[{", ".join(f"{p:.0f}" for p in rb_pct)}]  (n={rn})')

    def _log_fE_outcome(self) -> None:
        '''
        f(E) diagnostics, Tier 4: the energy-continuity validation outcome of the
        k-points attached during the last get_components call (the joins the
        sample-to-cluster score, and therefore f(E), helped decide) compared to the
        whole grid. If the attached points are disproportionately NOT/MIS, the
        attachment scoring is making bad joins in exactly the region it acted on.
        '''
        if not _FE_DEBUG:
            return
        attached = getattr(self, 'fE_attached_k', None)
        val = getattr(self, 'correct_signalfinal', None)
        if attached is None or val is None or len(attached) == 0:
            return

        def bad_frac(rows):
            sub = val[rows]
            bad = int(np.sum((sub == NOT_SOLVED) | (sub == MISTAKE)))
            return (100.0 * bad / sub.size) if sub.size else 0.0

        all_rows = np.arange(val.shape[0])
        self.logger.info(
            f'{BODY_INDENT}f(E) attached this iter: {len(attached)} k-point(s) | '
            f'NOT/MIS over them: {bad_frac(attached):.1f}%  vs grid {bad_frac(all_rows):.1f}%')

    def _partition_fused_component(self, sub_graph: nx.Graph) -> Union[list, None]:
        '''
        Seeded region-growing split of an oversized (fused) graph component.
        Returns a list of node-sets, or None when it cannot run (no growth
        target reaches a usable seed; caller falls back to Louvain). A
        single-target component (one band plus debris) is grown too: the band's
        chain is extracted and the debris becomes leftover pieces.

        Louvain optimises modularity -- an edge-density objective with no notion
        of energy, dot-product quality or the one-state-per-k constraint -- so
        when crossing filaments fuse several bands into one component its cuts
        through the ambiguous regions are arbitrary (phosphorene: 4-9 oversized
        components survived every iteration and all conduction bands failed).
        Here each fused band is grown from a seed that is correct by
        construction, claiming one node at a time on wavefunction evidence:

          1. Growth targets: the raw bands holding >= RG_BAND_PRESENCE of their
             nks points inside the component.
          2. Seed of a band = its largest k-connected SOVEREIGN region: nodes
             where the local gap resolves the band's dispersion
             (local_gap > 2*band_disp, the fix-1 local scales) AND every
             in-component same-band neighbour overlap is >= RG_SEED_DP --
             energy and dot product agree and are unambiguous there.
          3. Multi-source cheapest-frontier growth (Prim/watershed style):
             frontier edges pop globally cheapest first with cost = 1 - dp.
             A claim is admissible only when the one-step energy residual is
             inside the candidate's LOCAL gross-jump gate (_local_accept) and
             the label does not already own a node at that k-point -- the per-k
             bijection is enforced DURING the partition, not patched after.
             A node whose best offers from two labels lie within
             RG_DEFER_MARGIN is deferred: the crossing filaments Louvain used
             to cut arbitrarily become explicit leftovers for the dp-aware
             repair instead of silent mis-cuts.

        Cross-band graph edges are followed naturally (character-following at
        genuine crossings). Deferred + unclaimed nodes are returned grouped as
        their connected pieces, so they enter the normal sample machinery and
        cannot shatter into singletons (and _ensure_connectivity still guards).
        '''
        nks = self.nks
        comp_nodes = set(int(n) for n in sub_graph.nodes)

        ###########################################################################
        # 1. Growth targets: bands with a substantial presence in the component.
        ###########################################################################
        band_count: dict = {}
        for n in comp_nodes:
            band_count[n // nks] = band_count.get(n // nks, 0) + 1
        targets = sorted(b for b, c in band_count.items() if c >= RG_BAND_PRESENCE * nks)
        if not targets:
            return None

        # One group per state node (the P0 certain-sheet quotient was removed
        # Jul 2026 -- atomic mixed sheets starved their twin slot and froze the
        # sweep; see docs/cluster0_dp_seam_correction_proposal.md).
        group_of = {n: n for n in comp_nodes}           # state node -> group key
        members = {n: [n] for n in comp_nodes}          # group key -> [state nodes]
        gks = {n: {n % nks} for n in comp_nodes}        # group key -> {k-points}

        ###########################################################################
        # 2. Seeds: largest k-connected sovereign region per target band
        # (sovereign = the local gap resolves the band's dispersion AND every
        # in-component same-band neighbour overlap is certain), expressed as
        # the set of GROUPS covering that region. With singleton groups
        # the seed IS the region.
        # (The Jul-11 sheet-atomic rewrite seeded "greedily by sovereign
        # mass" per group, which with singleton groups degenerates to mass-1
        # ties -- ONE arbitrary state per band -- and wrecked the in-loop
        # solve: phosphorene NS stuck ~3100-3500 vs run 6's descent to 5.)
        ###########################################################################
        neigh = np.asarray(self.neighbors)
        ks_of_band = {b: set() for b in targets}
        for n in comp_nodes:
            b = n // nks
            if b in ks_of_band:
                ks_of_band[b].add(n % nks)

        regions: dict = {}                              # band -> largest sovereign region (k set)
        for b in targets:
            band_ks = ks_of_band[b]
            sov = set()
            for k in band_ks:
                if self.local_gap[k, b] <= 2.0 * self.band_disp[b]:
                    continue                            # energy does not resolve this band here
                dps = [float(self.connections[k, i, b, b])
                       for i, kn in enumerate(neigh[k])
                       if kn != -1 and int(kn) in band_ks]
                if dps and min(dps) >= RG_SEED_DP:
                    sov.add(k)
            best = None
            while sov:                                  # largest k-connected sovereign region
                start = sov.pop()
                region = {start}
                queue = deque([start])
                while queue:
                    kq = queue.popleft()
                    for kn in neigh[kq]:
                        kn = int(kn)
                        if kn != -1 and kn in sov:
                            sov.discard(kn)
                            region.add(kn)
                            queue.append(kn)
                if best is None or len(region) > len(best):
                    best = region
            if best:
                regions[b] = best

        seeds: dict = {}                                # band -> [seed groups]
        seeded_groups: set = set()
        for b, region in sorted(regions.items(), key=lambda t: -len(t[1])):
            gs: list = []
            ks_held: set = set()
            for k in sorted(region):
                g = group_of[k + b * nks]
                if g in seeded_groups or gks[g] & ks_held:
                    continue
                seeded_groups.add(g)
                ks_held |= gks[g]
                gs.append(g)
            if gs:
                seeds[b] = gs
        if not seeds:
            return None

        ###########################################################################
        # 3. Multi-source cheapest-frontier growth over groups.
        ###########################################################################
        assigned: dict = {}                             # group -> label (a seed band id)
        owns_k = {b: set() for b in seeds}              # label -> k-points it holds
        offers: dict = {}                               # group -> {label: best offered cost}
        deferred: set = set()
        heap: list = []

        def push_frontier(g, label: int) -> None:
            for v in members[g]:
                kv, bv = v % nks, v // nks
                Ev = float(self.eigenvalues[kv, bv])
                for w in sub_graph[v]:
                    w = int(w)
                    gw = group_of[w]
                    if gw == g or gw in assigned or gw in deferred:
                        continue
                    kw, bw = w % nks, w // nks
                    if kw in owns_k[label]:
                        continue                        # cheap pre-filter; full group check at claim
                    if abs(float(self.eigenvalues[kw, bw]) - Ev) > self._local_accept(kw, bw):
                        continue                        # across a local energy wall
                    cost = max(0.0, 1.0 - float(sub_graph[v][w].get('weight', 0.0)))
                    node_offers = offers.setdefault(gw, {})
                    if cost < node_offers.get(label, np.inf):
                        node_offers[label] = cost
                        heappush(heap, (cost, label, gw))

        for b, gs in seeds.items():
            for g in gs:
                assigned[g] = b
                owns_k[b] |= gks[g]
        for b, gs in seeds.items():
            for g in gs:
                push_frontier(g, b)

        while heap:
            cost, label, g = heappop(heap)
            if g in assigned or g in deferred:
                continue
            if gks[g] & owns_k[label]:
                continue                                # k-collision materialized since the push
            # Contested: another label still able to claim g (no k-collision)
            # offered a cost within the margin -- defer instead of guessing.
            rivals = offers.get(g, {})
            if any(l2 != label and (c2 - cost) < RG_DEFER_MARGIN
                   and not (gks[g] & owns_k[l2]) for l2, c2 in rivals.items()):
                deferred.add(g)
                continue
            assigned[g] = label
            owns_k[label] |= gks[g]
            push_frontier(g, label)

        ###########################################################################
        # 4. Output: one node-set per grown band + leftover connected pieces.
        ###########################################################################
        label_nodes = {b: set() for b in seeds}
        claimed: set = set()
        for g, b in assigned.items():
            label_nodes[b].update(members[g])
            claimed.update(members[g])
        parts = [nodes for nodes in label_nodes.values() if nodes]
        leftover = comp_nodes - claimed
        n_pieces = 0
        if leftover:
            for piece in nx.connected_components(sub_graph.subgraph(leftover)):
                parts.append(set(int(x) for x in piece))
                n_pieces += 1
        grown = sorted(seeds)
        self.logger.info(f'{BODY_INDENT}Region-growing: {len(comp_nodes)}-node component '
                         f'({len(members)} group(s)), '
                         f'{len(targets)} fused band(s) {targets} -> grown '
                         f'{[len(label_nodes[b]) for b in grown]} node(s) for bands {grown} '
                         f'(seeds {[sum(len(members[g]) for g in seeds[b]) for b in grown]}), '
                         f'{len(deferred)} contested group(s) deferred, '
                         f'{len(leftover)} leftover in {n_pieces} piece(s)')
        return parts

    def get_components(self, alpha: float=0.5, compute_communities=False) -> None:
        '''
        The make_connections function constructs the graph, in which
        it can detect components well constructed.
            - A component is denominated solved when it has all
              k points attributed.
            - A cluster is a significant component that can not join
              with any other cluster.
            - Otherwise, It is a sample that has to be grouped with
              some cluster.
        
        Parameters
            alpha : float
                The weight of connection to consider for score calculation.
                    score = alpha*<i|j> + (1-alpha)*f(E_i)
        '''

        ###########################################################################
        # Identify connected components inside the GRAPH
        ###########################################################################
        self.logger.info('\n\n\t\tNumber of Components: '.rstrip('\n'))
        self.logger.info(f'\t\t{nx.number_connected_components(self.GRAPH)}')

        resolution = 1
        step = 0.1 
        flag_resolution = True
        max_iter = 5
        iteration = 0

        best_score = 0
        best_iteration = None
        max_solved = 0

        N_g_nks_prev = self.nbnd

        def communites2clusters(communities) -> list[COMPONENT]:
            self.components = [COMPONENT(c,
                                        self.kpoints_index,
                                        self.matrix,
                                        self.dimensions)
                            for c in communities]    # Identify the components

            index_sorted = np.argsort([component.N
                                    for component in self.components])[::-1] # Sort the components by the number of nodes in decreasing order

            ###########################################################################
            # Identify the clusters and samples
            ###########################################################################
            self.solved : list[COMPONENT] = []
            clusters : list[COMPONENT] = []
            samples : list[COMPONENT] = []
            number_nodes = []
            for i in index_sorted:
                # The first biggest components that can not join to the others are identified as clusters
                component = self.components[i]
                number_nodes.append(component.N)
                if component.N == self.nks:
                    #  If the number of nodes inside the component equals the total number of k points, the cluster is considered solved
                    self.solved.append(component)
                    continue
                    
                component.calculate_pointsMatrix()  # Computes the projection into k-space
                component.calc_boundary()           # Undersample the points by representative points identification
                if len(clusters) == 0:
                    # The biggest component if it is not complete then it is the first cluster
                    clusters.append(component)
                    continue
                if not np.any([cluster.validate(component)
                            for cluster in clusters]):
                    # Verification if the component can join other clusters.
                    # If it can not, then it is a cluster.
                    clusters.append(component)
                else:
                    # If it can, then it is a sample.
                    samples.append(component)
            
            self.logger.info(f'\t\tNumber of Communities: {len(self.components)}')
            self.logger.info(f'\t\tPhase 1: {len(self.solved)}/{self.nbnd} Solved')
            self.logger.info(f'\t\tInitial clusters: {len(clusters)} Samples: {len(samples)}')

            number_nodes = np.array(number_nodes)
            N_g_nks = np.sum(number_nodes > self.nks)

            self.logger.info(f'\t\tNumber of components with more than {self.nks} nodes: {N_g_nks}')

            return clusters, samples, N_g_nks

        while flag_resolution:
            ###########################################################################
            # Always start from the RAW connected components: they are the true
            # fused structures. The old path ran Louvain over the whole graph
            # first (when compute_communities), which pre-fragmented a fused
            # multi-band component into mid-size mixed chunks that slipped under
            # the oversized threshold and were never split on physics -- exactly
            # the arbitrary cuts the region-growing is meant to remove. Now every
            # component too big to be a single band is split by seeded
            # region-growing (physics-anchored, per-k bijection enforced during
            # the partition -- see _partition_fused_component); Louvain (with the
            # resolution sweep) remains only as the per-component fallback when
            # region-growing finds no usable seeds.
            ###########################################################################
            if compute_communities:
                self.logger.info(f'\n\t\tResolution: {resolution:.2f}, Iteration: {iteration + 1}')

            used_fallback = False
            sub_graphs = []
            for c in nx.connected_components(self.GRAPH):
                sub_graph = self.GRAPH.subgraph(c)
                if sub_graph.number_of_nodes() <= self.nks:
                    sub_graphs.append(sub_graph)
                    continue
                parts = self._partition_fused_component(sub_graph)
                if parts is None:
                    self.logger.info(f'{BODY_INDENT}Region-growing: no usable seeds in a '
                                     f'{sub_graph.number_of_nodes()}-node component -- Louvain fallback')
                    parts = nx.community.louvain_communities(sub_graph, resolution=resolution, weight='weight')
                    used_fallback = True
                for new_c in parts:
                    sub_graphs.append(sub_graph.subgraph(new_c))


            clusters, samples, N_g_nks = communites2clusters(sub_graphs)

            if not used_fallback:
                # Fully deterministic split (no Louvain ran): the resolution
                # sweep cannot change anything, so one pass is the result.
                break

            total_bands_computed = len(self.solved) + len(clusters)
            if not compute_communities or np.abs(10 - len(clusters) * len(samples)) < 10:
                break

            if total_bands_computed == self.nbnd and N_g_nks == 0:
                break

            if len(clusters) * len(samples) and (best_iteration is None or np.abs(self.nbnd - total_bands_computed) < best_score and len(self.solved) > max_solved and N_g_nks < N_g_nks_prev):
                best_iteration = sub_graphs
                best_score = np.abs(self.nbnd - total_bands_computed)
                max_solved = len(self.solved)
            N_g_nks_prev = N_g_nks                  # track progress for the acceptance test above

            iteration += 1
            if iteration == max_iter:
                if best_iteration is not None:
                    sub_graphs = best_iteration
                    self.logger.info(f'\n\t\tBest Iteration:')
                    clusters, samples, N_g_nks = communites2clusters(sub_graphs)
                else:
                    # No sweep iteration ever qualified as "best": keep the
                    # current split instead of crashing on a None graph list.
                    self.logger.info(f'\n\t\tResolution sweep exhausted with no best '
                                     f'iteration recorded -- keeping the last split')
                break

            if total_bands_computed > self.nbnd:
                resolution  -= step
            else:
                resolution  += step
            step /= 2

        ###########################################################################
        # Assigning samples to clusters by selecting the best option
        ###########################################################################
        def evaluate_sample(iterator):
            if _FE_DEBUG:
                # Per-chunk reset; the chunk's delta is returned below and summed in
                # the parent (forked workers each carry their own copy).
                _FE_STATS.clear(); _FE_STATS.update(_fe_stats_new())
            result = []
            for i_s in iterator:
                sample = samples[i_s]
                # Comparison of each sample to each cluster
                scores = np.zeros(len(clusters))                                    # Storage the score of each cluster with the sample
                for j_s, cluster in enumerate(clusters):
                    if not cluster.validate(sample):
                        # If this sample can not join the cluster, the score is 0
                        continue
                    scores[j_s] = sample.get_cluster_score(cluster,
                                                           self.neighbors,
                                                           self.ENERGIES,
                                                           self.connections,
                                                           alpha=alpha,
                                                           accept_E=self.accept_E,
                                                           disp_scale=self.disp_scale,
                                                           fe_eweight=self.fe_eweight,
                                                           tol=self.tol)  # Calculate the score (new f(E))
                result.append([i_s, [np.max(scores), np.argmax(scores)], sample.scores])

            if _FE_DEBUG:
                result.append(['__FE_STATS__', dict(_FE_STATS), None])           # chunk's f(E) counts
            return result

        count = np.array([0, len(samples) * len(samples)])
        fe_agg = _fe_stats_new()                                                 # f(E) diagnostics for this get_components call
        fe_attached_k = set()                                                    # k-points joined this call (Tier 4)
        while len(samples) > 0:
            evaluate_samples = np.zeros((len(samples), 2))                          # Samples' scores storage
            self.logger.debug(f'Len samples: {len(samples)}')
            self.logger.debug(f'Len clusters: {len(clusters)}')
            if len(clusters) == 0 and len(samples) > 0:
                self.logger.debug('No more clusters')
                self.logger.debug(f'Len samples: {len(samples)}')
                self.logger.debug(f'Size samples: {samples[0].N}')
                self.logger.debug(f'Number solved: {len(self.solved)}')
                break

            evaluate_samples_result = self.parallelize('\t\tClustering Samples', evaluate_sample, range(len(samples)), per_actual=count[0], N_total=count[1])
            count[0] += len(samples)
            for i_s, res, sample_scores in evaluate_samples_result:
                if i_s == '__FE_STATS__':
                    _fe_stats_merge(fe_agg, res)                      # res = this chunk's f(E) counts
                    continue
                samples[i_s].scores = sample_scores
                evaluate_samples[i_s] = np.array(res)                 # Store the best cluster's score

            for cluster in clusters:
                # Flag used to identify if the score must be calculated again
                cluster.was_modified = False
                
            bn_list = []
            args_list = []
            clusters_completed = []
            args_sort = np.argsort(evaluate_samples[:, 0])[::-1]                       # Sort the samples by the score
            for arg_max in args_sort:
                score, bn = evaluate_samples[arg_max]                                   # Get the values
                bn = int(bn)
                if score <= 0:
                    # No admissible cluster for this sample (every validate() failed,
                    # so argmax landed on cluster 0 by default). Joining it anyway
                    # would break the non-overlap invariant; args_sort is descending,
                    # so every remaining sample is inadmissible too.
                    break
                if bn in bn_list:
                    # If the cluster was already modified, the score is not updated
                    break
                args_list.append(arg_max)
                bn_list.append(bn)
                sample = samples[arg_max]                                               # Get the sample
                count[0] += 1                                                           # Update the counter
                clusters[bn].join(sample)                                               # Join the sample to the best cluster
                clusters[bn].was_modified = True
                if _FE_DEBUG:
                    fe_attached_k.update((np.array(sample.GRAPH.nodes) % self.nks).tolist())  # Tier 4
    
                self.logger.percent_complete(count[0], count[1], title='\tClustering Samples')
                self.logger.debug(f'\t\t{count[0]}/{count[1]} Sample corrected: {score}')
                if clusters[bn].N == self.nks:
                    #  If the number of nodes inside the component equals the total number of k points, the cluster is considered solved
                    clusters_completed.append(bn)
                    self.logger.debug('\n\tCluster Solved')
                    
            # Eliminate the samples that were joined to a cluster
            samples = [samples[arg] for arg in args_sort if arg not in args_list]

            if not args_list:
                # Nothing joined this round: every remaining sample is inadmissible
                # for every cluster, so another round cannot change anything (the
                # loop would spin forever). Their nodes stay unattributed and flow
                # into the repair / completeness passes.
                self.logger.info(f'\t\t{len(samples)} unattachable sample(s) left '
                                 f'unclustered (no admissible cluster)')
                break

            new_clusters = []
            for bn, cluster in enumerate(clusters):
                # Remove the solved clusters
                if bn not in clusters_completed:
                    new_clusters.append(cluster)
                    continue
                self.solved.append(cluster)

            clusters = new_clusters

        self.logger.percent_complete(count[1], count[1], title='\tClustering Samples')
        self.fE_attached_k = np.array(sorted(fe_attached_k), dtype=int)          # for the Tier-4 outcome log
        self._log_fE_stats(fe_agg)                                               # Tier 1 + 2 per-iteration block
        self.logger.info(f'\t\tPhase 2: {len(self.solved)}/{self.nbnd} Solved')

        if len(self.solved)/self.nbnd < 1:
            self.logger.info(f'\t\tNew clusters: {len(clusters)}')

        self.clusters : list[COMPONENT] = clusters

    def obtain_output(self) -> None:
        '''
        This function prepares the final data structures
        that are essential to other programs.
        '''

        ###########################################################################
        # Conflict-aware slot assignment (option 3: 0.6*dot-product + 0.4*energy).
        #
        # Each component (a "solved" full-grid band or a partial cluster) is placed
        # into the output column ``bn`` (a raw band index) that maximises its
        # fitness:
        #     fitness(C, bn) = W_DP * dp_coherence(C) + W_E * energy_fit(C, bn)
        # restricted to the bands the component actually contains AND to those
        # whose energy line stays within the gross-jump limit ``self.accept_E``
        # (the "limit in difference of energy"). When two components want the same
        # column the fitter one keeps it and the loser is re-placed at its next-best
        # admissible band -- the conflict solver is re-applied to the loser. Every
        # component only ever advances down its own preference list, so this is a
        # deferred-acceptance assignment that always terminates; a component that
        # exhausts its admissible bands is left unattributed for a later iteration.
        #
        # NOTE: self.bands_final is intentionally NOT reset here. Stale entries from
        # earlier iterations therefore persist (cross-iteration carry-over is under
        # separate review); this change only makes THIS pass's contested columns be
        # decided by fitness instead of by arrival order.
        ###########################################################################
        W_DP, W_E = 0.6, 0.4

        def _dp_coherence(comp) -> float:
            # Mean intra-component dot product: how coherent the band trajectory is.
            kset = set(int(k) for k in comp.k_points)
            total = 0.0
            n = 0
            for k in comp.k_points:
                bk = comp.bands_number[k]
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    if k_neig == -1 or int(k_neig) not in kset:
                        continue
                    bkn = comp.bands_number.get(k_neig)
                    if bkn is None:
                        continue
                    total += self.connections[k, i_neig, bk, bkn]
                    n += 1
            return total / n if n else 0.0

        def _energy_fit(comp, bn: int) -> Tuple[bool, float]:
            # Agreement of the component's energy trajectory with band-slot bn's
            # energy line. Returns (admissible, score); ``admissible`` is the hard
            # energy-difference gate: mean |E_C - E(.,bn)| <= accept_E.
            resid_sum = 0.0
            score_sum = 0.0
            n = 0
            for k in comp.k_points:
                Ec = float(self.eigenvalues[k, comp.bands_number[k]])
                Es = float(self.eigenvalues[k, bn])
                resid_sum += abs(Ec - Es)
                score_sum += _energy_continuity_score(Ec, Es, accept_E=self.accept_E, disp_scale=self.disp_scale)
                n += 1
            if n == 0:
                return False, 0.0
            admissible = (resid_sum / n) <= self.accept_E
            return admissible, score_sum / n

        def _candidates(comp) -> list:
            # Ranked (band, fitness) preference list over the component's own raw
            # bands, energy-admissible only, best first.
            comp.get_bands()                                                        # sets comp.bands / bands_number / k_points
            dp = _dp_coherence(comp)
            prefs = []
            for bn in comp.bands:
                admissible, eng = _energy_fit(comp, int(bn))
                if not admissible:
                    continue
                prefs.append((int(bn), W_DP * dp + W_E * eng))
            prefs.sort(key=lambda t: t[1], reverse=True)
            return prefs

        def _assign(components, forbidden: set) -> dict:
            # Deferred-acceptance assignment with eviction. ``forbidden`` is the set
            # of columns already owned by an earlier phase (immovable here).
            prefs = [_candidates(c) for c in components]
            owner = {}                                                              # column -> component index holding it
            owner_fit = {}                                                          # column -> that holder's fitness
            placed = {}                                                             # component index -> column
            rank = [0] * len(components)
            queue = deque(range(len(components)))
            guard = 0
            guard_max = (len(components) + 1) * (self.total_bands + 2) + 16
            while queue:
                guard += 1
                if guard > guard_max:                                               # belt-and-suspenders: cannot loop forever
                    self.logger.warning(f'{BODY_INDENT}conflict solver guard hit; '
                                        f'{len(queue)} component(s) left unattributed')
                    break
                ci = queue.popleft()
                pl = prefs[ci]
                r = rank[ci]
                while r < len(pl):
                    bn, fit = pl[r]
                    if bn in forbidden:                                             # owned by an earlier phase
                        r += 1
                        continue
                    if bn not in owner:                                             # free column: take it
                        owner[bn] = ci; owner_fit[bn] = fit
                        placed[ci] = bn; rank[ci] = r
                        break
                    if fit > owner_fit[bn]:                                         # better suited: evict the holder
                        cj = owner[bn]
                        placed.pop(cj, None)
                        owner[bn] = ci; owner_fit[bn] = fit
                        placed[ci] = bn; rank[ci] = r
                        rank[cj] += 1                                               # loser resumes past the lost band
                        queue.append(cj)                                           # re-apply the conflict solver to it
                        break
                    r += 1                                                          # lose this column, try our next band
                else:
                    rank[ci] = len(pl)                                              # exhausted: unattributed this iteration
            return placed

        # Phase 1: full-grid solved components compete among themselves.
        solved_slot = _assign(self.solved, set())
        # Phase 2: partial clusters fill the columns the solved bands did not take.
        cluster_slot = _assign(self.clusters, set(solved_slot.values()))

        ###########################################################################
        # Write the resolved attribution and the per-k signal.
        ###########################################################################
        for ci, solved in enumerate(self.solved):
            bn = solved_slot.get(ci)
            if bn is None:                                                          # unattributed this iteration
                continue
            bands = solved.get_bands()                                              # Getting the k-points' raw bands inside the solved cluster
            self.bands_final[solved.k_points, bn] = bands                           # Update the resultant bands' attribution array

            for k in solved.k_points:
                # For each k-point is calculate the solution score
                bn1 = solved.bands_number[k]                                        # The k-point's band
                connections = []                                                    # The array that store the dot-product with the k-point's neighbors
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    # Obtain the dot-product with each neighbor
                    if k_neig == -1 or k_neig not in solved.k_points:
                        continue
                    if solved.bands_number.get(k_neig) is None:
                        solved.calculate_values()
                    bn2 = solved.bands_number[k_neig]                               # The neighbor's band
                    connections.append(self.connections[k, i_neig, bn1, bn2])       # <k, k neighbor>

                self.signal_final[k, bn] = evaluate_result(connections)             # Computes the k-point's signal

        for ci, cluster in enumerate(self.clusters):
            bn = cluster_slot.get(ci)
            if bn is None:                                                          # unattributed this iteration
                continue
            bands = cluster.get_bands()                                             # Getting the k-points' raw bands inside the cluster
            self.bands_final[cluster.k_points, bn] = bands                          # Update the resultant bands' attribution array
            for k in cluster.k_points:
                # For each k-point is calculate the solution score
                bn1 = cluster.bands_number[k]                                       # The k-point's band
                connections = []                                                    # The array that store the dot-product with the k-point's neighbors
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    # Obtain the dot-product with each neighbor
                    if k_neig == -1:
                        continue
                    if k_neig not in cluster.k_points:
                        # If the neighbor does not exist inside the cluster, the dot-product is 0
                        connections.append(0)
                        continue
                    bn2 = cluster.bands_number[k_neig]                              # The neighbor's band
                    connections.append(self.connections[k, i_neig, bn1, bn2])       # <k, k neighbor>

                self.signal_final[k, bn] = evaluate_result(connections)             # Computes the k-point's signal


        ###########################################################################
        # Scoring the result.
        # Signaling and storage of degenerate k-points.
        ###########################################################################
        self.degenerate_final = []                                                 # Final degenerates k-points
        for d1, d2 in self.degenerates:
            # Signaling the numerically degenerate points Ei ~ Ej
            k1 = d1 % self.nks                                              # k point
            bn1 = d1 // self.nks                                            # band
            k2 = d2 % self.nks                                              # k point
            bn2 = d2 // self.nks                                            # band
            Bk1 = self.bands_final[k1] == bn1                               # Find in which  band the k-point was attributed
            Bk2 = self.bands_final[k2] == bn2                               # Find in which  band the k-point was attributed
            bn1 = np.argmax(Bk1) if np.sum(Bk1) != 0 else bn1               # Final band
            bn2 = np.argmax(Bk2) if np.sum(Bk2) != 0 else bn2               # Final band

            self.signal_final[k1, bn1] = DEGENERATE                         # Signal k_point as Degenerate
            self.signal_final[k2, bn2] = DEGENERATE                         # Signal k_point as Degenerate

            k_neigs = self.neighbors[k1]                                    # k-point's neighbors
            flag_neig = k_neigs != -1                                       # Flag to obtain the allowed neighbors
            i_neigs = np.arange(self.number_neighbors)[flag_neig]                         # Neighbors' index
            k_neigs = k_neigs[flag_neig]                                    # Allowed neighbors

            if len(k_neigs) == 0:
                # If there are no neighbors the k-point's score is 0
                continue
            bn1 = self.bands_final[k1, bn1]                                 # K-point's band
            bn2 = self.bands_final[k2, bn2]                                 # K-point's band
            if bn1 < 0 or bn2 < 0:
                # Unattributed slot: no dp evidence to test (connections[..., -1, ...]
                # would silently read the LAST band) and a [k, -1, b] row must never
                # reach degeneratefinal.npy. The DEGENERATE signal above stands.
                continue
            dps = self.connections[k1, i_neigs, bn1, bn2]        # Dot-product between the k-point and their neighbors
            if np.any(np.logical_and(dps >= 0.5, dps <= 0.8)):
                # It is considered degenerate if the k-point has some 
                # neighbor's dot-product between 0.5 and 0.8
                self.degenerate_final.append([k1, bn1, bn2])                # Storage the degenerate k-point



        for bn in range(self.total_bands):
            # Calculating the score of the result
            score = 0
            for k in range(self.nks):
                # Evaluate each k-point
                if self.signal_final[k, bn] == NOT_SOLVED:
                    # If this k-point had not been solved the analysis can not be done
                    continue
                kneigs = self.neighbors[k]                                                  # k-point's neighbors
                flag_neig = kneigs != -1                                                    # Flag to obtain the allowed neighbors 
                i_neigs = np.arange(self.number_neighbors)[flag_neig]                                     # Neighbors' index
                kneigs = kneigs[flag_neig]                                                  # Allowed neighbors
                flag_neig = self.signal_final[kneigs, bn] != NOT_SOLVED                     # Flag to obtain only attributed neighbors
                i_neigs = i_neigs[flag_neig]                                                # Update Neighbors' index
                kneigs = kneigs[flag_neig]                                                  # Update neighbors
                if len(kneigs) == 0:
                    # If there are no neighbors the k-point's score is 0
                    continue
                bn_k = self.bands_final[k, bn]                                              # K-point's band
                bn_neighs = self.bands_final[kneigs, bn]                                    # Neighbors' bands
                k = np.repeat(k, len(kneigs))                                               # Array with the same k-point
                bn_k = np.repeat(bn_k, len(kneigs))                                         # Array with the same K-point's band
                dps = self.connections[k, i_neigs, bn_k, bn_neighs]                         # The array with the dot-product between the k-point and their neighbors
                score += np.mean(dps)                                                       # Update the band score
            score /= self.nks                                                               # Compute the mean socore
            self.final_score[bn] = score                                                    # Storage the band score

        self.degenerate_final = np.array(self.degenerate_final)
    
    def print_report(self, signal_report: np.ndarray, description:str, show:bool=True, header_text:list=None) -> (str, np.ndarray):
        '''
        Shows on screen the report for each band.

        Parameters
            signal_report : array_like
                An array with the k-point's signal information.
            description : string
                Describes the table
            show : bool
                If it is true then the table is shown. Otherwise, the string and the report are returned.
        Return
            final_report : string
        '''
        final_report = f'\n{TITLE_INDENT}====== {description} ======\n'
        bands_report = []
        # The set of signal codes (and therefore the column layout) depends on
        # which array is being reported, so the caller passes the matching header.
        if header_text is None:
            header_text = VALIDATE_RESULT_HEADER
        n_codes = len(header_text)
        ###########################################################################
        # Prepare the summary for each band
        ###########################################################################
        for bn in range(self.nbnd):
            band_result = signal_report[:, bn]                              # Obtain all k-point' signals for band bn
            report = [np.sum(band_result == s) for s in range(n_codes)]     # Set the band report
            report.append(np.round(self.final_score[bn], 4))                # Set the final score
            bands_report.append(report)

            self.logger.debug(f'\t\t\tNew Band: {bn}\tnr fails: {report[0]}')
            if report[0] and self.logger.level <= logging.DEBUG:        # list the failing k-points (were only counted)
                not_ks = np.where(band_result == NOT_SOLVED)[0]
                self.logger.debug(f'\t\t\t  band {bn} NOT at {len(not_ks)} k-point(s): {not_ks.tolist()}')

        ###########################################################################
        # Set up the data representation
        ###########################################################################
        bands_report = np.array(bands_report)

        header = list(header_text) + ['Score']
        n_spaces = len(str(np.max(bands_report[:, -1]))) + 4

        # Visible (indent-free) header line, so the underline matches its width.
        header_line = ' Band | ' + ''.join(f"{h:^{n_spaces}}" for h in header[:-1])
        header_line += f'{header[-1]:>8}'

        final_report += f'\n{BODY_INDENT} Signaling: how many events in each band signaled.\n'
        final_report += _format_legend(header)                          # meaning of every column
        final_report += f'\n{BODY_INDENT}{header_line}'
        final_report += f'\n{BODY_INDENT}' + '-' * len(header_line)

        for bn, report in enumerate(bands_report):
            row = f' {bn+self.min_band:<4d} | '
            row += ''.join(f"{int(r):^{n_spaces}d}" for r in report[:-1])
            row += f'{report[-1]:>8.4f}'
            final_report += f'\n{BODY_INDENT}{row}'
        final_report += '\n'
        if show:
            self.logger.info(final_report)              # Show on screen
            return None
        return final_report, bands_report
    
    def correct_signal(self, last=False) -> None:
        '''
        This function evaluates the k-point signal calculated on previous analysis and attributes
        a new signal value depending only on energy continuity.
        '''
        del self.GRAPH              # Clean memory
        OTHER = 3
        MISTAKE = 1

        ###########################################################################
        # Set up the necessary data structures
        ###########################################################################


        self.correct_signalfinal = np.copy(self.signal_final)                           # New array to store the corrected signal
        self.correct_signalfinal[self.signal_final == CORRECT] = CORRECT-1              # Change the CORRECT signal to CORRECT - 1

        if last:
            # Final pass: validate EVERY attributed slot, not only the POTENTIAL_* ones.
            # In-loop, CORRECT slots skip re-evaluation (cheap, and it keeps the solver
            # dynamics stable), but that exempts whichever side of a label swap was
            # assigned FIRST from ever being energy-checked: of a swapped pair only the
            # slot filled last gets flagged and its partner ships as clean (MoS2 bands
            # 12/13: slot 13 held the swapped band at the same 5 k-points with zero dp
            # support and still reported CORRECT). DEGENERATE cells keep their marker
            # (ROTATE grading) and FORCED cells keep their audit trail (solve() stamps
            # them FORCED_CONTINUITY right after this returns).
            sel = ((self.bands_final >= 0) &
                   (self.signal_final != DEGENERATE) &
                   (self.signal_final != FORCED))
            ks, bnds = np.where(sel)
        else:
            ks_pC, bnds_pC = np.where(self.signal_final == POTENTIAL_CORRECT)           # Select the points marked as POTENTIAL_CORRECT
            ks_pM, bnds_pM = np.where(self.signal_final == POTENTIAL_MISTAKE)           # Select the points marked as POTENTIAL_MISTAKE

            ks = np.concatenate((ks_pC, ks_pM))                                         # Join all k-points
            bnds = np.concatenate((bnds_pC, bnds_pM))                                   # Join the k-points' bands

        ###########################################################################
        # Correct the k-point's signal
        ###########################################################################
        def evaluate_points_chunk(iterator: Union[list, np.ndarray]) -> list:
            '''
            Evaluate the energy-continuity signal for a chunk of (k, bn) pairs.
            It only reads shared (read-only) arrays, so it is safe to run in parallel.
            '''
            chunk_result = []
            for k, bn in iterator:
                signal, scores = evaluate_point(self.dimensions, k, bn, self.kpoints_index,
                                                self.matrix,
                                                self.bands_final, self.eigenvalues,
                                                accept_E=self.accept_E, disp_scale=self.disp_scale,
                                                local_gap=self.local_gap, band_disp=self.band_disp)  # Obtain the new signal
                chunk_result.append([k, bn, signal, scores])
            return chunk_result

        # The evaluation of each (k, bn) point is independent, so it is parallelized
        kbnds = list(zip(ks, bnds))
        evaluated_points = self.parallelize('\t\tCorrecting signal', evaluate_points_chunk, kbnds) \
            if len(kbnds) > 0 else []

        for k, bn, signal, scores in evaluated_points:
            # Iterate over all k-point signaled as potential points where the energy continuity may fail
            
            if last and self.final_score[bn] > 0.96 and signal == MISTAKE:
                signal = OTHER

            self.correct_signalfinal[k, bn] = signal                                    # Store this new signal
            if signal <= OTHER:                 # log only actionable signals (NOT/MIS/DEG/OTH); skip ~11k CORRECT lines/iter
                self.logger.debug(f'K point: {k} Band: {bn}    New Signal: {signal} Directions: {scores}')

        if last:
            # Observability for the widened final-pass scope: how many slots the
            # in-loop validation had promoted to CORRECT actually fail energy
            # continuity once checked (they are invalidated and refilled below).
            demoted = int(np.sum((self.signal_final == CORRECT) &
                                 (self.correct_signalfinal == MISTAKE)))
            if demoted:
                self.logger.info(f'{BODY_INDENT}Final revalidation: {demoted} in-loop CORRECT '
                                 f'slot(s) fail energy continuity and were flagged')

        ###########################################################################
        # Create a new problem for another solver iteration
        ###########################################################################
        k_error, bn_error = np.where(self.correct_signalfinal == MISTAKE)           # Identify Mistakes
        k_other, bn_other = np.where(self.correct_signalfinal == OTHER)             # Identify k-points with some discontinuity

        ###########################################################################
        # Invalidate the canvas only where the attribution failed energy continuity
        # (NOT_SOLVED / MISTAKE -- not OTHER). A stale or duplicated value there can
        # no longer (a) survive in the output array nor (b) seed a bogus continuity
        # edge in the rebuild below; the cell is re-derived next iteration. CORRECT
        # cells and the completed bands are left untouched. The continuity-edge
        # builder below guards against the resulting -1 (an invalidated cell
        # originates no edge).
        ###########################################################################
        wrong = (self.correct_signalfinal == NOT_SOLVED) | (self.correct_signalfinal == MISTAKE)
        self.bands_final[wrong] = -1

        # Complete energy-isolated near-degenerate groups by a within-group bijection
        # BEFORE the generic bijection. Inside a degenerate manifold the per-band label
        # is gauge, so the dp graph can split a Kramers/SOC pair across components and
        # leave a slot empty even where the dot product is clean (band-7 NOT=38). This
        # fills those slots from the group's own bands and never touches isolated bands,
        # so genuine crossings of well-separated bands stay dp-tracked. See the method.
        self._resolve_degenerate_groups()

        # Restore the per-k bijection: resolve any band attributed to more than one
        # slot, reassigning each losing slot to a band missing from that k-point's
        # row (best energy fit). Runs before the graph rebuild so the continuity
        # edges below are built from a duplicate-free canvas.
        self._enforce_bijection()

        total_not_solved = np.sum(self.correct_signalfinal == NOT_SOLVED) + np.sum(self.correct_signalfinal == MISTAKE)
        self.logger.info(f'\n{BODY_INDENT}Total not solved: ' + str(total_not_solved) + '\n')
        self.total_not_solved = total_not_solved        # exposed so solve() can pace the alpha sweep by convergence

        if last:
            # Final pass: the graph rebuild below (continuity edges + error-region
            # dot-product edges + connectivity guard) is consumed only by the NEXT
            # get_components, which never runs after this -- skip the dead work and
            # just verify the bijection invariant on what is shipped.
            self._verify_no_duplicates()
            return

        ks = np.concatenate((k_error, k_other))                                     # Join the k-points marked as a mistake or other signal
        bnds = np.concatenate((bn_error, bn_other))                                 # Join the k-points' bands

        bands_signaling = np.zeros((self.total_bands, *self.matrix.shape), int)     # The array used to identify the k-points' projection in k-space
        k_index = self.kpoints_index[ks]                                            # k-points' indeces

        if self.dimensions == 1:
            ik = k_index

        if self.dimensions >= 2:
            ik = k_index[:, 0]
            jk = k_index[:, 1]
        if self.dimensions == 3:
            kk = k_index[:, 2]

        # Mark the k-points' projection in k-space
        if self.dimensions == 1:
            bands_signaling[bnds, ik] = 1   
        elif self.dimensions == 2:
            bands_signaling[bnds, ik, jk] = 1   
        elif self.dimensions == 3:
            bands_signaling[bnds, ik, jk, kk] = 1

        self.GRAPH = nx.Graph()                                                     # The new Graph
        self.GRAPH.add_nodes_from(np.arange(len(self.vectors)))                     # Set the nodes
        edges = []

        # Near-degenerate cells for the glue test below: inside a collapsed gap
        # (adjacent gap < degen_ethr) a vanishing overlap is gauge, not evidence.
        _close = self._near_degenerate_mask()

        # Degenerate K-POINTS only (first column of the [k, bn1, bn2] triples):
        # a plain `kp in self.degenerate_final` would match kp against the BAND
        # columns too, spuriously flagging every k-point whose index equals a
        # stored band number. Set lookup is also O(1) per k-point.
        _df = np.asarray(self.degenerate_final)
        degen_ks = set(int(x) for x in _df[:, 0]) if _df.ndim == 2 and len(_df) else set()

        def continuity_edge(kp, kneig, b_p, b_pn):
            '''
            Append the continuity edge (same slot at adjacent k-points), UNLESS the
            step it asserts is GLUE: dot product < DP_FORBID with both endpoints
            energy-resolved (outside any near-degenerate group). Such a step is an
            assignment-agreement across a vanishing overlap -- e.g. the 12/13 seam
            staircase, dp ~0.02 across a 13 mRy jump -- and re-fused the
            mis-attributed patch every iteration. Steps inside near-degenerate /
            crossing corridors keep their edge regardless of dp (there the label
            is gauge and slot chains legitimately switch sheet-owners).

            The weight stays 1 (accumulating += 1 on repeat, below) ON PURPOSE:
            these edges are the solver's RATCHET -- they re-inject the previous
            iteration's accepted assignment at region-growing cost 0, which is
            what lets each iteration re-claim the full bands and converge.
            Replacing them with dp-metric weights removed the ratchet and the
            in-loop descent froze at its first fixed point (phosphorene: 3312
            not-solved forever, vs 18 with the ratchet; bands 10-24 collapsed).
            '''
            i_neig = np.where(self.neighbors[kp] == kneig)[0]
            if len(i_neig) == 0:
                return
            dp = float(self.connections[kp, i_neig[0], b_p, b_pn])
            if dp < DP_FORBID and not (_close[kp, b_p] or _close[kneig, b_pn]):
                return                                              # resolved forbidden step: glue, no edge
            edges.append((int(kp + b_p * self.nks), int(kneig + b_pn * self.nks)))

        for bn, band in enumerate(bands_signaling):
            # For each band construct the new graph
            identify_points = band > 0          # The marked points are considered
        
            if self.dimensions == 1:
                directions = np.array([1])                                                  # Auxiliary array with the directions to evaluate the edges' existence
                # If the problem is 1D, the graph is built by the neighbors
                for k, need_correction in enumerate(identify_points):
                    # For each not identified point the graph is built
                    kp = self.matrix[k]                                                         # The k-point on position k in k-space
                    if need_correction and kp not in degen_ks:
                        # If the point was identified as an error or as degenerate
                        # It does not have an edge in the graph
                        continue
                    for direction in directions:
                        # It is verified for each direction (Down, Right) if the points,
                        # and their neighbors belong to the same band
                        kn = k + direction                                                      # Neighbor's idices in k-space
                        if kn >= self.matrix.shape[0]:
                            # The neighbor is outside of the boundaries
                            continue
                        kneig = self.matrix[kn]                                                 # Neighbor k-point
                        if not identify_points[kn]:
                            b_p = self.bands_final[k, bn]
                            b_pn = self.bands_final[kn, bn]
                            if b_p < 0 or b_pn < 0:
                                continue                                                        # invalidated cell -> no continuity edge
                            continuity_edge(kp, kneig, b_p, b_pn)                               # weight-1 ratchet edge unless glue (see continuity_edge)

            if self.dimensions == 2:
                directions = np.array([[1, 0], [0, 1]])                                     # Auxiliary array with the directions to evaluate the edges' existence
                for ik, row in enumerate(identify_points):
                    for jk, need_correction in enumerate(row):
                        # For each not identified point the graph is built
                        kp = self.matrix[ik, jk]                                                        # The k-point on position ik, jk in k-space
                        if need_correction and kp not in degen_ks:
                            # If the point was identified as an error or as degenerate
                            # It does not have an edge in the graph
                            continue
                        for direction in directions:
                            # It is verified for each direction (Down, Right) if the points,
                            # and their neighbors belong to the same band
                            ikn, jkn = np.array([ik, jk]) + direction                                   # Neighbor's idices in k-space
                            if ikn >= self.matrix.shape[0] or jkn >= self.matrix.shape[1]:
                                # The neighbor is outside of the boundaries
                                continue
                            kneig = self.matrix[ikn, jkn]                                               # Neighbor k-point
                            if not identify_points[ikn, jkn]:
                                b_p = self.bands_final[kp, bn]                          # The kpoint's attributed band
                                b_pn = self.bands_final[kneig, bn]                      # The neighbor's attributed band
                                if b_p < 0 or b_pn < 0:
                                    continue                                            # invalidated cell -> no continuity edge
                                continuity_edge(kp, kneig, b_p, b_pn)                   # weight-1 ratchet edge unless glue (see continuity_edge)

            if self.dimensions == 3:
                directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])                        # Auxiliary array with the directions to evaluate the edges' existence
                for ik, plane in enumerate(identify_points):
                    for jk, row in enumerate(plane):
                        for kk, need_correction in enumerate(row):
                            # For each not identified point the graph is built
                            kp = self.matrix[ik, jk, kk]                                                        # The k-point on position ik, jk in k-space
                            if need_correction and kp not in degen_ks:
                                # If the point was identified as an error or as degenerate
                                # It does not have an edge in the graph
                                continue
                            for direction in directions:
                                # It is verified for each direction (Down, Right) if the points,
                                # and their neighbors belong to the same band
                                ikn, jkn, kkn = np.array([ik, jk, kk]) + direction                            # Neighbor's idices in k-space
                                if ikn >= self.matrix.shape[0] or jkn >= self.matrix.shape[1] or kkn >= self.matrix.shape[2]:
                                    # The neighbor is outside of the boundaries
                                    continue
                                kneig = self.matrix[ikn, jkn, kkn]                                            # Neighbor k-point
                                if not identify_points[ikn, jkn, kkn]:
                                    b_p = self.bands_final[kp, bn]                            # The kpoint's attributed band
                                    b_pn = self.bands_final[kneig, bn]                       # The neighbor's attributed band
                                    if b_p < 0 or b_pn < 0:
                                        continue                                             # invalidated cell -> no continuity edge
                                    continuity_edge(kp, kneig, b_p, b_pn)                     # weight-1 ratchet edge unless glue (see continuity_edge)

        # Which k-points still carry an error in any band. A k-point is "bad" if any of
        # its slots failed the energy-continuity validation (NOT_SOLVED / MISTAKE / OTHER);
        # these are the nodes that must originate fresh dot-product edges.
        bad_mask = ((self.correct_signalfinal == NOT_SOLVED) |
                    (self.correct_signalfinal == MISTAKE) |
                    (self.correct_signalfinal == OTHER))
        bad_k = np.where(np.any(bad_mask, axis=1))[0]

        if len(bad_k) > 0:
            ###########################################################################
            # Error-region-only rebuild. Only the nodes of the bad k-points (across all
            # bands, since the band identity there is uncertain) originate fresh
            # dot-product edges, so the bad region can re-cluster and re-attach to good
            # bands at its boundary. The good bands keep ONLY their same-band continuity
            # edges (added in the loop below) and survive intact as components -- unlike a
            # full-graph rebuild, whose dense cross-band edges + Louvain shattered the
            # done bands on every iteration.
            #
            # This rebuild used to be gated on `total_not_solved > 1000`. That gate was
            # REMOVED: once not_solved dropped below 1000 the code fell through to the
            # continuity-only `else`, which omits these dot-product edges. With a freshly
            # reset graph (every node isolated) plus only sparse continuity edges, the
            # graph fragmented into thousands of singleton components and get_components
            # then ran the O(samples^2) assignment (count = len(samples)**2) -- a ~2.4e8
            # evaluation blow-up (~11 h for a single step on MoS2). The dot-product
            # reconnection must stay active for the whole descent, so it now fires whenever
            # ANY error remains; the alpha sweep is paced by convergence in solve()
            # instead of by this point count.
            ###########################################################################
            all_bands = np.arange(self.total_bands)
            error_nodes = (bad_k[:, None] + all_bands[None, :] * self.nks).ravel()
            self.logger.info(f'{BODY_INDENT}Error-region rebuild: {len(bad_k)} bad k-point(s), '
                             f'{len(error_nodes)} node(s) of {len(self.vectors)} '
                             f'({100.0*len(error_nodes)/max(len(self.vectors),1):.0f}%)')
            self.make_connections(self.tol, not_first_iteration=True, node_subset=error_nodes)
            self.repeat_communities = True
            for p, pn in edges:
                # For each edge, the graph is built. The weight-1 (+= 1 on repeat)
                # ratchet is intentional -- see continuity_edge; only the glue
                # edges (resolved dp < DP_FORBID steps) are excluded above.
                if self.GRAPH.has_edge(p, pn):
                    # If the edge already exists, the weight is updated
                    self.GRAPH[p][pn]['weight'] += 1                    # This value can be updated later
                else:
                    # Otherwise, the edge is created
                    self.GRAPH.add_edge(p, pn, weight=1)
        else:
            # No remaining errors: the continuity edges already connect each band into a
            # single component, so plain connected-components clustering is correct and
            # cheap. No dot-product edges are needed here.
            self.repeat_communities = False
            self.GRAPH.add_edges_from(edges)                                                        # Build the identified edges

        # Structural anti-shatter safety net: guarantee no node is left isolated,
        # independent of the `tol` threshold or which branch built the graph above.
        self._ensure_connectivity()

        # Invariant check before the next solver iteration: every k-point must map
        # each band index to at most one slot (a per-k bijection over attributed
        # cells). The wrong-cell invalidation above removes the NOT_SOLVED/MISTAKE
        # copies; anything still duplicated here is surfaced as a warning.
        self._verify_no_duplicates()

    def _verify_no_duplicates(self) -> int:
        '''
        Verify that ``self.bands_final`` is a per-k bijection: at each k-point no
        band index is attributed to more than one slot. Unattributed cells (-1) are
        ignored. Logs an explicit OK/violation line and returns the number of excess
        (duplicate) attributions so a caller could gate on it.
        '''
        bf = self.bands_final
        nk, nb = bf.shape
        # counts[k, b] = how many slots at k hold band index b (ignoring -1).
        counts = np.zeros((nk, self.total_bands), dtype=int)
        for col in range(nb):
            valid = bf[:, col] >= 0
            np.add.at(counts, (np.where(valid)[0], bf[valid, col]), 1)
        dup = counts > 1
        excess = int((counts[dup] - 1).sum())
        if excess:
            dup_k = int(np.any(dup, axis=1).sum())
            self.logger.warning(f'{BODY_INDENT}Duplicate-attribution check: bijection VIOLATED -- '
                                f'{dup_k} k-point(s) hold a band index in >1 slot '
                                f'({excess} excess attribution(s))')
        else:
            self.logger.info(f'{BODY_INDENT}Duplicate-attribution check: OK '
                             f'(every k-point maps each band to at most one slot)')
        return excess

    def _verify_no_cliffs(self) -> int:
        '''
        Final continuity invariant: no band-slot may ship a raw-band SWITCH across
        an energy step larger than the local continuity gate (see
        ``_cliff_seam_edges``) -- the sub-accept_E cliffs the flat gross-jump gate
        is too loose to catch. Recomputed AFTER every relocation pass, so what
        remains is exactly the cliffs no pairwise exchange could remove (a genuine
        two-sided braid, or a defect the flood refused). Stores the per-slot count
        (``self.cliff_counts``) and the surviving edges (``self.cliff_edges``) so
        the report downgrades those bands to CHECK/FAIL instead of shipping a
        silent USABLE band, and logs a loud warning. Soft like the bijection check
        -- it does not raise. Returns the number of surviving cliff edges.
        '''
        edges = self._cliff_seam_edges()
        nslots = self.bands_final.shape[1]
        counts = np.zeros(nslots, dtype=int)
        for _jump, _k, _kn, s in edges:
            counts[int(s)] += 1
        self.cliff_edges = edges
        self.cliff_counts = counts
        if edges:
            slots = sorted({int(s) for _j, _k, _kn, s in edges})
            wj, wk, wkn, ws = edges[0]
            self.logger.warning(
                f'{BODY_INDENT}Cliff check: {len(edges)} intra-slot energy cliff(s) '
                f'survive in slot(s) {slots} (worst {wj:.4g} Ry at slot {ws}, '
                f'k={wk}<->{wkn}) -- these bands are graded CHECK/FAIL, not silently USABLE')
        else:
            self.logger.info(f'{BODY_INDENT}Cliff check: OK '
                             f'(no sub-gate intra-slot energy cliffs)')
        return len(edges)

    def _verify_no_dp_cycles(self) -> int:
        '''
        Final wavefunction-continuity invariant: no band-slot may ship a raw-band
        SWITCH that breaks a CERTAIN dp link (see ``_dp_refuted_seam_edges``) --
        the energy-quiet permutation cycles that pass every energy gate yet are a
        visible discontinuity in the wavefunction character. Recomputed AFTER the
        cycle-rotation and all other passes, so what remains is exactly the seams
        no rotation or exchange could remove (a genuine sub-certain braid ->
        basisrotation territory). Stores the per-slot count (``self.refuted_counts``)
        and the surviving edges (``self.refuted_edges``) so the report downgrades
        those bands to CHECK/FAIL instead of shipping a silent USABLE band, and
        logs a loud warning. Soft like the cliff/bijection checks -- it does not
        raise. Returns the number of surviving refuted edges.
        '''
        edges = self._dp_refuted_seam_edges()
        nslots = self.bands_final.shape[1]
        counts = np.zeros(nslots, dtype=int)
        for _dp, _k, _kn, s in edges:
            counts[int(s)] += 1
        self.refuted_edges = edges
        self.refuted_counts = counts
        if edges:
            slots = sorted({int(s) for _v, _k, _kn, s in edges})
            wv, wk, wkn, ws = edges[0]
            self.logger.warning(
                f'{BODY_INDENT}dp-cycle check: {len(edges)} wavefunction-broken '
                f'seam edge(s) survive in slot(s) {slots} (worst dp {wv:.3g} at '
                f'slot {ws}, k={wk}<->{wkn}) -- these bands are graded CHECK/FAIL/'
                f'ROTATE, not silently USABLE')
        else:
            self.logger.info(f'{BODY_INDENT}dp-cycle check: OK '
                             f'(no wavefunction-broken seam edges)')
        return len(edges)

    def _handoff_dp_residue_to_rotation(self) -> int:
        '''
        Conservative basisrotation handoff for the dp-broken seams
        ``_repair_dp_cycles`` could not remove. The strict refuted census already
        excludes cells whose two SWITCHING bands are near-degenerate; a survivor
        is handed to basisrotation only when an endpoint band is near-degenerate
        (adjacent gap < ``degen_ethr``) with a THIRD band -- a genuine multi-band
        crowding where the per-band label is gauge and no rotation can name a
        unique continuation. Such cells are marked DEGENERATE (the report grades
        the band ROTATE, not a silent USABLE) and appended to ``degenerate_final``
        (the array basisrotation consumes). Clean, well-separated cycles never
        touch a degeneracy, so they keep their honest CHECK grade from the
        dpBreak column and are NOT sent here. Runs before ``_verify_no_dp_cycles``
        so the handed-off cells leave the refuted census (now DEGENERATE-exempt)
        and land in the report's Degen/ROTATE column instead. Returns the number
        of cells handed off.
        '''
        edges = self._dp_refuted_seam_edges()
        if not edges:
            return 0
        ev = self.eigenvalues
        ethr = getattr(self, 'degen_ethr', 0.005)
        nbnd = ev.shape[1]
        handed: list = []
        seen: set = set()
        for _v, k, kn, s in edges:
            for kk in (int(k), int(kn)):
                if (kk, s) in seen:
                    continue
                b = int(self.bands_final[kk, s])
                if b < 0:
                    continue
                adj = -1
                if b > 0 and abs(ev[kk, b] - ev[kk, b - 1]) < ethr:
                    adj = b - 1
                elif b + 1 < nbnd and abs(ev[kk, b + 1] - ev[kk, b]) < ethr:
                    adj = b + 1
                if adj < 0:
                    continue                          # no degeneracy: keep CHECK grade
                self.signal_final[kk, s] = DEGENERATE
                self.correct_signalfinal[kk, s] = DEGENERATE
                handed.append([kk, b, adj])
                seen.add((kk, s))
        if not handed:
            self.logger.info(f'{BODY_INDENT}dp-residue handoff: no near-degenerate gauge '
                             f'seam among the {len(edges)} surviving dp break(s) '
                             f'(all graded CHECK/FAIL)')
            return 0
        handed_arr = np.array(handed, dtype=int)
        existing = getattr(self, 'degenerate_final', None)
        if existing is not None and len(existing) > 0:
            self.degenerate_final = np.vstack([np.asarray(existing), handed_arr])
        else:
            self.degenerate_final = handed_arr
        self.logger.info(f'{BODY_INDENT}dp-residue handoff: {len(handed)} near-degenerate '
                         f'gauge seam cell(s) marked DEGENERATE for basisrotation')
        return len(handed)

    def _slot_energy_penalty(self, k: int, c: int, b: int) -> float:
        '''
        Energy mismatch of putting band ``b`` in slot ``c`` at k-point ``k``: the mean
        |E(k,b) - E(neighbour, band-currently-in-slot-c)| over the k-point's attributed
        neighbours in that slot. Falls back to |E(k,b) - E(k,c)| (slot index used as the
        band-energy proxy) when slot c has no attributed neighbour. Lower is better.
        '''
        Eb = float(self.eigenvalues[k, b])
        diffs = []
        for kn in self.neighbors[k]:
            if kn == -1:
                continue
            bn = self.bands_final[kn, c]
            if bn < 0:
                continue
            diffs.append(abs(Eb - float(self.eigenvalues[kn, bn])))
        if diffs:
            return float(np.mean(diffs))
        return abs(Eb - float(self.eigenvalues[k, c]))

    def _resolve_degenerate_groups(self, ethr: float = None) -> int:
        '''
        Complete energy-isolated near-degenerate band groups by a per-k bijection.

        For each k the bands are partitioned by energy gap: adjacent (energy-sorted)
        bands closer than ``ethr`` form one group. Inside a group of >= 2 bands the
        individual band label is gauge -- any orthonormal basis of the degenerate
        subspace is an eigenstate -- so the group's slots (== its band indices) are
        filled bijectively from the group's own bands. This closes the empty/duplicate
        slots the dot-product graph leaves when a Kramers/SOC pair is split across
        connected components (the band-7 NOT=38 / empty slot-7 symptom) WITHOUT
        touching isolated bands, so genuine crossings of well-separated bands (which
        the dp tracking follows correctly) are preserved.

        A group is only completed when one of its bands is missing from the WHOLE
        row (or duplicated): a group band parked at an out-of-group slot by the dp
        tracking (legitimate crossing) counts as attributed, and slots holding a
        unique band are never overwritten. Fills prefer identity (slot == band,
        i.e. energy order) and are marked FORCED_CONTINUITY in
        ``correct_signalfinal`` so the band no longer counts as not-solved while
        the fills stay visible (FOR column) for audit.

        ``ethr`` defaults to ``self.degen_ethr`` (0.005 Ry ~ 68 meV; eigenvalues.npy
        is stored in Ry -- the same scale degenrotation groups on, and the centre of
        the ~3-8 mRy stability plateau). Returns the number of slots filled.
        '''
        if ethr is None:
            ethr = getattr(self, 'degen_ethr', 0.005)
        bf  = self.bands_final
        csf = self.correct_signalfinal
        nb  = bf.shape[1]
        filled = 0
        grown  = 0
        resolved = {}                                                   # slot -> [k, ...] for the verbose dump

        for k in range(self.nks):
            e = self.eigenvalues[k]
            lo = 0
            for b in range(1, nb + 1):
                if b < nb and (e[b] - e[b - 1]) < ethr:
                    continue                                            # still inside the current group
                hi = b - 1                                              # close group of bands [lo, hi]
                if hi > lo:                                             # degenerate group (singletons skipped)
                    # Whole-ROW census, not group-slot census: when bands cross, the
                    # dp tracking legitimately parks a group band at an out-of-group
                    # slot (e.g. the mirror-axis rows where band 5 sits at slot 7).
                    # A band held exactly once ANYWHERE in the row is attributed --
                    # judging only slots lo..hi declared such rows broken, force-
                    # filled duplicates over correct cells and left FOR/"suspect"
                    # grades on them (cost phosphorene band 5 its CLEAN grade).
                    row_counts = {}
                    for v in bf[k]:
                        if v >= 0:
                            row_counts[int(v)] = row_counts.get(int(v), 0) + 1
                    miss = [bd for bd in range(lo, hi + 1)
                            if row_counts.get(bd, 0) == 0]
                    if miss:
                        if hi - lo + 1 > 2:
                            grown += 1
                        # Fill only group slots that are empty or hold a band
                        # duplicated elsewhere in the row; a slot holding a unique
                        # band (in- or out-of-group) keeps its dp-chosen value --
                        # overwriting it would orphan that band.
                        free = [s for s in range(lo, hi + 1)
                                if int(bf[k, s]) < 0
                                or row_counts.get(int(bf[k, s]), 0) > 1]
                        miss_set = set(miss)
                        def _fill(s, bd):
                            old = int(bf[k, s])
                            if old >= 0:
                                row_counts[old] -= 1
                            bf[k, s] = bd; csf[k, s] = FORCED_CONTINUITY
                            row_counts[bd] = row_counts.get(bd, 0) + 1
                        for s in [s for s in free if s in miss_set]:    # identity first (slot == band)
                            _fill(s, s)
                            miss_set.discard(s); free.remove(s); filled += 1
                            resolved.setdefault(s, []).append(k)
                        for s, bd in zip(free, sorted(miss_set)):       # pair any leftovers in energy order
                            _fill(s, bd); filled += 1
                            resolved.setdefault(bd, []).append(k)
                lo = b                                                  # next group starts here

        if filled:
            self.logger.info(f'{BODY_INDENT}Degenerate-group completion: filled {filled} slot(s) '
                             f'in {grown} grown block(s) by within-group bijection '
                             f'(gap < {ethr * 1000:.1f} mRy)')
            if self.logger.level <= logging.DEBUG:
                for bd in sorted(resolved):
                    ks = resolved[bd]
                    self.logger.debug(f'{BODY_INDENT}  slot {bd}: completed at {len(ks)} k-point(s) '
                                      f'by degenerate-group bijection: {ks}')
        return filled

    def _enforce_bijection(self) -> int:
        '''
        Make ``self.bands_final`` a per-k permutation, repairing both bijection defects:

          (a) DUPLICATE: a band attributed to more than one slot. The copy in the best
              slot is kept (highest energy-continuity signal in ``correct_signalfinal``,
              ties broken by smallest energy residual to that slot's neighbours) and each
              LOSING slot is reassigned to a band missing from the row, by best energy fit
              (ungated -- a duplicate must always move).

          (b) GAP: an empty slot (-1) together with a band absent from the row. The
              duplicate logic never reaches it (empty slots are not in ``where``), so it
              used to be left for the force pass. Here every empty slot is paired ONE-TO-
              ONE to a missing band by best energy fit, GATED by ``accept_E`` so a band is
              never forced across a gross energy jump (an inadmissible slot stays -1 for
              the force/repair pass). This is the near-degenerate partner the community
              partition dropped -- see the get_cluster_score cross-band veto.

        Returns the total number of slots reassigned or filled.
        '''
        bf = self.bands_final
        sig = self.correct_signalfinal
        nk, nb = bf.shape
        accept_E = getattr(self, 'accept_E', np.inf)                                # gross-jump gate for the gap branch
        reassigned = 0                                                              # (a) duplicate-loser reassignments
        filled = 0                                                                  # (b) empty-slot gap fills
        dbg = self.logger.level <= logging.DEBUG                                    # verbose (-v): per-action trace
        for k in range(nk):
            row = bf[k]                                                             # view: reflects edits made below
            where = {}                                                              # band value -> slots holding it (attributed only)
            for c in range(nb):
                v = int(row[c])
                if v >= 0:
                    where.setdefault(v, []).append(c)
            dup_vals = [v for v, cs in where.items() if len(cs) > 1]
            missing = [b for b in range(self.total_bands) if b not in where]        # bands absent from the whole row

            # (a) Duplicate branch (unchanged): move each duplicate's losing slot(s)
            #     onto a missing band by best energy fit, ungated.
            if dup_vals and missing:
                loser_slots = []
                # Signal rank for choosing which duplicate copy to KEEP. On the
                # raw csf scale FORCED_CONTINUITY (5) would outrank CORRECT (4),
                # letting a force-fill evict a validated cell -- remap it below
                # everything validated (above MISTAKE only): a force-fill is not
                # a genuine solve. In the LAST pass the completeness-pass cells
                # still carry the raw dot-product-scale FORCED (6) here (they are
                # excluded from re-evaluation and the FORCED_CONTINUITY stamp
                # happens only after correct_signal returns), so remap that too.
                _rank = lambda c: 1.5 if sig[k, c] in (FORCED_CONTINUITY, FORCED) else sig[k, c]
                for v in dup_vals:
                    cs = where[v]
                    keep = max(cs, key=lambda c: (_rank(c), -self._slot_energy_penalty(k, c, v)))
                    loser_slots.extend(c for c in cs if c != keep)
                pairs = sorted(((self._slot_energy_penalty(k, c, b), c, b)
                                for c in loser_slots for b in missing),
                               key=lambda t: t[0])
                used_c, used_b = set(), set()
                for _pen, c, b in pairs:
                    if c in used_c or b in used_b:
                        continue
                    if dbg and sig[k, c] >= POTENTIAL_CORRECT:    # only audit dups that evict a (near-)solved slot; routine churn is in the INFO summary
                        self.logger.debug(f'{BODY_INDENT}  [bijection] dup  k={k} slot={c}: '
                                          f'band {int(bf[k, c])}(duplicate)->{b}  penalty={_pen:.4f}')
                    bf[k, c] = b
                    used_c.add(c); used_b.add(b)
                    reassigned += 1
                missing = [b for b in missing if b not in used_b]                   # bands consumed above are no longer missing

            # (b) Gap branch (new): pair empty slots to remaining missing bands, one-to-
            #     one by smallest energy penalty first, gated by accept_E.
            empty_slots = [c for c in range(nb) if int(row[c]) == -1]
            if empty_slots and missing:
                pairs = sorted(((self._slot_energy_penalty(k, c, b), c, b)
                                for c in empty_slots for b in missing),
                               key=lambda t: t[0])
                used_c, used_b = set(), set()
                for pen, c, b in pairs:
                    if c in used_c or b in used_b:
                        continue
                    if pen > accept_E:                                              # nothing continuous fits: leave -1
                        continue
                    if dbg:
                        self.logger.debug(f'{BODY_INDENT}  [bijection] gap  k={k} slot={c}: '
                                          f'-1->band {b}  penalty={pen:.4f}')
                    bf[k, c] = b
                    used_c.add(c); used_b.add(b)
                    filled += 1
        if reassigned or filled:
            self.logger.info(f'{BODY_INDENT}Bijection enforcement: reassigned {reassigned} '
                             f'duplicate slot(s) and filled {filled} empty slot(s) with missing bands')
        return reassigned + filled

    def _dp_support_counts(self, tol: float = None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Per-cell dot-product support of the current attribution: for every (k, slot)
        count the directions whose edge connects the cell's assigned band to the band
        assigned to the SAME slot at the neighbouring k-point with dp >= ``tol``
        (default ``DP_SUP_TOL``).

        Returns ``(sup, tot)``, both (nks, nbands) int arrays: ``sup`` = number of
        dp-supported directions, ``tot`` = number of comparable directions (the
        neighbour exists and both cells are attributed). ``sup == 0`` with
        ``tot > 0`` means the slot's content is orthogonal to the same slot at every
        neighbour: the wavefunction evidence contradicts the attribution no matter
        how quiet the energy is. This is the signature of a label swap between
        near-parallel bands (MoS2 12/13: a ~5 mRy energy error passes every energy
        gate, but the cross dot-product is exactly 0).
        '''
        if tol is None:
            tol = DP_SUP_TOL
        bf = self.bands_final
        nk, nb = bf.shape
        sup = np.zeros((nk, nb), int)
        tot = np.zeros((nk, nb), int)
        neigh = np.asarray(self.neighbors)
        idx = np.arange(nk)
        for d in range(neigh.shape[1]):
            kn = neigh[:, d]
            ok = kn >= 0
            kn_safe = np.where(ok, kn, 0)
            b1 = bf                                                     # band in this cell
            b2 = bf[kn_safe]                                            # band in the same slot at the d-neighbour
            comparable = ok[:, None] & (b1 >= 0) & (b2 >= 0)
            vals = self.connections[idx[:, None], d,
                                    np.where(comparable, b1, 0),
                                    np.where(comparable, b2, 0)]
            tot += comparable
            sup += comparable & (vals >= tol)
        return sup, tot

    def _certain_sheet_forest(self) -> np.ndarray:
        '''
        The maximal collision-free certain forest (the "sheets"). POST-LOOP
        MEASUREMENT ONLY: the in-loop locking that once consumed this (P0
        SHEET_LOCK: glue in every graph build, whole-sheet claims, Louvain
        reunification, final audit) froze the sweep and was removed Jul 2026
        (see docs/cluster0_dp_seam_correction_proposal.md). The measurement
        itself is sound and feeds the post-loop seam passes (refusal cuts =
        free seam locations) and a future sheet-guided relabel.

        Every certain link (dp >= DP_CERTAIN, unique partner by Bessel) is a
        must-keep constraint -- but only locally: character transport around a
        conical intersection returns on the other sheet, so the full closure is
        not single-slottable (measured: 44k-state component, 41k same-k
        collisions). The maximal consistent subset is built here: union-find
        over states (node id = k + b*nks), processing links strongest-first and
        REFUSING any merge that would put two states at the same k-point in one
        component. The refusals land exactly on the frustrated loops -- the
        physical branch cuts of the crossings.

        Energy safety is measured, not assumed: on phosphorene run 6 the energy
        step along every one of the 139,184 certain links is <= 0.033 Ry --
        under half of accept_E -- so locking sheets can never create a cliff.

        Returns ``sheet_id`` (nks, nbnd) int32, -1 for singleton states, and
        caches ``self.sheet_id`` and ``self._sheet_members`` {sid: [state ids]}.
        The connections array is static input, so this is computed once.
        '''
        cached = getattr(self, 'sheet_id', None)
        if cached is not None:
            return cached
        nks, nb = self.nks, self.nbnd
        conn = self.connections                                     # (nks, ndirs, nbnd, nbnd)
        part = np.argmax(conn, axis=3)
        best = np.take_along_axis(conn, part[..., None], axis=3)[..., 0]
        neigh = np.asarray(self.neighbors)

        links = []                                                  # (dp, state_a, state_b)
        for d in range(self.number_neighbors):
            kn = neigh[:, d]
            for b in range(nb):
                sel = (kn >= 0) & (best[:, d, b] >= DP_CERTAIN)
                for k in np.where(sel)[0]:
                    links.append((float(best[k, d, b]),
                                  int(k) + b * nks,
                                  int(kn[k]) + int(part[k, d, b]) * nks))
        links.sort(key=lambda t: -t[0])                             # strongest first

        parent = np.arange(nks * nb)

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:                                # path compression
                parent[x], x = root, parent[x]
            return root

        ksets: dict = {}                                            # root -> set of k it covers

        def kset(r: int) -> set:
            s = ksets.get(r)
            if s is None:
                s = {r % nks}
                ksets[r] = s
            return s

        refused_links: list = []
        for _, a, c in links:
            ra, rc = find(a), find(c)
            if ra == rc:
                continue                                            # consistent cycle inside a sheet
            sa, sc = kset(ra), kset(rc)
            if len(sa) < len(sc):
                ra, rc, sa, sc = rc, ra, sc, sa
            if sa & sc:
                refused_links.append((a, c))                        # frustrated loop: branch cut stays open
                continue
            parent[rc] = ra
            sa |= sc
            ksets.pop(rc, None)
        refused = len(refused_links)
        # The refusal cuts are the physical branch cuts of the crossing
        # holonomy: laying a label seam ON one is free (the certain link is
        # already unkeepable). Consumed by the min-cut seam placement.
        self._sheet_refusals = set(frozenset(l) for l in refused_links)

        # components -> sheet ids (size >= 2)
        roots: dict = {}
        for st in range(nks * nb):
            roots.setdefault(find(st), []).append(st)
        sheet_id = np.full((nks, nb), -1, np.int32)
        self._sheet_members = {}
        sid = 0
        for mem in sorted(roots.values(), key=len, reverse=True):
            if len(mem) < 2:
                break
            for st in mem:
                sheet_id[st % nks, st // nks] = sid
            self._sheet_members[sid] = mem
            sid += 1
        self.sheet_id = sheet_id

        sizes = sorted((len(m) for m in self._sheet_members.values()), reverse=True)
        covered = int(sum(sizes))
        self.logger.info(f'{BODY_INDENT}Certain-sheet forest: {len(links)} certain link(s), '
                         f'{refused} refused (same-k collision = branch cuts), '
                         f'{sid} sheet(s) covering {covered}/{nks*nb} states '
                         f'({100.0*covered/(nks*nb):.1f}%); largest: {sizes[:10]}')
        return sheet_id

    def _certain_partner_map(self, threshold: float = DP_CERTAIN) -> np.ndarray:
        '''
        ``cert[k, d, b]`` = the unique raw band at neighbour ``d`` of ``k`` whose
        dot product with raw band ``b`` at ``k`` is >= ``threshold`` (default
        ``DP_CERTAIN``), or -1 when no such partner exists (or the neighbour is
        outside the grid).

        Well-defined by normalization: sum_b' dp^2 <= 1 (Bessel), so at most one
        partner can clear any threshold above 1/sqrt(2) ~ 0.707, and a certain
        partner caps every rival at sqrt(1 - threshold^2). Cached per threshold
        -- the connections array is static input.
        '''
        cached = getattr(self, '_cert_maps', None)
        if cached is None:
            cached = self._cert_maps = {}
        if threshold in cached:
            return cached[threshold]
        conn = self.connections                                     # (nks, ndirs, nbnd, nbnd)
        part = np.argmax(conn, axis=3)
        best = np.take_along_axis(conn, part[..., None], axis=3)[..., 0]
        cert = np.where(best >= threshold, part, -1).astype(np.int32)
        cert[np.asarray(self.neighbors) == -1] = -1                 # no neighbour that way
        cached[threshold] = cert
        return cert

    def _near_degenerate_mask(self) -> np.ndarray:
        '''
        ``close[k, b]`` = the (k, band) cell sits inside a collapsed gap
        (adjacent-band spacing < degen_ethr). There a vanishing overlap is
        gauge, not evidence, so certain-link accounting and seam costs exempt
        these cells. Cached -- eigenvalues are static input.
        '''
        cached = getattr(self, '_near_degen_mask', None)
        if cached is not None:
            return cached
        ethr = getattr(self, 'degen_ethr', 0.005)
        dE_close = np.diff(self.eigenvalues, axis=1) < ethr
        close = np.zeros(self.eigenvalues.shape, bool)
        close[:, 1:] |= dE_close
        close[:, :-1] |= dE_close
        self._near_degen_mask = close
        return close

    def _dir_index(self) -> dict:
        '''
        ``{(k1, k2): d}`` for every ordered neighbour pair -- the direction
        index needed to look a step up in the connections array. Cached.
        '''
        cached = getattr(self, '_dir_idx_map', None)
        if cached is not None:
            return cached
        neigh = np.asarray(self.neighbors)
        dir_idx: dict = {}
        for d in range(neigh.shape[1]):
            for k1, k2 in enumerate(neigh[:, d]):
                if k2 >= 0:
                    dir_idx[(int(k1), int(k2))] = d
        self._dir_idx_map = dir_idx
        return dir_idx

    def _repair_certain_links(self, max_rounds: int = 16) -> int:
        '''
        Post-loop wavefunction-evidence repair: at every energy-broken k-point
        row, each slot takes the band its trusted neighbours' CERTAIN successors
        mandate.

        A certain link (see ``_certain_partner_map``) identifies a state's
        continuation with near-certainty, so at a k-point where slot ``s`` is
        trusted at a neighbour ``kn`` (holding band ``b_n``), the band that slot
        ``s`` must hold at ``k`` is ``cert[kn, d_rev, b_n]`` -- unique when it
        exists. The pairwise dp-swap repair cannot use this evidence on two
        whole classes of defects:

          * a self-consistent wrong PATCH (a smooth sheet carried by the wrong
            slot over a stretch of k-points): its interior cells keep dp support
            from their equally-wrong neighbours, so the ``sup == 0`` trigger
            never fires (phosphorene bands 12/13, rows 37/38 seam);
          * permutation CYCLES of order > 2, and cells invalidated to -1 (a
            pairwise swap needs two occupied cells).

        Scope is strictly LOCAL and trigger-based -- no global structure is
        assumed. (Both global formulations fail on real data: character
        transport is path-dependent around conical intersections, and the
        labelling legitimately changes sheet-owners at crossing corridors where
        the dp dips below certainty, so neither "certain component" nor
        "consistent region" is single-slotted in a correct output.)

          1. TRIGGER k-points: any slot that is unattributed (-1), graded
             NOT/MIS, or sitting on a gross energy jump (> accept_E) to a
             same-slot neighbour. Deliberately NOT triggered: energy-quiet
             cells whose links are dp-forbidden -- those mark the corridors
             where sheet ownership legitimately switches (and the braid), and
             relabelling them trades energy continuity away (tested: FAIL
             11 -> 72 on phosphorene when they are included).
          2. At each triggered k-point, EVERY slot collects the certain
             successors of its trusted neighbours. A slot with a UNANIMOUS
             mandate b* != current takes b*; disagreeing or absent mandates
             propose nothing (every tested arbitration between disagreeing
             mandates -- component majority, region consensus, region-size
             dominance -- damaged the crossing corridors and the braid on real
             data, where sheet ownership legitimately switches and no labelling
             satisfies every certain link). Near-degenerate states (adjacent
             gap < ``degen_ethr``) neither mandate nor move -- inside a
             collapsed gap the label is gauge (basisrotation territory).
          3. The mandated moves are applied per row at once: bands displaced
             from mandated slots are matched to the vacated/empty slots by dp
             affinity with an energy tie-break, and a row is committed only if
             it remains a duplicate-free permutation (reverted otherwise). A
             row-scale permutation cycle (the 12->13->15->14 seam) resolves in
             one shot because every member slot receives its mandate from the
             SAME trusted neighbouring row.

        Rounds repeat (a repaired row becomes the next row's trusted anchor, so
        a patch peels inward) until a fixed point. Runs on the final canvas
        BEFORE the pairwise dp-swap pass (which keeps handling sub-certain
        evidence). Both signals of every changed cell are re-derived and the
        energy-continuity signal of their same-slot neighbours is re-graded, so
        the report grades what is actually shipped. Returns the number of
        changed cells.
        '''
        bf = self.bands_final
        nks, nslots = bf.shape
        nbnd = self.connections.shape[2]
        neigh = np.asarray(self.neighbors)
        n_dir = neigh.shape[1]
        cert = self._certain_partner_map()

        # reverse-direction lookup: neigh[k, d] == kn  <=>  neigh[kn, rev[k, d]] == k
        rev = np.full((nks, n_dir), -1, dtype=np.int8)
        for d in range(n_dir):
            kn_col = neigh[:, d]
            ok = np.where(kn_col >= 0)[0]
            for dr in range(n_dir):
                back = neigh[kn_col[ok], dr] == ok
                rev[ok[back], d] = dr

        # near-degenerate (k, band) cells: exempt (gauge territory)
        close = self._near_degenerate_mask()

        changed_cells: list = []
        frozen: set = set()                                          # cells changed once stay put
        n_revert = n_ambig = 0
        n_trigger0 = 0

        for round_ in range(max_rounds):
            ###########################################################################
            # 1. Trigger cells and trusted anchors on the current canvas.
            ###########################################################################
            bad = (bf < 0) | np.isin(self.correct_signalfinal, (NOT_SOLVED, MISTAKE))
            for d in range(n_dir):                                   # gross same-slot energy jumps
                kn_col = neigh[:, d]
                ok = kn_col >= 0
                kn_s = np.where(ok, kn_col, 0)
                b1, b2 = bf, bf[kn_s]
                comp = ok[:, None] & (b1 >= 0) & (b2 >= 0)
                E1 = self.eigenvalues[np.arange(nks)[:, None], np.where(b1 >= 0, b1, 0)]
                E2 = self.eigenvalues[kn_s[:, None], np.where(b2 >= 0, b2, 0)]
                bad |= comp & (np.abs(E1 - E2) > self.accept_E)
            bad &= self.signal_final != DEGENERATE
            trusted = (bf >= 0) & ~bad & (self.signal_final != DEGENERATE)
            if round_ == 0:
                n_trigger0 = int(bad.sum())
            if not bad.any():
                break

            ###########################################################################
            # 2. Certain mandates at every triggered k-point. UNANIMOUS only: a
            #    wrong patch always counter-mandates its own continuation at the
            #    boundary, so any arbitration between disagreeing mandates has to
            #    guess which side is wrong -- and every tested arbiter (component
            #    majority, region consensus, region-size dominance at 3x) damaged
            #    the crossing corridors and the braid on real data, where sheet
            #    ownership legitimately switches and no labelling satisfies every
            #    certain link. Disagreement therefore proposes nothing; the cost
            #    is that patches whose trusted rim is thinner than the wrong core
            #    stay (honestly) broken.
            ###########################################################################
            proposals: dict = {}                                     # k -> [(slot, band)]
            for k in np.where(bad.any(axis=1))[0]:
                k = int(k)
                row_props = {}
                mandated_bands = {}
                for s in range(nslots):
                    exps = set()
                    for d in range(n_dir):
                        kn = int(neigh[k, d])
                        if kn < 0 or not trusted[kn, s]:
                            continue
                        dr = int(rev[k, d])
                        if dr < 0:
                            continue
                        b_exp = int(cert[kn, dr, int(bf[kn, s])])
                        if b_exp >= 0:
                            exps.add(b_exp)
                    if len(exps) != 1:
                        if len(exps) > 1:
                            n_ambig += 1
                        continue                                     # no unanimous mandate
                    b_exp = exps.pop()
                    if close[k, b_exp] or (bf[k, s] >= 0 and close[k, int(bf[k, s])]):
                        continue                                     # gauge territory: hands off
                    if b_exp == int(bf[k, s]):
                        continue
                    if (k, s) in frozen:
                        continue                                     # no flip-flopping across rounds
                    if b_exp in mandated_bands:                      # two slots mandated one band
                        row_props.pop(mandated_bands[b_exp], None)
                        n_ambig += 1
                        continue
                    mandated_bands[b_exp] = s
                    row_props[s] = b_exp
                if row_props:
                    proposals[k] = list(row_props.items())

            if not proposals:
                break

            ###########################################################################
            # 3. Apply per k-point row: mandates first, then close the induced
            #    permutation (displaced bands -> vacated/empty slots).
            ###########################################################################
            round_changed = 0
            for k, props in proposals.items():
                old_row = bf[k].copy()
                new_row = old_row.copy()
                moved = set()
                targets = set()
                for s, b in props:
                    new_row[s] = b
                    moved.add(int(b))
                    targets.add(int(s))
                for s in range(nslots):                              # vacate the moved bands' old slots
                    if s not in targets and int(old_row[s]) in moved:
                        new_row[s] = -1
                empty = [s for s in range(nslots) if int(new_row[s]) == -1]
                in_row = {int(v) for v in new_row if v >= 0}
                free = [b for b in range(nbnd) if b not in in_row]
                if len(free) == len(empty) and free:
                    cost = np.zeros((len(free), len(empty)))
                    for i_, b in enumerate(free):
                        for j_, s in enumerate(empty):
                            aff = self._dp_affinity(k, int(s), int(b))
                            dp_term = (1.0 - aff) if aff is not None else 0.5
                            pen = self._slot_energy_penalty(k, int(s), int(b))
                            cost[i_, j_] = dp_term + 0.25 * min(pen / max(self.accept_E, 1e-12), 4.0)
                    rows_, cols_ = linear_sum_assignment(cost)
                    for r_, c_ in zip(rows_, cols_):
                        new_row[empty[c_]] = free[r_]
                vals = new_row[new_row >= 0]
                if len(np.unique(vals)) != len(vals) or (new_row == -1).sum() > (old_row == -1).sum():
                    n_revert += 1                                    # bijection safety first
                    continue
                diff = np.where(new_row != old_row)[0]
                # A changed cell must keep at least one dp-supported direction
                # (the mandated ones do by construction; this guards the
                # Hungarian row-closing from parking a band with no evidence).
                ok_row = True
                for s in diff:
                    b1 = int(new_row[s])
                    if b1 < 0:
                        continue
                    sup_n = tot_n = 0
                    for d_ in range(n_dir):
                        kn_ = int(neigh[k, d_])
                        if kn_ < 0:
                            continue
                        b2_ = int(bf[kn_, s]) if kn_ != k else -1
                        if b2_ < 0:
                            continue
                        tot_n += 1
                        if self.connections[k, d_, b1, b2_] >= DP_SUP_TOL:
                            sup_n += 1
                    if tot_n > 0 and sup_n == 0:
                        ok_row = False
                        break
                if not ok_row:
                    n_revert += 1
                    continue
                bf[k] = new_row
                changed_cells.extend((int(k), int(s)) for s in diff)
                frozen.update((int(k), int(s)) for s in diff)
                round_changed += len(diff)
            self.logger.debug(f'{BODY_INDENT}  [cert-repair] round {round_ + 1}: '
                              f'{int(bad.sum())} trigger cell(s), {len(proposals)} row(s) '
                              f'with mandates, {round_changed} cell(s) changed')
            if round_changed == 0:
                break

        if n_trigger0:
            self.logger.info(f'{BODY_INDENT}Certain-link repair: {n_trigger0} energy-broken/'
                             f'unattributed cell(s) triggered; certain-successor mandates changed '
                             f'{len(changed_cells)} cell(s) at '
                             f'{len(set(k for k, _ in changed_cells))} k-point(s) '
                             f'over {round_ + 1} round(s)'
                             + (f', {n_ambig} ambiguous mandate(s) skipped' if n_ambig else '')
                             + (f', {n_revert} row(s) reverted' if n_revert else ''))
            for k, s in changed_cells:
                self.logger.debug(f'{BODY_INDENT}  [cert-repair] k={k} slot={s}: '
                                  f'band ->{int(bf[k, s])}  '
                                  f'E={float(self.eigenvalues[k, bf[k, s]]) if bf[k, s] >= 0 else np.nan:.4f}')
        else:
            self.logger.info(f'{BODY_INDENT}Certain-link repair: no energy-broken or '
                             f'unattributed cells (nothing to do)')
        if not changed_cells:
            return 0

        ###########################################################################
        # Re-derive both signals of the changed cells and re-grade the energy-
        # continuity signal of their same-slot neighbours (their fits changed).
        # DEGENERATE keeps its marker; untouched FORCED cells keep their audit.
        ###########################################################################
        changed_set = set(changed_cells)
        regrade = set(changed_cells)
        for k, s in changed_cells:
            for kn in neigh[k]:
                if kn >= 0:
                    regrade.add((int(kn), s))
        for k, s in sorted(regrade):
            if self.signal_final[k, s] == DEGENERATE:
                continue
            if (k, s) in changed_set:
                b1 = int(bf[k, s])
                if b1 >= 0:
                    values = []
                    for i_neig, kn in enumerate(self.neighbors[k]):
                        if kn == -1:
                            continue
                        b2 = int(bf[kn, s])
                        if b2 < 0:
                            continue
                        values.append(self.connections[k, i_neig, b1, b2])
                    self.signal_final[k, s] = min(evaluate_result(values) if values else NOT_SOLVED,
                                                  POTENTIAL_CORRECT)
                if hasattr(self, 'forced_mask'):
                    self.forced_mask[k, s] = False                   # a mandate is not a force-fill
            elif self.correct_signalfinal[k, s] == FORCED_CONTINUITY:
                continue
            if int(bf[k, s]) < 0:
                self.correct_signalfinal[k, s] = NOT_SOLVED
                continue
            sig, _ = evaluate_point(self.dimensions, k, s, self.kpoints_index,
                                    self.matrix, bf, self.eigenvalues,
                                    accept_E=self.accept_E, disp_scale=self.disp_scale,
                                    local_gap=self.local_gap, band_disp=self.band_disp)
            self.correct_signalfinal[k, s] = sig
        return len(changed_cells)

    def _slot_wall_edges(self) -> list:
        '''
        Every undirected same-slot adjacency whose energy step is a gross jump
        (> accept_E), as ``(jump, k, kn, slot)`` sorted worst first -- the edge set
        behind both the Silent report column and the visible cliffs. Degenerate-
        signalled endpoints are excluded (their label is gauge).
        '''
        bf = self.bands_final
        ev = self.eigenvalues
        neigh = np.asarray(self.neighbors)
        nks, nslots = bf.shape
        karr = np.arange(nks)
        walls = []
        for d in range(neigh.shape[1]):
            kn_col = neigh[:, d]
            ok = kn_col >= 0
            kns = np.where(ok, kn_col, 0)
            comp = (ok & (karr < kns))[:, None] & (bf >= 0) & (bf[kns] >= 0) \
                   & (self.signal_final != DEGENERATE) \
                   & (self.signal_final[kns] != DEGENERATE)
            E1 = ev[karr[:, None], np.where(bf >= 0, bf, 0)]
            E2 = ev[kns[:, None], np.where(bf[kns] >= 0, bf[kns], 0)]
            jump = np.abs(E1 - E2)
            for k, s in zip(*np.where(comp & (jump > self.accept_E))):
                walls.append((float(jump[k, s]), int(k), int(kns[k]), int(s)))
        walls.sort(reverse=True)
        return walls

    def _cliff_seam_edges(self) -> list:
        '''
        The sub-accept_E energy cliffs ``_slot_wall_edges`` cannot see. Every
        undirected same-slot adjacency where the slot SWITCHES raw band
        (``bf[k] != bf[kn]``) across an energy step larger than the LOCAL
        continuity gate ``continuity_gate`` (``min`` of the two endpoints' gates)
        -- i.e. a switch that lands where the two bands are far apart in energy,
        not at a genuine near-degenerate crossing.

        This is the intra-slot continuity guard the flat ``accept_E`` gate is too
        loose to enforce (accept_E ~ 3*disp_scale ~ 1 eV; a real 1 eV cliff sits
        just under it and reads as "continuous"). ``continuity_gate`` sharpens to
        half the local adjacent-band gap where that gap resolves the band's own
        dispersion, and only there -- near-degenerate crowding keeps the loose
        accept_E, so genuine 50/50 crossings (small energy step, bands meeting)
        are never flagged. Near-degenerate / DEGENERATE-signalled endpoints are
        excluded outright (their per-band label is gauge -- basisrotation
        territory), exactly as ``_slot_wall_edges`` excludes DEGENERATE.

        Restricting to raw-band SWITCH edges is load-bearing: a legitimate steep
        one-band step (same band both ends, no switch) can exceed the local gate
        yet is genuine dispersion, so it must never be flagged. Returned as
        ``(jump, k, kn, slot)`` sorted worst (largest jump) first -- the same
        shape ``_slot_wall_edges`` uses, so the wall-relocation core consumes it
        unchanged.
        '''
        bf = self.bands_final
        ev = self.eigenvalues
        neigh = np.asarray(self.neighbors)
        cg = self.continuity_gate
        close = self._near_degenerate_mask()
        nks, nslots = bf.shape
        karr = np.arange(nks)
        cliffs = []
        for d in range(neigh.shape[1]):
            kn_col = neigh[:, d]
            ok = kn_col >= 0
            kns = np.where(ok, kn_col, 0)
            b1 = np.where(bf >= 0, bf, 0)
            b2v = bf[kns]
            b2 = np.where(b2v >= 0, b2v, 0)
            comp = (ok & (karr < kns))[:, None] & (bf >= 0) & (b2v >= 0) \
                   & (bf != b2v) \
                   & (self.signal_final != DEGENERATE) \
                   & (self.signal_final[kns] != DEGENERATE) \
                   & ~close[karr[:, None], b1] & ~close[kns[:, None], b2]
            E1 = ev[karr[:, None], b1]
            E2 = ev[kns[:, None], b2]
            jump = np.abs(E1 - E2)
            gate = np.minimum(cg[karr[:, None], b1], cg[kns[:, None], b2])
            for k, s in zip(*np.where(comp & (jump > gate))):
                cliffs.append((float(jump[k, s]), int(k), int(kns[k]), int(s)))
        cliffs.sort(reverse=True)
        return cliffs

    def _dp_refuted_seam_edges(self) -> list:
        '''
        Every undirected same-slot adjacency where the slot switches raw band
        AGAINST the wavefunction evidence: the state held at one endpoint has a
        CERTAIN successor (dp >= DP_CERTAIN, unique by normalization -- see
        ``_certain_partner_map``) at the other endpoint, and it is not the band
        the slot holds there. By the Bessel bound the assigned continuation then
        carries dp <= sqrt(1 - DP_CERTAIN^2) ~ 0.44: the seam breaks
        wavefunction continuity while its energy step may sit anywhere below
        the gross-jump gate, invisible to ``_slot_wall_edges``. Returned as
        ``(dp_assigned, k, kn, slot)`` sorted worst (lowest dp) first.
        Degenerate-signalled endpoints and near-degenerate cells are excluded
        (their label is gauge -- basisrotation territory).
        '''
        bf = self.bands_final
        neigh = np.asarray(self.neighbors)
        nks, nslots = bf.shape
        karr = np.arange(nks)
        cert = self._certain_partner_map()
        close = self._near_degenerate_mask()

        best: dict = {}
        for d in range(neigh.shape[1]):
            kn_col = neigh[:, d]
            ok = kn_col >= 0
            kns = np.where(ok, kn_col, 0)
            b1 = np.where(bf >= 0, bf, 0)
            b2v = bf[kns]
            b2 = np.where(b2v >= 0, b2v, 0)
            comp = ok[:, None] & (bf >= 0) & (b2v >= 0) & (bf != b2v) \
                   & (self.signal_final != DEGENERATE) \
                   & (self.signal_final[kns] != DEGENERATE) \
                   & ~close[karr[:, None], b1] & ~close[kns[:, None], b2]
            succ = cert[:, d, :][karr[:, None], b1]     # certain successor of the held state
            comp &= (succ >= 0) & (succ != b2v)
            dpv = self.connections[karr[:, None], d, b1, b2]
            for k, s in zip(*np.where(comp)):
                kn = int(kns[k])
                key = (min(int(k), kn), max(int(k), kn), int(s))
                v = float(dpv[k, s])
                if key not in best or v < best[key]:
                    best[key] = v
        out = [(v, a, c, s) for (a, c, s), v in best.items()]
        out.sort()
        return out

    def dp_interp_mask(self) -> np.ndarray:
        '''
        ``(nks, nslots)`` boolean mask of the k-points the Berry-geometry pass
        should interpolate over: the endpoints of every dp-broken seam edge
        (``_dp_refuted_seam_edges`` -- the report's dpBrk column) on a band that
        is NOT graded FAIL. These are exactly the wavefunction-discontinuity
        points the clustering signals; at each the assigned band switches to a
        near-orthogonal state, so the central finite-difference gradient feeding
        the connection/curvature spikes ~1/h. Both endpoints are flagged (each
        one's stencil reaches across the broken edge).

        FAIL-graded bands are excluded outright -- their attribution is not
        trustworthy anywhere, so patching a few seam cells is meaningless. The
        grade is read from ``self.band_grade`` (populated by ``report()``); if
        that is unavailable, a slot is treated as FAIL when it carries any
        genuine energy break (NOT_SOLVED / MISTAKE) in ``correct_signalfinal``.
        Cliff edges and the loose |dp|<tau scan are deliberately NOT included:
        only the signalled dp-broken seams, no more.
        '''
        nks, nslots = self.bands_final.shape
        grade = getattr(self, 'band_grade', None)
        if grade is not None:
            fail = {s for s in range(nslots) if s < len(grade) and grade[s] == 'FAIL'}
        else:
            csf = self.correct_signalfinal
            fail = {s for s in range(nslots)
                    if bool(np.any((csf[:, s] == NOT_SOLVED) | (csf[:, s] == MISTAKE)))}
        mask = np.zeros((nks, nslots), bool)
        for _dp, k, kn, s in self._dp_refuted_seam_edges():
            if s in fail:
                continue
            mask[k, s] = True
            mask[kn, s] = True
        return mask

    def _degenerate_seam_edges(self) -> list:
        '''
        The near-degenerate counterpart of ``_dp_refuted_seam_edges``: every
        undirected same-slot adjacency where the slot switches raw band against
        a FLOOD-CERTAIN successor (dp >= ``DP_FLOOD_SUPPORT``, unique by Bessel)
        AND at least one endpoint is near-degenerate or DEGENERATE-signalled.

        These are exactly the seams tier 2 (``_dp_refuted_seam_edges``) SKIPS:
        there the per-band label is gauge, so the strict pass hands them to
        basisrotation. But a small patch that attached to the WRONG member of a
        near-degenerate pair still has clean wavefunction evidence naming the
        right band (dp ~ 1 to the correct continuation, exactly what
        ``DP_FLOOD_SUPPORT`` detects); only the vanishing energy gap fooled the
        assignment. This seed feeds ``_relocate_degenerate_patches`` so the
        block is exchanged into the correct slot. A genuine 50/50 crossing has
        no 0.8 partner (dp ~ 0.7 each) and produces no seed -- it is left to
        basisrotation, untouched. Returned as ``(dp_assigned, k, kn, slot)``
        sorted worst (lowest dp) first, the same shape tier 1/2 seeds use.
        '''
        bf = self.bands_final
        neigh = np.asarray(self.neighbors)
        nks, nslots = bf.shape
        karr = np.arange(nks)
        cert = self._certain_partner_map(DP_FLOOD_SUPPORT)
        close = self._near_degenerate_mask()

        best: dict = {}
        for d in range(neigh.shape[1]):
            kn_col = neigh[:, d]
            ok = kn_col >= 0
            kns = np.where(ok, kn_col, 0)
            b1 = np.where(bf >= 0, bf, 0)
            b2v = bf[kns]
            b2 = np.where(b2v >= 0, b2v, 0)
            near = (close[karr[:, None], b1] | close[kns[:, None], b2]
                    | (self.signal_final == DEGENERATE)
                    | (self.signal_final[kns] == DEGENERATE))
            comp = ok[:, None] & (bf >= 0) & (b2v >= 0) & (bf != b2v) & near
            succ = cert[:, d, :][karr[:, None], b1]     # flood successor (0.8) of the held state
            comp &= (succ >= 0) & (succ != b2v)
            dpv = self.connections[karr[:, None], d, b1, b2]
            for k, s in zip(*np.where(comp)):
                kn = int(kns[k])
                key = (min(int(k), kn), max(int(k), kn), int(s))
                v = float(dpv[k, s])
                if key not in best or v < best[key]:
                    best[key] = v
        out = [(v, a, c, s) for (a, c, s), v in best.items()]
        out.sort()
        return out

    def _log_refuted_census(self, tag: str) -> int:
        '''
        Log the global strict refuted-edge census (len of _dp_refuted_seam_edges)
        so consecutive runs are comparable pass-by-pass without an external
        audit script. Cheap: the 0.9 partner map is cached.
        '''
        n = len(self._dp_refuted_seam_edges())
        self.logger.info(f'{BODY_INDENT}dp-refuted census [{tag}]: {n} edge(s)')
        return n

    def _exchange_region_counts(self, R: set, s: int, t: int, swapped: bool,
                                cert_acc: np.ndarray,
                                ignore_close: bool = False,
                                wall_gate: np.ndarray = None) -> Tuple[int, int, int]:
        '''
        (gross-jump walls, split ``cert_acc`` links, strict refuted edges) over
        every edge touching region ``R`` for slots s and t, in the current
        (``swapped=False``) or the exchanged (``swapped=True``) configuration.
        The refuted count applies ``_dp_refuted_seam_edges``' predicate verbatim
        (0.9 map, corridor- and DEGENERATE-exempt, the bands must differ),
        counted undirected like the census. Since an s<->t exchange only changes
        edges of slots s,t touching R, the refuted delta IS the global census
        delta: an acceptance guard built on it is exact, not heuristic.
        Signals travel with the exchange, so the DEGENERATE exemption is read
        through the same source columns as the bands.

        ``ignore_close=True`` drops the near-degenerate exemption from the SPLIT
        ``cert_acc``-link count only (the degen-patch pass' halving metric, whose
        seams live inside collapsed gaps and would otherwise all be exempted to
        zero). The strict refuted census stays corridor- and DEGENERATE-exempt
        regardless, so it remains comparable to the global ``_log_refuted_census``.

        ``wall_gate`` overrides the gross-jump gate the wall count keys to. Left
        None it is the global ``accept_E`` (every seed except ``'cliff'``), and a
        wall is any same-slot edge over that flat gate. Passed the per-(k, band)
        ``continuity_gate`` array (the ``'cliff'`` pass) a wall is instead a
        raw-band SWITCH edge whose step exceeds the LOCAL gate ``min`` of the two
        endpoints -- the sub-accept_E cliffs; same-band steps (however steep) are
        never counted, so a legitimate steep dispersion cannot block a repair.
        '''
        bf = self.bands_final
        ev = self.eigenvalues
        neigh = np.asarray(self.neighbors)
        n_dir = neigh.shape[1]
        cert = self._certain_partner_map()
        close = self._near_degenerate_mask()
        n_wall = n_cert = 0
        ref_keys: set = set()
        for k1 in R:
            for d in range(n_dir):
                k2 = int(neigh[k1, d])
                if k2 < 0:
                    continue
                in2 = k2 in R
                for slot, other in ((s, t), (t, s)):
                    src1 = other if swapped else slot            # column the content
                    src2 = other if (swapped and in2) else slot  # is read from
                    b1 = int(bf[k1, src1])
                    b2 = int(bf[k2, src2])
                    if b1 < 0 or b2 < 0:
                        continue
                    if wall_gate is None:
                        if abs(ev[k1, b1] - ev[k2, b2]) > self.accept_E:
                            n_wall += 1                       # gross wall (flat accept_E)
                    elif b1 != b2 and abs(ev[k1, b1] - ev[k2, b2]) > min(
                            wall_gate[k1, b1], wall_gate[k2, b2]):
                        n_wall += 1                           # local cliff (switch only)
                    p = int(cert_acc[k1, d, b1])
                    cl_cert = False if ignore_close else (close[k1, b1] or close[k2, b2])
                    if p >= 0 and p != b2 and not cl_cert:
                        n_cert += 1
                    if (b1 != b2
                            and self.signal_final[k1, src1] != DEGENERATE
                            and self.signal_final[k2, src2] != DEGENERATE
                            and not (close[k1, b1] or close[k2, b2])
                            and int(cert[k1, d, b1]) >= 0
                            and int(cert[k1, d, b1]) != b2):
                        ref_keys.add((min(k1, k2), max(k1, k2), slot))
        return n_wall, n_cert, len(ref_keys)

    def _seam_partner_candidates(self, anchor: int, start: int, s: int,
                                 use_dp: bool, dp_tol: float = DP_CERTAIN) -> list:
        '''
        Partner slots ``t`` for exchanging slot ``s`` content at the seam cell
        ``start``, as ``[(energy_mismatch, t), ...]`` best first: the (at most
        two) energy-matched slots within accept_E of the anchored state's
        energy -- content never crosses a genuine inter-band wall -- and, with
        ``use_dp``, the slot holding the ``dp_tol``-certain successor of the
        anchored state front-inserted (dp names the partner outright, still
        subject to the accept_E veto). The degen-patch pass passes
        ``dp_tol=DP_FLOOD_SUPPORT`` so a sub-certain-but-clear (0.8) successor
        still names the target slot.
        '''
        bf = self.bands_final
        ev = self.eigenvalues
        nslots = bf.shape[1]
        ba = int(bf[anchor, s])
        if ba < 0:
            return []
        Ea = float(ev[anchor, ba])
        cands = sorted((abs(float(ev[start, bf[start, t]]) - Ea), t)
                       for t in range(nslots)
                       if t != s and bf[start, t] >= 0)
        cands = [c for c in cands[:2] if c[0] <= self.accept_E]
        if use_dp:
            d_as = self._dir_index().get((anchor, start), -1)
            succ = int(self._certain_partner_map(dp_tol)[anchor, d_as, ba]) if d_as >= 0 else -1
            if succ >= 0:
                for t in range(nslots):
                    if t != s and bf[start, t] == succ:
                        dmatch = abs(float(ev[start, succ]) - Ea)
                        if dmatch <= self.accept_E:
                            cands.insert(0, (dmatch, t))
                        break
        return cands

    def _apply_slot_exchange(self, R: set, s: int, t: int) -> list:
        '''
        Exchange slots s and t over every k-point of ``R`` on the final canvas
        (bands, both signals, and the forced/repaired audit masks travel with
        the content -- the per-row exchange preserves the bijection by
        construction). Returns the changed (k, slot) cells for re-grading.
        '''
        bf = self.bands_final
        changed = []
        for kk in sorted(R):
            for arr in (bf, self.signal_final, self.correct_signalfinal):
                arr[kk, s], arr[kk, t] = arr[kk, t], arr[kk, s]
            for name in ('forced_mask', 'repaired_mask'):
                m = getattr(self, name, None)
                if m is not None:                    # audit travels with the content
                    m[kk, s], m[kk, t] = m[kk, t], m[kk, s]
            changed.extend(((kk, s), (kk, t)))
        return changed

    def _apply_slot_rotation(self, R: set, order: tuple) -> list:
        '''
        Rotate slots along ``order`` = (c0, c1, ..., c_{L-1}) over every
        k-point of ``R`` on the final canvas: each slot takes the content of
        the NEXT slot in ``order`` (``new[c_i] = old[c_{i+1 mod L}]``), so a
        slot whose certain successor currently sits in the next cycle member
        recovers its own band. Bands, both signals and the forced/repaired
        audit masks travel together; the per-row cyclic permutation preserves
        the bijection by construction. A length-2 ``order`` is exactly
        ``_apply_slot_exchange``. Returns the changed (k, slot) cells.
        '''
        bf = self.bands_final
        L = len(order)
        arrays = [bf, self.signal_final, self.correct_signalfinal]
        arrays += [m for m in (getattr(self, 'forced_mask', None),
                               getattr(self, 'repaired_mask', None)) if m is not None]
        changed = []
        for kk in sorted(R):
            for arr in arrays:
                vals = [arr[kk, c] for c in order]           # snapshot before writing
                for i, c in enumerate(order):
                    arr[kk, c] = vals[(i + 1) % L]
            changed.extend((kk, c) for c in order)
        return changed

    def _regrade_cells(self, changed_cells: list) -> None:
        '''
        Re-grade on the final canvas: both signals of the exchanged cells and
        the energy-continuity signal of their same-slot neighbours. The audit
        masks moved with the content, so force-filled cells keep their FORCED
        marker wherever they landed.
        '''
        bf = self.bands_final
        neigh = np.asarray(self.neighbors)
        changed_set = set(changed_cells)
        regrade = set(changed_cells)
        for k, s in changed_cells:
            for kn in neigh[k]:
                if kn >= 0:
                    regrade.add((int(kn), s))
        forced = getattr(self, 'forced_mask', None)
        for k, s in sorted(regrade):
            if self.signal_final[k, s] == DEGENERATE:
                continue
            if (k, s) in changed_set:
                b1 = int(bf[k, s])
                if b1 >= 0:
                    values = []
                    for i_neig, kn in enumerate(self.neighbors[k]):
                        if kn == -1:
                            continue
                        b2 = int(bf[kn, s])
                        if b2 < 0:
                            continue
                        values.append(self.connections[k, i_neig, b1, b2])
                    self.signal_final[k, s] = min(evaluate_result(values) if values else NOT_SOLVED,
                                                  POTENTIAL_CORRECT)
                    if forced is not None and forced[k, s]:
                        self.signal_final[k, s] = FORCED
                        self.correct_signalfinal[k, s] = FORCED_CONTINUITY
                        continue
            elif self.correct_signalfinal[k, s] == FORCED_CONTINUITY:
                continue
            if int(bf[k, s]) < 0:
                self.correct_signalfinal[k, s] = NOT_SOLVED
                continue
            sig, _ = evaluate_point(self.dimensions, k, s, self.kpoints_index,
                                    self.matrix, bf, self.eigenvalues,
                                    accept_E=self.accept_E, disp_scale=self.disp_scale,
                                    local_gap=self.local_gap, band_disp=self.band_disp)
            self.correct_signalfinal[k, s] = sig

    def _relocate_wall_patches(self, max_patches: int = 64) -> int:
        '''
        Tier-1 patch relocation, seeded on the SILENT cliffs (gross energy
        jumps > accept_E between same-slot neighbours).

        WHERE THE CLIFFS FORM. In-loop, the seeded region-growing splits a fused
        multi-band component by racing label frontiers (cheapest dot-product step
        first). A frontier can walk through a crossing corridor onto the other
        band's sheet -- every one-step energy residual inside the corridor passes
        the local gate -- and claim a chunk of it before the rightful label
        arrives. The two frontiers then collide MID-SHEET, and nothing in the
        race makes the collision line coincide with the corridor: the wall
        between the labels lands on a gross energy jump. The patch interior is
        internally perfect (its dp support comes from its equally-swapped
        neighbours), so evaluate_point's 3-of-4 census validates it, the
        continuity-edge ratchet re-injects it on every following iteration, and
        the unanimity-gated post-loop repairs refuse to touch it because BOTH
        sides of the wall hold dp ~0.99 mandates (it is a genuinely two-sided
        disagreement between two internally-consistent regions). It therefore
        ships as a validated band with a visible cliff -- the Silent column.

        The exchange machinery is shared with the dp-seeded tier 2; see
        ``_relocate_patches`` for the flood and acceptance rules.
        '''
        return self._relocate_patches('walls', max_patches)

    def _relocate_cliff_patches(self, max_patches: int = 64) -> int:
        '''
        The tighter sibling of ``_relocate_wall_patches``: the SUB-accept_E
        cliffs. The same mid-sheet collision that produces a gross wall (see
        ``_relocate_wall_patches``) also lands on jumps that are large on the
        scale of the LOCAL band gap yet still below the flat gross-jump gate --
        the phosphorene band-13 seam (~1 eV, but accept_E ~ 1.02 eV, so it reads
        as "0 silent"). Seeded on ``_cliff_seam_edges`` (raw-band switch edges
        whose step exceeds the per-(k, band) ``continuity_gate``), it exchanges
        the mis-slotted patch onto the crossing corridor exactly as tier 1 does.

        Same energy-based flood and acceptance as tier 1 (``_relocate_patches``),
        but the wall count is taken at the local ``continuity_gate`` instead of
        the flat ``accept_E`` (``wall_gate`` argument), and only over switch
        edges -- a legitimate steep one-band step never counts, so it can neither
        seed nor block a repair. Near-degenerate / DEGENERATE cells are exempt
        (gauge -- basisrotation), so a genuine 50/50 crossing is left untouched.
        Runs after tier 1 so the gross walls are already placed. A cliff no
        pairwise exchange can remove survives to ``_verify_no_cliffs`` and is
        graded honestly (CHECK/FAIL), never shipped as a silent USABLE band.
        '''
        return self._relocate_patches('cliff', max_patches)

    def _relocate_dp_seam_patches(self, max_patches: int = 64) -> int:
        '''
        Tier-2 patch relocation, seeded on dp-REFUTED seams (split certain
        links, ``_dp_refuted_seam_edges``): mis-slotted patches whose energy
        step sits BELOW accept_E, so tier 1 never sees them, while the
        wavefunction evidence -- the quantity this program exists to make
        continuous -- is broken across the seam with near-certainty
        (phosphorene run 5: the 13<->15 staircase attenuated to 93% of the
        gate, the 10-12 mirror filament, the 19-21 skirt).

        Same patch-exchange machinery as tier 1 with the roles of the two
        guards inverted (see ``_relocate_patches``): seeds and flood cost come
        from dp, acceptance requires the split certain links to at least HALVE
        while gross-jump walls may not increase. Running after tier 1 the
        touched edges normally carry zero walls, so "not increase" means no
        exchange may ever create a jump > accept_E: dp evidence acts strictly
        within the energy gate. Cell-scale residue (row cycles, braids) is
        left to _repair_certain_links, unchanged.

        Cannot undo tier 1: tier-1 exchanges land their seams on crossing
        corridors where dp is sub-certain, which produces no split certain
        links and is therefore never seeded here.
        '''
        return self._relocate_patches('dp', max_patches)

    def _relocate_degenerate_patches(self, max_patches: int = 64) -> int:
        '''
        Near-degenerate patch relocation, seeded on ``_degenerate_seam_edges``:
        the small patches that attached to the WRONG member of a near-degenerate
        band pair. Tiers 1/2 cannot see them -- the energy step hides below the
        gate (near-degenerate) and the strict 0.9 seam predicate exempts the
        collapsed-gap cells as gauge -- yet the wavefunction evidence still
        cleanly names the correct band (dp ~ 1, caught by the 0.8 flood map).

        Same exchange machinery as tier 2 (``_relocate_patches('degen')``): dp
        flood cost and pattern constraint, partner slot named by the 0.8-certain
        successor (energy-vetoed at accept_E, trivially satisfied here). An
        exchange is accepted when the near-degenerate split-link count at least
        HALVES and no gross-jump wall appears; with ``DEGEN_FREEZE_REFUTED`` the
        strict refuted census may additionally not rise on the border.

        Complementary to basisrotation, not competing: this only moves the
        near-degenerate subspace into the correct SLOT (healing the seam);
        resolving the intra-subspace gauge is still basisrotation's job, now
        started from the right slot. Genuine 50/50 crossings have no 0.8 partner,
        produce no seed, and are left untouched.
        '''
        return self._relocate_patches('degen', max_patches)

    def _relocate_patches(self, seed: str, max_patches: int = 64) -> int:
        '''
        Shared patch-relocation core: exchange whole mis-slotted patches
        between two slots, relocating the label seam onto the crossing corridor
        where it belongs.

        Since the per-k bijection holds, a slot sitting on the wrong sheet
        forces some other slot to hold its sheet: the defect is an ORDER-2
        exchange over a k-region (verified on the real canvas: every wall edge
        has an energy-matched partner slot on the far side). For the worst seed
        edge first, the partner slot ``t`` is identified by energy continuation
        across the seam (``seed='dp'`` additionally lets the certain successor
        of the anchored state name ``t`` outright, still energy-vetoed at
        accept_E so content is never pulled across a genuine inter-band wall),
        and the patch is flooded with a seam-cost rule: a neighbouring k-point
        joins the patch iff swapping it too is strictly cheaper than leaving
        the seam on that edge -- cost measured in energy steps (``'walls'``) or
        dp mis-continuity (``'dp'``). At a corridor the two options tie, inside
        correct territory keeping wins: the flood is self-bounding. The dp
        flood is additionally confined to the seed cell's (raw_s, raw_t)
        PATTERN region: an edge's gluing is invariant under swapping both of
        its endpoints, so the cost rule alone cannot tell a wrong patch
        interior from correct territory, and sub-gate strips hug genuine
        corridors through which an unconstrained flood leaks (runaway); a
        same-pattern region is sheet-coherent by construction (any crossing
        involving either slot's sheet flips the pattern), and a multi-pattern
        wrong strip is simply relocated as several patches over the rounds.
        The dp acceptance counts split links at ``DP_FLOOD_SUPPORT`` instead
        of ``DP_CERTAIN`` so a patch cannot relocate its damage into merely
        sub-certain seams. The s<->t exchange over the patch is accepted only when,
        over every edge touching the patch,

          * ``seed='walls'``: the gross-jump wall count at least HALVES and the
            split certain links (dp >= DP_CERTAIN steps assigned across labels,
            near-degenerate endpoints exempt) do not increase;
          * ``seed='dp'``:   the split certain links at least HALVE and the
            gross-jump wall count does not increase;
          * both tiers: the strict refuted-edge census (0.9-map predicate of
            _dp_refuted_seam_edges) may not increase. Without this guard the
            tier-2 halving acts on the looser 0.8-map count -- a SUPERSET of
            the refuted edges -- so an accepted exchange could reduce 0.8-splits
            while increasing strict refuted edges on its border (observed:
            run-6 refuted 429 -> 474).

        Halving (not mere decrease) keeps out the marginal mega-exchanges that
        shuffle hundreds of cells for a few defects and relocate damage instead
        of removing it (A/B on the real canvas: new silent cells 24 -> 1, walls
        healed 450->47 vs 450->32). Signals and the forced/repaired audit
        masks travel with the exchanged content, changed cells are re-graded on
        the final canvas, and the per-row exchange preserves the bijection by
        construction. Braids that no pairwise exchange improves are left alone
        and keep their honest CHECK/FAIL grade.

        Runs before _repair_certain_links: the patches fixed here are exactly
        the two-sided defects unanimity cannot arbitrate, and removing them
        turns their seam cells into trusted anchors for the remaining passes.
        Returns the number of exchanged cells.
        '''
        bf = self.bands_final
        ev = self.eigenvalues
        conn = self.connections
        neigh = np.asarray(self.neighbors)
        nks, nslots = bf.shape
        n_dir = neigh.shape[1]
        cert = self._certain_partner_map()
        # dp acceptance counts split links against the looser support map:
        # a patch may not zero its certain links by relocating the damage
        # into merely sub-certain (0.8-0.9) seams the 0.9 map cannot see.
        cert_acc = cert if seed in ('walls', 'cliff') else self._certain_partner_map(DP_FLOOD_SUPPORT)
        is_degen = seed == 'degen'
        is_cliff = seed == 'cliff'
        # the cliff pass counts walls at the per-(k, band) local continuity gate
        # over raw-band SWITCH edges only; every other seed keeps the flat accept_E.
        wall_gate = self.continuity_gate if is_cliff else None
        max_patch_cells = nks // 3
        label = {'walls': 'Wall-patch relocation', 'dp': 'dp-seam relocation',
                 'degen': 'degenerate-patch relocation',
                 'cliff': 'cliff-patch relocation'}[seed]
        unit = {'walls': 'wall edge(s)', 'dp': 'dp-refuted seam edge(s)',
                'degen': 'near-degenerate seam edge(s)',
                'cliff': 'sub-gate cliff edge(s)'}[seed]

        def grow(start: int, s: int, t: int) -> Union[set, None]:
            # Seam-cost flood: k2 joins iff moving the seam past it is strictly
            # cheaper than leaving the seam on the (k1, k2) edge.
            R = {start}
            queue = deque([start])
            while queue:
                k1 = queue.popleft()
                bs1, bt1 = int(bf[k1, s]), int(bf[k1, t])
                for d in range(n_dir):
                    k2 = int(neigh[k1, d])
                    if k2 < 0 or k2 in R:
                        continue
                    bs2, bt2 = int(bf[k2, s]), int(bf[k2, t])
                    if bs2 < 0 or bt2 < 0 or bs1 < 0 or bt1 < 0:
                        continue
                    if seed in ('walls', 'cliff'):
                        keep = abs(ev[k1, bt1] - ev[k2, bs2]) + abs(ev[k1, bs1] - ev[k2, bt2])
                        swap = abs(ev[k1, bt1] - ev[k2, bt2]) + abs(ev[k1, bs1] - ev[k2, bs2])
                    else:
                        # Pattern-constrained: within a same-(raw_s, raw_t)
                        # region both slots follow fixed raw bands and any
                        # crossing involving either slot's sheet flips the
                        # pattern, so the region is sheet-coherent -- the
                        # strip cannot leak through a corridor or a braid
                        # junction onto correctly-labelled territory (an
                        # unconstrained dp-cost flood does: edge gluing is
                        # invariant under swapping both endpoints, so cost
                        # alone cannot tell wrong-interior from correct).
                        if (bs2, bt2) != (bs1, bt1):
                            continue
                        c12 = conn[k1, d]
                        keep = (1.0 - c12[bt1, bs2]) + (1.0 - c12[bs1, bt2])
                        swap = (1.0 - c12[bt1, bt2]) + (1.0 - c12[bs1, bs2])
                    if swap < keep:
                        R.add(k2)
                        queue.append(k2)
                        if len(R) > max_patch_cells:
                            return None                  # runaway: not a patch
            return R

        seed_edges = {'walls': self._slot_wall_edges,
                      'dp': self._dp_refuted_seam_edges,
                      'degen': self._degenerate_seam_edges,
                      'cliff': self._cliff_seam_edges}[seed]

        changed_cells: list = []
        n_patch = 0
        edges0 = None
        for _ in range(max_patches):
            worst = seed_edges()
            if edges0 is None:
                edges0 = len(worst)
            if not worst:
                break
            accepted = None
            tried = set()
            for metric, k, kn, s in worst:
                for anchor, start in ((k, kn), (kn, k)):
                    for dmatch, t in self._seam_partner_candidates(
                            anchor, start, s, use_dp=(seed not in ('walls', 'cliff')),
                            dp_tol=(DP_FLOOD_SUPPORT if is_degen else DP_CERTAIN)):
                        key = (start, s, t)
                        if key in tried:
                            continue
                        tried.add(key)
                        R = grow(start, s, t)
                        if R is None:
                            continue
                        w0, c0, r0 = self._exchange_region_counts(R, s, t, False, cert_acc,
                                                                  ignore_close=is_degen,
                                                                  wall_gate=wall_gate)
                        w1, c1, r1 = self._exchange_region_counts(R, s, t, True, cert_acc,
                                                                  ignore_close=is_degen,
                                                                  wall_gate=wall_gate)
                        if seed in ('walls', 'cliff'):
                            ok_acc = w1 < w0 and 2 * w1 <= w0 and c1 <= c0 and r1 <= r0
                        else:
                            ok_acc = c1 < c0 and 2 * c1 <= c0 and w1 <= w0
                            if not is_degen or DEGEN_FREEZE_REFUTED:
                                ok_acc = ok_acc and r1 <= r0
                        if ok_acc:
                            accepted = (R, s, t, w0, w1, c0, c1, r0, r1)
                            break
                    if accepted:
                        break
                if accepted:
                    break
            if accepted is None:
                break
            R, s, t, w0, w1, c0, c1, r0, r1 = accepted
            changed_cells.extend(self._apply_slot_exchange(R, s, t))
            n_patch += 1
            self.logger.info(f'{BODY_INDENT}{label}: slots ({s},{t}) '
                             f'exchanged over {len(R)} k-point(s) -- walls {w0}->{w1}, '
                             f'split certain links {c0}->{c1}, refuted {r0}->{r1}')

        if not changed_cells:
            self.logger.info(f'{BODY_INDENT}{label}: no exchangeable patch '
                             f'({edges0 or 0} {unit} on the final canvas)')
            return 0
        self.logger.info(f'{BODY_INDENT}{label}: {n_patch} patch(es), '
                         f'{len(changed_cells)} cell(s) exchanged; {unit} '
                         f'{edges0} -> {len(seed_edges())}')

        self._regrade_cells(changed_cells)
        return len(changed_cells)

    def _pattern_region(self, start: int, s: int, t: int) -> set:
        '''
        The k-connected region around ``start`` where slots s and t hold the
        same raw-band pair as at ``start`` -- the pattern constraint of the
        greedy dp flood without its cost rule (the min-cut supplies the
        labeling the greedy rule approximated). Sheet-coherent by construction:
        any crossing involving either slot's sheet flips the pattern. No
        runaway cap is needed -- the min-cut may legally swap any subset (the
        seed is pinned, so never none).
        '''
        bf = self.bands_final
        neigh = np.asarray(self.neighbors)
        pat = (int(bf[start, s]), int(bf[start, t]))
        if pat[0] < 0 or pat[1] < 0:
            return set()
        R = {start}
        queue = deque([start])
        while queue:
            k1 = queue.popleft()
            for k2 in neigh[k1]:
                k2 = int(k2)
                if k2 < 0 or k2 in R:
                    continue
                if (int(bf[k2, s]), int(bf[k2, t])) == pat:
                    R.add(k2)
                    queue.append(k2)
        return R

    def _pattern_region_cycle(self, start: int, slots: tuple) -> set:
        '''
        ``_pattern_region`` generalized to a set of ``slots``: the k-connected
        region around ``start`` where EVERY slot in ``slots`` holds the same
        raw band as at ``start`` (the cyclic pattern). Sheet-coherent by the
        same argument -- any crossing involving one of the slots' sheets flips
        the tuple -- so the region is bounded by the crossing corridors. Used
        by ``_repair_dp_cycles`` to size the atomic rotation of a permutation
        cycle. Capped at ``nks // 3`` (a rotation over a third of the BZ is not
        a patch); returns the empty set on runaway or an unattributed pattern.
        '''
        bf = self.bands_final
        neigh = np.asarray(self.neighbors)
        pat = tuple(int(bf[start, s]) for s in slots)
        if any(b < 0 for b in pat):
            return set()
        cap = self.nks // 3
        R = {start}
        queue = deque([start])
        while queue:
            k1 = queue.popleft()
            for k2 in neigh[k1]:
                k2 = int(k2)
                if k2 < 0 or k2 in R:
                    continue
                if tuple(int(bf[k2, s]) for s in slots) == pat:
                    R.add(k2)
                    queue.append(k2)
                    if len(R) > cap:
                        return set()                 # runaway: not a patch
        return R

    def _mincut_labeling(self, start: int, s: int, t: int) -> Union[set, None]:
        '''
        Exact binary seam placement for the (s, t) pattern region around
        ``start``: one keep/swap variable per k-point, solved as an s-t min-cut
        (scipy maximum_flow). Pairwise capacities are the seam-laying cost
        minus the gluing cost of each grid edge (clamped at 0 -- the
        non-submodular clamp only under-counts seam cost, and the caller's
        exact global guard vets the final labeling); border cells outside the
        region are fixed to keep via terminal attachments, and the seed cell is
        pinned to swap so the trivial labeling is excluded. A seam step is free
        inside a near-degenerate corridor and on a certain-forest refusal cut
        (the physical branch cuts), pays the dp-certainty weight where it
        splits a certain link, and is impossible (INF) across a gross energy
        jump. Returns the swap set, or None when no finite seam exists.
        '''
        SCALE = 1000
        INF_CAP = 2 ** 29
        bf = self.bands_final
        ev = self.eigenvalues
        conn = self.connections
        neigh = np.asarray(self.neighbors)
        nks = self.nks
        n_dir = neigh.shape[1]
        close = self._near_degenerate_mask()
        cert = self._certain_partner_map()
        dir_idx = self._dir_index()
        refusals = getattr(self, '_sheet_refusals', None) or set()

        B_s, B_t = int(bf[start, s]), int(bf[start, t])
        R = self._pattern_region(start, s, t)
        if len(R) < 2:
            return None
        Rl = sorted(R)
        pos = {k: i for i, k in enumerate(Rl)}

        def step_cost(k1: int, d: int, k2: int, b1: int, b2: int) -> float:
            # cost of the same-slot step b1@k1 -> b2@k2 along direction d
            if abs(ev[k1, b1] - ev[k2, b2]) > self.accept_E:
                return np.inf                    # a step never crosses a gross jump
            cost = 0.0
            p = int(cert[k1, d, b1])
            if p >= 0 and p != b2 and not (close[k1, b1] or close[k2, b2]):
                # splitting a certain link costs its dp weight -- unless the
                # link is a forest refusal cut (branch cut: unkeepable anyway)
                if frozenset((k1 + b1 * nks, k2 + p * nks)) not in refusals:
                    cost += float(conn[k1, d, b1, p])
            # smooth tie-break toward character-continuous placement; epsilon
            # keeps it below any single split-link weight
            cost += 1e-3 * (1.0 - float(conn[k1, d, b1, b2]))
            return cost

        def as_cap(w: float) -> int:
            return INF_CAP if not np.isfinite(w) else int(round(SCALE * w))

        caps: dict = {}

        def add_cap(u: int, v: int, w: int) -> None:
            if w > 0:
                caps[(u, v)] = min(caps.get((u, v), 0) + w, INF_CAP)
            caps.setdefault((u, v), 0)
            caps.setdefault((v, u), 0)           # reverse entry for the residual

        n_clamped = 0
        # node 0 = source (keep side), node 1 = sink (swap side), cell i -> 2+i
        for k1 in Rl:
            u = 2 + pos[k1]
            for d in range(n_dir):
                k2 = int(neigh[k1, d])
                if k2 < 0:
                    continue
                if k2 in pos:
                    if k1 > k2:
                        continue                 # one undirected capacity per pair,
                    d21 = dir_idx.get((k2, k1))  # both directional costs summed
                    sides = [(k1, d, k2)] + ([(k2, d21, k1)] if d21 is not None else [])
                    g = c = 0.0
                    for ka, dd, kb in sides:
                        g += step_cost(ka, dd, kb, B_s, B_s) + step_cost(ka, dd, kb, B_t, B_t)
                        c += step_cost(ka, dd, kb, B_s, B_t) + step_cost(ka, dd, kb, B_t, B_s)
                    # objective per edge = const + [labels differ] * (c - g);
                    # clamp handles the non-submodular case c < g (a glue step
                    # that itself splits a certain link while the seam step
                    # follows one): the edge becomes free to cut, which only
                    # UNDER-counts seam cost -- acceptance stays exact.
                    if np.isfinite(c) and np.isfinite(g) and c < g:
                        n_clamped += 1
                    if not np.isfinite(c):
                        w = INF_CAP
                    elif not np.isfinite(g):
                        w = 0
                    else:
                        w = max(as_cap(c - g), 0)
                    v = 2 + pos[k2]
                    add_cap(u, v, w)
                    add_cap(v, u, w)
                else:
                    C_s, C_t = int(bf[k2, s]), int(bf[k2, t])
                    if C_s < 0 or C_t < 0:
                        continue                 # matches _exchange_region_counts
                    th_keep = step_cost(k1, d, k2, B_s, C_s) + step_cost(k1, d, k2, B_t, C_t)
                    th_swap = step_cost(k1, d, k2, B_t, C_s) + step_cost(k1, d, k2, B_s, C_t)
                    if not (np.isfinite(th_keep) or np.isfinite(th_swap)):
                        return None              # conflicted border cell: no finite seam
                    add_cap(u, 1, as_cap(th_keep))   # paid when the cell keeps
                    add_cap(0, u, as_cap(th_swap))   # paid when the cell swaps
        add_cap(2 + pos[start], 1, INF_CAP)      # pin the seed to swap

        n_nodes = len(Rl) + 2
        rows = np.fromiter((uv[0] for uv in caps), int, len(caps))
        cols = np.fromiter((uv[1] for uv in caps), int, len(caps))
        data = np.fromiter(caps.values(), np.int32, len(caps))
        graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.int32)
        res = maximum_flow(graph, 0, 1)
        if res.flow_value >= INF_CAP:
            return None                          # no finite seam exists: refuse
        # keep side = nodes reachable from the source in the residual network
        residual = graph - res.flow
        seen = np.zeros(n_nodes, bool)
        seen[0] = True
        queue = deque([0])
        indptr, indices, rdata = residual.indptr, residual.indices, residual.data
        while queue:
            u = queue.popleft()
            for e in range(indptr[u], indptr[u + 1]):
                v = indices[e]
                if rdata[e] > 0 and not seen[v]:
                    seen[v] = True
                    queue.append(v)
        S = {k for k in Rl if not seen[2 + pos[k]]}
        if n_clamped:
            self.logger.debug(f'{BODY_INDENT}min-cut ({s},{t}): {n_clamped} '
                              f'non-submodular edge(s) clamped over {len(Rl)} cells')
        return S if S else None

    def _mincut_seam_relocate(self, max_moves: int = 64) -> int:
        '''
        S2 of the dp-seam correction plan: place the label seam of each
        remaining refuted (s, t) pattern region at its exact optimum via s-t
        min-cut, iterating pairwise exchanges alpha-expansion-style over the
        refuted-edge seeds until no pair improves. Runs AFTER the greedy
        dp-seam relocation (which shrinks its work; every greedy-accepted move
        is a feasible labeling here) and before _repair_certain_links, for the
        same reason the greedy passes do. Each move is accepted only on a
        STRICT decrease of the exact global refuted-edge census (no halving --
        the optimum needs no marginal-exchange filter) with walls and split
        support-links not increasing, so the pass terminates: every accepted
        move decreases a non-negative integer. Braids no pairwise exchange
        improves are honest residue (sub-certain: basisrotation territory).
        '''
        if not DP_SEAM_MINCUT:
            return 0
        bf = self.bands_final
        # forest refusal cuts = free seam locations (computed once, cached);
        # post-loop only -- this cannot freeze anything
        self._certain_sheet_forest()
        cert_acc = self._certain_partner_map(DP_FLOOD_SUPPORT)

        moves = 0
        total_cells = 0
        edges0 = None
        while moves < max_moves:
            seeds = self._dp_refuted_seam_edges()
            if edges0 is None:
                edges0 = len(seeds)
            if not seeds:
                break
            accepted = False
            tried = set()
            for metric, k, kn, s in seeds:
                for anchor, start in ((k, kn), (kn, k)):
                    for dmatch, t in self._seam_partner_candidates(
                            anchor, start, s, use_dp=True):
                        key = (int(bf[start, s]), int(bf[start, t]), s, t)
                        if key in tried:
                            continue
                        tried.add(key)
                        S = self._mincut_labeling(start, s, t)
                        if not S:
                            continue
                        w0, c0, r0 = self._exchange_region_counts(S, s, t, False, cert_acc)
                        w1, c1, r1 = self._exchange_region_counts(S, s, t, True, cert_acc)
                        if not (r1 < r0 and w1 <= w0 and c1 <= c0):
                            continue
                        self._regrade_cells(self._apply_slot_exchange(S, s, t))
                        moves += 1
                        total_cells += 2 * len(S)
                        self.logger.info(f'{BODY_INDENT}min-cut seam placement: slots '
                                         f'({s},{t}) exchanged over {len(S)} k-point(s) '
                                         f'-- walls {w0}->{w1}, split links {c0}->{c1}, '
                                         f'refuted {r0}->{r1}')
                        accepted = True
                        break
                    if accepted:
                        break
                if accepted:
                    break
            if not accepted:
                break                            # no pair improves: honest residue
        if moves:
            self.logger.info(f'{BODY_INDENT}min-cut seam placement: {moves} move(s), '
                             f'{total_cells} cell(s) exchanged; refuted seam edges '
                             f'{edges0} -> {len(self._dp_refuted_seam_edges())}')
        else:
            self.logger.info(f'{BODY_INDENT}min-cut seam placement: no improving move '
                             f'({edges0 or 0} refuted seam edge(s) on the final canvas)')
        return total_cells

    def _dp_cycle_at(self, k: int, kn: int) -> Union[tuple, None]:
        '''
        Detect a k-local permutation CYCLE at ``kn`` mandated by the trusted
        neighbour ``k`` across the (k -> kn) edge, from the CERTAIN successor
        map (dp >= DP_CERTAIN). For each slot ``s`` whose k-content ``b_s`` has
        a certain successor ``p_s`` toward ``kn``, band ``p_s`` BELONGS in slot
        ``s`` at ``kn`` -- but currently lives in the slot ``t`` where
        ``bf[kn, t] == p_s``. That defines ``src[s] = t`` (slot s must receive
        the content of slot t). A closed cycle in ``src`` whose every leg is a
        certain, non-degenerate switch is a wrong cyclic relabelling that no
        pairwise exchange can unwind (the first swap only relocates a leg).

        Returns the cycle as an ``order`` tuple for ``_apply_slot_rotation``
        (``new[order[i]] = old[order[i+1]]``, i.e. ``order[i+1] = src[order[i]]``)
        containing the seed relation of the (k, kn) edge, or None when no such
        cycle exists here. A genuine crossing corridor has sub-certain dp / is
        near-degenerate, so it produces no certain cycle and is never returned.
        '''
        bf = self.bands_final
        d = self._dir_index().get((k, kn), -1)
        if d < 0:
            return None
        cert = self._certain_partner_map()               # DP_CERTAIN
        close = self._near_degenerate_mask()
        nslots = bf.shape[1]
        # band held at kn -> its slot (unique; the row is a permutation)
        band_to_slot = {int(bf[kn, t]): t for t in range(nslots) if int(bf[kn, t]) >= 0}
        src = {}                                          # slot s <- content of slot src[s]
        for s in range(nslots):
            b_s = int(bf[k, s])
            if b_s < 0 or self.signal_final[k, s] == DEGENERATE:
                continue
            p_s = int(cert[k, d, b_s])                    # certain successor toward kn
            if p_s < 0 or p_s == int(bf[kn, s]):
                continue                                  # no mandate, or already satisfied
            t = band_to_slot.get(p_s, -1)
            if t < 0 or t == s:
                continue
            if close[k, b_s] or close[kn, p_s] or self.signal_final[kn, t] == DEGENERATE:
                continue                                  # gauge territory: hands off
            src[s] = t
        if not src:
            return None
        # the seed edge must lie on a closed cycle of src that stays within src
        for seed in src:
            order = [seed]
            cur = src[seed]
            ok = True
            while cur != seed:
                if cur not in src or cur in order:
                    ok = False
                    break
                order.append(cur)
                cur = src[cur]
            if ok and len(order) >= 2:
                return tuple(order)
        return None

    def _repair_dp_cycles(self, max_rounds: int = 32) -> int:
        '''
        Post-loop repair of the energy-QUIET permutation cycles the pairwise
        correctors cannot unwind and ``_repair_certain_links`` never sees.

        A label frontier collision inside a near-parallel band group leaves a
        k-region where the slots hold a cyclically PERMUTED set of bands
        (phosphorene: 11<->12 2-cycles, the 15/16/17 and 17/18/19 3-cycles).
        The energy is flat across the group so every gate passes (zero walls)
        and the swapped interior validates -- so ``_repair_certain_links`` (its
        trigger is energy-keyed: unattributed / NOT-MIS / gross jump) never
        looks at it. The region correctors DO see the broken dp seams but are
        strictly PAIRWISE: the first swap of a 3-cycle relocates a transposition
        rather than removing it, so it fails their strict/halving acceptance and
        is rejected. The cycle survives as a validated band carrying an invisible
        wavefunction discontinuity.

        This pass closes that gap. Seeded on ``_dp_refuted_seam_edges``, at each
        seam it builds the slot permutation mandated by the CERTAIN (dp >= 0.9)
        successors (``_dp_cycle_at``), finds a closed cycle, grows its cyclic
        pattern region (``_pattern_region_cycle``) and applies the atomic L-way
        rotation (``_apply_slot_rotation``). The rotation is COMMITTED only when
        it strictly decreases the exact global refuted census with the gross-jump
        wall count not increasing -- measured by trial-apply / revert, so the
        guard is exact, not heuristic. Three guards keep the genuine crossing
        corridors untouched (the failure mode that made the energy-quiet trigger
        exclusion necessary, tested there as FAIL 11 -> 72): only CLOSED cycles
        with all-certain mandates are candidates, near-degenerate cells are
        exempt, and a move that does not strictly reduce refuted (a corridor
        relabel merely relocates it) or that raises a wall is reverted.

        Runs after the pairwise passes (they clear the 2-sided defects first) and
        before ``_repair_certain_links``. Bijection preserved by the per-row
        rotation; changed cells are re-graded on the final canvas. Returns the
        number of exchanged cells.
        '''
        n_ref0 = len(self._dp_refuted_seam_edges())
        if n_ref0 == 0:
            return 0
        changed_cells: list = []
        n_cyc = 0
        for _ in range(max_rounds):
            seeds = self._dp_refuted_seam_edges()
            if not seeds:
                break
            ref0 = len(seeds)
            wall0 = len(self._slot_wall_edges())
            accepted = None
            tried: set = set()
            for _v, k, kn, _s in seeds:
                for a, b in ((k, kn), (kn, k)):
                    order = self._dp_cycle_at(a, b)
                    if order is None:
                        continue
                    key = (b, order)
                    if key in tried:
                        continue
                    tried.add(key)
                    R = self._pattern_region_cycle(b, order)
                    if len(R) < 1:
                        continue
                    # trial-apply, measure exact global census, revert if worse.
                    # Snapshot every array _apply_slot_rotation touches (bands,
                    # both signals, and the audit masks) for the cycle's slots.
                    arrays = [self.bands_final, self.signal_final,
                              self.correct_signalfinal]
                    arrays += [m for m in (getattr(self, 'forced_mask', None),
                                           getattr(self, 'repaired_mask', None))
                               if m is not None]
                    snap = [(arr, {s: arr[:, s].copy() for s in order})
                            for arr in arrays]
                    cells = self._apply_slot_rotation(R, order)
                    ref1 = len(self._dp_refuted_seam_edges())
                    wall1 = len(self._slot_wall_edges())
                    if ref1 < ref0 and wall1 <= wall0:
                        accepted = (R, order, cells, ref0, ref1, wall0, wall1)
                        break
                    for arr, cols in snap:                      # revert every array
                        for s, col in cols.items():
                            arr[:, s] = col
                if accepted:
                    break
            if accepted is None:
                break
            R, order, cells, ref0v, ref1v, wall0v, wall1v = accepted
            self._regrade_cells(cells)
            changed_cells.extend(cells)
            n_cyc += 1
            self.logger.info(f'{BODY_INDENT}dp-cycle repair: slots {order} rotated over '
                             f'{len(R)} k-point(s) -- refuted {ref0v}->{ref1v}, '
                             f'walls {wall0v}->{wall1v}')
        if not changed_cells:
            self.logger.info(f'{BODY_INDENT}dp-cycle repair: no improving rotation '
                             f'({n_ref0} refuted seam edge(s) on the final canvas)')
            return 0
        n_k = len(set(c[0] for c in changed_cells))
        self.logger.info(f'{BODY_INDENT}dp-cycle repair: {n_cyc} cycle(s) rotated over '
                         f'{n_k} k-point(s); refuted {n_ref0} -> '
                         f'{len(self._dp_refuted_seam_edges())}')
        return len(changed_cells)

    def _repair_dp_swaps(self, max_rounds: int = 32) -> int:
        '''
        Post-loop repair of the label swaps the energy machinery cannot see.

        A swapped pair of near-parallel bands (e.g. an SOC pair split by a few mRy)
        costs almost nothing in energy -- every energy gate passes -- but the
        wavefunction evidence is decisive: the swapped cell has dot product ~0 to its
        own slot at every correctly-attributed neighbour while the exchanged
        assignment scores ~1. This pass takes every attributed cell with ZERO
        dp-supported directions and exchanges its content with the row's best-fitting
        slot, accepting a swap only when it STRICTLY increases the two cells'
        combined number of dp-supported directions. Genuine crossings are safe: there
        the current (band-following) attribution is the dp-supported one, so any swap
        strictly decreases support and is rejected. Swap filaments unwind end-first
        over the rounds (fixing an end removes its neighbour's only supporter, which
        exposes that neighbour as the next round's candidate).

        Runs after the final correct_signal, so the canvas is complete; only values
        inside a row are exchanged, so the per-k bijection is preserved. At the end
        the energy-continuity signal of the still-failed cells is re-evaluated on the
        final canvas, so cells fixed here, cells whose neighbourhood was fixed here,
        and the bijection gap fills (flagged before they were filled) are graded on
        what is actually shipped rather than on the state they were flagged in.

        Returns the number of swaps applied.
        '''
        OTHER = 3
        bf = self.bands_final
        neigh = np.asarray(self.neighbors)
        n_dir = neigh.shape[1]

        def n_supported(k: int, s: int, b: int) -> int:
            # dp-supported directions of band b sitting in slot s at k, on the
            # current (possibly mid-round) canvas.
            good = 0
            for d in range(n_dir):
                kn = int(neigh[k, d])
                if kn < 0:
                    continue
                b2 = int(bf[kn, s])
                if b2 < 0:
                    continue
                if self.connections[k, d, b, b2] >= DP_SUP_TOL:
                    good += 1
            return good

        swaps = 0
        swapped_cells = []
        for _ in range(max_rounds):
            sup, tot = self._dp_support_counts()
            cand = np.argwhere((bf >= 0) & (sup == 0) & (tot > 0))
            round_swaps = 0
            for k, s in cand:
                k, s = int(k), int(s)
                b_s = int(bf[k, s])
                if b_s < 0 or n_supported(k, s, b_s) > 0:
                    continue                            # already touched earlier this round
                best_gain, best_t = 0, -1
                for t in range(bf.shape[1]):
                    if t == s:
                        continue
                    b_t = int(bf[k, t])
                    if b_t < 0:
                        continue
                    cur = n_supported(k, t, b_t)        # slot s contributes 0 (candidate)
                    new = n_supported(k, s, b_t) + n_supported(k, t, b_s)
                    if new - cur > best_gain:
                        best_gain, best_t = new - cur, t
                if best_t >= 0:
                    bf[k, s], bf[k, best_t] = int(bf[k, best_t]), b_s
                    swapped_cells.append((k, s, best_t))
                    round_swaps += 1
            swaps += round_swaps
            if round_swaps == 0:
                break
        if swaps:
            n_k = len(set(c[0] for c in swapped_cells))
            self.logger.info(f'{BODY_INDENT}DP-evidence swap repair: exchanged {swaps} slot '
                             f'pair(s) at {n_k} k-point(s)')
            for k, s, t in swapped_cells:
                self.logger.debug(f'{BODY_INDENT}  [dp-swap] k={k}: slot {s} <-> slot {t} '
                                  f'(now bands {int(bf[k, s])}/{int(bf[k, t])})')

        # Re-grade the failed cells (NOT/MIS/OTH) on the final canvas. DEGENERATE
        # keeps its marker and FORCED_CONTINUITY cells are not in the refresh set.
        refreshed = 0
        upgraded = 0
        failed_now = np.isin(self.correct_signalfinal, (NOT_SOLVED, MISTAKE, OTHER)) & (bf >= 0)
        for k, s in np.argwhere(failed_now):
            k, s = int(k), int(s)
            if self.signal_final[k, s] == DEGENERATE:
                continue
            sig, _ = evaluate_point(self.dimensions, k, s, self.kpoints_index,
                                    self.matrix, bf, self.eigenvalues,
                                    accept_E=self.accept_E, disp_scale=self.disp_scale,
                                    local_gap=self.local_gap, band_disp=self.band_disp)
            if sig != self.correct_signalfinal[k, s]:
                if sig > self.correct_signalfinal[k, s]:
                    upgraded += 1
                self.correct_signalfinal[k, s] = sig
                refreshed += 1
        if refreshed:
            self.logger.info(f'{BODY_INDENT}Failed-cell re-grade on the final canvas: '
                             f'{refreshed} signal(s) changed ({upgraded} upgraded)')
        return swaps

    def _flag_dp_unsupported(self) -> int:
        '''
        Final audit: demote to MISTAKE every attributed cell whose assignment has
        ZERO dp-supported directions and no degeneracy excuse.

        The energy validation cannot see a label swap between near-parallel bands (a
        few-mHa error passes every gate), so a cell can be energy-CORRECT while the
        wavefunction evidence flatly contradicts it. Cells inside a near-degenerate
        group (adjacent gap < ``degen_ethr``) are exempt: there the individual label
        is gauge and a low diagonal dot product is expected (basis-rotation
        territory, not a mis-attribution). Force-filled cells keep FORCED_CONTINUITY
        (the report already audits those as benign/suspect). The attribution itself
        is NOT modified -- this only makes the report honest about it.

        Returns the number of demoted cells.
        '''
        OTHER = 3
        CORRECT_C = 4                                               # energy-continuity scale
        bf = self.bands_final
        sup, tot = self._dp_support_counts()
        close = self._near_degenerate_mask()                        # (k, band) near-degenerate w/ a neighbour band
        att = bf >= 0
        in_degen = np.zeros(bf.shape, bool)
        in_degen[att] = close[np.where(att)[0], bf[att]]
        csf = self.correct_signalfinal
        target = (att & (tot > 0) & (sup == 0) & ~in_degen &
                  ((csf == OTHER) | (csf == CORRECT_C)))
        n = int(target.sum())
        if n:
            per_band = {int(s): np.where(target[:, s])[0]
                        for s in range(bf.shape[1]) if target[:, s].any()}
            summary = ', '.join(f'band {s}: {len(ks)}' for s, ks in per_band.items())
            self.logger.info(f'{BODY_INDENT}DP-support audit: {n} attributed point(s) have zero '
                             f'dot-product support in their slot -- demoted to MISTAKE ({summary})')
            for s, ks in per_band.items():
                self.logger.debug(f'{BODY_INDENT}  [dp-audit] band {s}: k = {list(ks[:50])}'
                                  + (' ...' if len(ks) > 50 else ''))
            csf[target] = MISTAKE
        else:
            self.logger.info(f'{BODY_INDENT}DP-support audit: every attributed point has '
                             f'dot-product support (no silent swaps)')
        return n

    def _ensure_connectivity(self) -> None:
        '''
        Connectivity guard (anti-shatter safety net). No node may be left as a degree-0
        singleton: thousands of isolated nodes would each become their own community and
        feed the O(samples^2) assignment loop in get_components (count = len(samples)**2)
        -- the ~2.4e8-evaluation blow-up seen on MoS2 (~11 h for one step). The
        error-region dot-product rebuild normally reconnects the bad nodes, but it only
        keeps edges whose overlap clears `tol`; a high `tol`, or a bad point whose every
        neighbour is also bad, can still leave slots edgeless. Here every remaining orphan
        is wired to its single best-overlap same-band k-space neighbour, ignoring `tol`.
        One best-overlap edge per orphan is the most defensible link: it bounds the
        connected-component count by construction, cannot create dense cross-band merges,
        and is a weak link the validation can still override on the next pass.
        '''
        isolated = [n for n in self.GRAPH.nodes if self.GRAPH.degree(n) == 0]
        if not isolated:
            return
        added = 0
        for node in isolated:
            bn = node // self.nks                       # band slot of this node
            k1 = node % self.nks
            best_w = -2.0
            best_pn = None
            for i_neig, kn in enumerate(self.neighbors[k1]):
                if kn == -1:
                    continue                            # neighbour outside the BZ grid
                conn = float(self.connections[k1, i_neig, bn, bn])      # <psi_{k,bn}|psi_{kn,bn}>
                w = 1 - 2 / np.pi * np.arccos(np.clip(conn, -1.0, 1.0))  # same metric as the edge build
                if w > best_w:
                    best_w = w
                    best_pn = kn + bn * self.nks
            if best_pn is not None and best_pn != node:
                if self.GRAPH.has_edge(node, best_pn):
                    self.GRAPH[node][best_pn]['weight'] += 1
                else:
                    self.GRAPH.add_edge(node, best_pn, weight=max(best_w, 1e-6))
                added += 1
        self.logger.info(f'{BODY_INDENT}Connectivity guard: linked {added} orphan node(s) '
                         f'of {len(isolated)} isolated to best-overlap neighbour')

    def report(self):
        self.final_report += '*************************************************************************************************\n'
        self.final_report += '|                                       SOLUTION REPORT                                         |\n'
        self.final_report += '*************************************************************************************************\n\n'

        # Per-band counts (validate scale: NOT, MIS, DEG, OTH, COR, FOR, Score)
        # are still needed for the usable/completed-band logic; we read them but
        # render a compact, actionable table instead of the full signal table.
        _, report_a2 = self.print_report(self.correct_signalfinal, 'Final Report', show=False, header_text=VALIDATE_RESULT_HEADER)

        ###########################################################################
        # Compact per-band table: only the actionable counts + a plain status.
        ###########################################################################
        TOL_CLEAN = 0.97            # at/above this (and nothing flagged) -> CLEAN; below -> USABLE
        DEGEN_GAP = 0.010           # Ry (~0.14 eV); a force-fill whose slot sits within this
                                    # gap of an adjacent band is a degenerate-doublet
                                    # relabeling (gauge), not a genuine mis-assignment
        FORCED_FAIL_FRAC = 0.05     # > this fraction of suspect force-fills -> FAIL
        bands_attention = []

        def _forced_split(i):
            # Split slot i's force-filled points into those ACROSS a real energy
            # gap ("suspect") and those inside a near-degenerate doublet
            # ("benign", gauge-level relabelings). Returns (suspect_count,
            # benign_ks): benign in-doublet force-fills do not count against the
            # band and are excluded from its score.
            ncols = self.bands_final.shape[1]
            ks = np.where(self.correct_signalfinal[:, i] == FORCED_CONTINUITY)[0]
            suspect = 0
            benign_ks = []
            for k in ks:
                b = self.bands_final[k, i]
                if b < 0:
                    suspect += 1
                    continue
                e = self.eigenvalues[k, b]
                gap = np.inf
                for j in (i - 1, i + 1):
                    if 0 <= j < ncols:
                        bj = self.bands_final[k, j]
                        if bj >= 0:
                            gap = min(gap, abs(e - self.eigenvalues[k, bj]))
                if gap >= DEGEN_GAP:
                    suspect += 1
                else:
                    benign_ks.append(k)
            return int(suspect), np.array(benign_ks, dtype=int)

        def _score_excl_benign(i, benign_ks):
            # Honest band score: the mean, over slot i's genuine points, of the
            # dot product to its attributed neighbours -- dropping any edge
            # incident to a benign (in-doublet) force-fill so a gauge-level
            # relabeling cannot deflate it. Mirrors the in-loop scoring loop but
            # averages over comparable points only (not over nks, which counted
            # force-filled/unattributed points as zeros). Falls back to the
            # in-loop score if the slot has no comparable points left.
            benign = np.zeros(self.nks, dtype=bool)
            benign[benign_ks] = True
            all_neigs = np.arange(self.number_neighbors)
            total = 0.0
            count = 0
            for k in range(self.nks):
                if benign[k]:
                    continue
                bk = self.bands_final[k, i]
                if bk < 0:
                    continue
                kneigs = self.neighbors[k]
                flag = kneigs != -1
                i_neigs = all_neigs[flag]
                kn = kneigs[flag]
                keep = (~benign[kn]) & (self.bands_final[kn, i] >= 0)
                i_neigs = i_neigs[keep]
                kn = kn[keep]
                if len(kn) == 0:
                    continue
                bn_neighs = self.bands_final[kn, i]
                dps = self.connections[np.repeat(k, len(kn)), i_neigs,
                                       np.repeat(bk, len(kn)), bn_neighs]
                total += float(np.mean(dps))
                count += 1
            return total / count if count else self.final_score[i]

        ###########################################################################
        # SILENT discontinuities: validated cells (OTHER/CORRECT) sitting on a
        # gross energy jump (> accept_E) to a same-slot neighbour. They pass the
        # per-direction validation (>= 3 of 4 directions agree) yet are exactly
        # the cliffs visible in a band plot -- a band carrying them must not be
        # called USABLE on the strength of its Failed column alone.
        ###########################################################################
        _jumps = self._nn_energy_jump_map()
        _silent_mask = ((self.bands_final >= 0) &
                        (np.nan_to_num(_jumps, nan=0.0) > self.accept_E) &
                        np.isin(self.correct_signalfinal, (3, 4)))          # OTHER, CORRECT
        silent_counts = _silent_mask.sum(axis=0)
        # Cliff column: per-slot count of surviving sub-accept_E cliff edges
        # (raw-band switches above the LOCAL continuity gate), populated by the
        # end-of-solve _verify_no_cliffs invariant. Fall back to a fresh census if
        # report() is called on a canvas that never ran the invariant.
        cliff_counts = getattr(self, 'cliff_counts', None)
        if cliff_counts is None:
            cliff_counts = np.zeros(self.bands_final.shape[1], dtype=int)
            for _j, _k, _kn, _s in self._cliff_seam_edges():
                cliff_counts[int(_s)] += 1
        # dpBreak column: per-slot count of surviving wavefunction-broken seam
        # edges (raw-band switches against a CERTAIN dp link, the energy-quiet
        # permutation cycles _repair_dp_cycles could not unwind), populated by
        # the end-of-solve _verify_no_dp_cycles invariant. Fall back to a fresh
        # census if report() is called on a canvas that never ran the invariant.
        refuted_counts = getattr(self, 'refuted_counts', None)
        if refuted_counts is None:
            refuted_counts = np.zeros(self.bands_final.shape[1], dtype=int)
            for _v, _k, _kn, _s in self._dp_refuted_seam_edges():
                refuted_counts[int(_s)] += 1

        header_line = ' Band | Failed | Degen | Benign | Susp | Silent | Cliff | dpBrk | Score | Status'
        table = f'\n{TITLE_INDENT}====== Final Report ======\n'
        table += _format_legend(['Failed', 'Degen', 'Benign', 'Susp', 'Silent', 'Cliff', 'dpBreak', 'Score', 'Status'])
        table += f'\n{BODY_INDENT}{header_line}'
        table += f'\n{BODY_INDENT}' + '-' * len(header_line)
        band_grade = []
        for i in range(self.nbnd):
            failed = int(report_a2[i, NOT_SOLVED] + report_a2[i, MISTAKE])
            degen = int(report_a2[i, DEGENERATE])
            forced = int(report_a2[i, FORCED_CONTINUITY])
            silent = int(silent_counts[i]) if i < len(silent_counts) else 0
            cliff = int(cliff_counts[i]) if i < len(cliff_counts) else 0
            dpbrk = int(refuted_counts[i]) if i < len(refuted_counts) else 0
            if forced:
                suspect, benign_ks = _forced_split(i)
            else:
                suspect, benign_ks = 0, np.empty(0, dtype=int)
            benign = forced - suspect
            # Honest score: recompute over the band's genuine points, excluding
            # benign (in-doublet) force-fills, so a gauge-level relabeling cannot
            # deflate it. Overwrite the in-loop score (which divided by nks and so
            # counted force-filled/unattributed points as zeros) so the table and
            # final_score.npy report the same trustworthy number. Bands with no
            # benign fills keep their in-loop score untouched.
            if benign_ks.size:
                score = _score_excl_benign(i, benign_ks)
                self.final_score[i] = score
            else:
                score = self.final_score[i]
            # Status is driven by what is actually flagged in the signal columns.
            # A band fails only on a genuine energy break (NOT/MIS) or a burst of
            # force-fills across a real gap. A merely low score with nothing
            # genuinely flagged is a usable (typically entangled conduction) band,
            # reported as USABLE -- never FAIL on score alone. A suspect force-fill
            # (across a real gap) asks for verification; a benign force-fill is
            # gauge-level and does not affect usability.
            if failed > 0 or suspect + silent + cliff + dpbrk > FORCED_FAIL_FRAC * self.nks:
                status = 'FAIL'
            elif suspect > 0 or silent > 0 or cliff > 0 or dpbrk > 0:
                status = 'CHECK'                # suspect fill / silent cliff / sub-gate cliff / dp break -> verify
            elif degen > 0:
                status = 'ROTATE'               # usable after basis rotation
            elif forced == 0 and score >= TOL_CLEAN:
                status = 'CLEAN'                # pristine, nothing flagged
            else:
                status = 'USABLE'               # nothing flagged (benign fills and/or score < clean)
            band_grade.append(status)
            table += (f'\n{BODY_INDENT} {i+self.min_band:<4d} | {failed:^6d} | {degen:^5d} | '
                      f'{benign:^6d} | {suspect:^4d} | {silent:^6d} | {cliff:^5d} | {dpbrk:^5d} | '
                      f'{score:>5.2f} | {status}')
            if status in ('FAIL', 'CHECK'):
                bands_attention.append((i, failed, degen, benign, suspect, silent, cliff, dpbrk, status))
        table += '\n'
        # Expose the per-band grade (indexed by slot) so downstream consumers --
        # e.g. dp_interp_mask(), which must drop FAIL bands -- can read it without
        # re-parsing the report string.
        self.band_grade = band_grade
        self.final_report += table

        ###########################################################################
        # The points that failed and why.
        ###########################################################################
        if bands_attention:
            self.final_report += f'\n{BODY_INDENT}Bands needing attention:'
            for i, failed, degen, benign, suspect, silent, cliff, dpbrk, status in bands_attention:
                reasons = []
                if failed > 0:
                    reasons.append(f'{failed} failed (energy discontinuity / low overlap)')
                if suspect > 0:
                    reasons.append(f'{suspect} of {benign+suspect} force-fill(s) across a real energy gap')
                if silent > 0:
                    reasons.append(f'{silent} silent cliff(s) (validated cells on a jump > accept_E)')
                if cliff > 0:
                    reasons.append(f'{cliff} sub-gate cliff edge(s) (raw-band switch above the local continuity gate)')
                if dpbrk > 0:
                    reasons.append(f'{dpbrk} dp-broken seam edge(s) (raw-band switch against a certain dp link)')
                reason = ' + '.join(reasons) if reasons else f'score {self.final_score[i]:.2f}'
                verdict = 'NOT usable' if status == 'FAIL' else 'verify before use'
                self.final_report += (f'\n{BODY_INDENT}  Band {i+self.min_band:>3d} : {reason} - {verdict}')
            self.final_report += '\n'

        p_report, problems = self.solved_problems_info

        self.final_report += '\n\t{:-^30}\n\t{: <30}\n\t{:-^30}\n'.format('', 'Summary:', '')
        self.final_report += p_report

        point2k_bn = lambda p: (p % self.nks, p // self.nks)

        for d1, d2 in problems:
            k1, bn1 = point2k_bn(d1)
            k2, bn2 = point2k_bn(d2)
            Bk1 = self.bands_final[k1] == bn1                               # Find in which  band the k-point was attributed
            Bk2 = self.bands_final[k2] == bn2                               # Find in which  band the k-point was attributed
            bn1 = np.argmax(Bk1) if np.sum(Bk1) != 0 else bn1               # Final band
            bn2 = np.argmax(Bk2) if np.sum(Bk2) != 0 else bn2               # Final band
        
            self.final_report += f'\n\t\t\tK-point: {k1} bands: {bn1+self.min_band}, {bn2+self.min_band}' # Report
        
        if len(problems) > 0:
            self.final_report += f'\n\t\t  These points were corrected.'
        
        degenerates = []

        for i, (d1, d2) in enumerate(self.degenerates):
            k1, bn1 = point2k_bn(d1)
            k2, bn2 = point2k_bn(d2)
            Bk1 = self.bands_final[k1] == bn1                               # Find in which  band the k-point was attributed
            Bk2 = self.bands_final[k2] == bn2                               # Find in which  band the k-point was attributed
            bn1 = np.argmax(Bk1) if np.sum(Bk1) != 0 else bn1               # Final band
            bn2 = np.argmax(Bk2) if np.sum(Bk2) != 0 else bn2               # Final band
            bn1 = self.bands_final[k1, bn1]
            bn2 = self.bands_final[k2, bn2]

            if self.signal_final[k1, bn1] == DEGENERATE:
                degenerates.append([k1, bn1, bn2])

        if len(degenerates) > 0:
            n = len(degenerates)
            self.final_report += f'\n\n\t\tNumber of degenerate points: {n}\n'
            for k1, bn1, bn2 in degenerates:
                self.final_report += f'\n\t\t\t* K-point: {k1} Bands: {bn1+self.min_band}, {bn2+self.min_band}'
            self.final_report += f'\n\n\t\t  ' + textwrap.fill(
                'Degenerate points refer to instances where a point shares the same numerical energy value with another k-point, and the dot product with its neighboring points falls within the range of 0.5 to 0.8. \n\tAs a result, it is necessary for the rotation basis program to be executed for these points.',
                width=110,
                subsequent_indent='\t\t  '
            ) + '\n'

        if len(self.degenerate_final) > 0:
            n = len(self.degenerate_final)
            self.final_report += f'\n\n\t\t  Found {n} points with one or more neighbor with a dot-product between 0.5 and 0.8.'
            self.final_report += f'\n\t\t  May not be degenerate points under energy criteria.'
            self.final_report += f'\n\t\t  So they were not signaled and no corrections were applied.'
            self.final_report += f'\n\t\t  However, they are saved in the degeneratefinal.npy file, in case they need analysis.'
            if self.logger.level == logging.DEBUG:
                self.final_report += '\n\t\t   Points:'
                i_sort = np.argsort([k for k, _, _ in self.degenerate_final])
                for k, bn1, bn2 in self.degenerate_final[i_sort]:
                    self.final_report += f'\n\t\t  * K-point: {k} \tBands: {bn1+self.min_band}, {bn2+self.min_band}'

        # Bands grouped by usability. A force-fill inside a near-degenerate
        # doublet (USABLE/ROTATE) is a gauge-level relabeling: it does NOT make a
        # band unusable, nor does it hide the bands above it. Only a genuine
        # energy break (FAIL) breaks the contiguous run from the bottom.
        usable = [i for i in range(self.nbnd) if band_grade[i] in ('CLEAN', 'USABLE', 'ROTATE')]
        check  = [i for i in range(self.nbnd) if band_grade[i] == 'CHECK']
        fail   = [i for i in range(self.nbnd) if band_grade[i] == 'FAIL']

        def _fmt_ranges(bands):
            if not bands:
                return '(none)'
            bands = sorted(bands)
            out, start, prev = [], bands[0], bands[0]
            for b in bands[1:]:
                if b == prev + 1:
                    prev = b
                    continue
                out.append(f'{start+self.min_band}-{prev+self.min_band}' if prev > start
                           else f'{start+self.min_band}')
                start = prev = b
            out.append(f'{start+self.min_band}-{prev+self.min_band}' if prev > start
                       else f'{start+self.min_band}')
            return ', '.join(out)

        self.completed_bands = np.array(usable, dtype=int)

        # Contiguous run of usable bands from the bottom; broken only by a FAIL.
        max_solved = 0
        for i in range(self.nbnd):
            if band_grade[i] == 'FAIL':
                break
            max_solved += 1
        self.max_solved = max_solved

        self.final_report += f'\n\n\n\tUsable bands: {_fmt_ranges(usable)}'
        self.final_report += (f'\n\t\t  {len(usable)} band(s) free of genuine errors '
                              f'(saved in `completed_bands.npy`).')
        self.final_report += (f'\n\t\t  Force-fills inside degenerate doublets are gauge-level relabelings; '
                              f'they are\n\t\t  reported in the table but do not affect usability.')
        if check:
            self.final_report += f'\n\n\tNeeds verification (CHECK): {_fmt_ranges(check)}'
            self.final_report += (f'\n\t\t  A force-fill across a real energy gap; '
                                  f'usable only after a manual check.')
        if fail:
            self.final_report += f'\n\n\tNot usable (FAIL): {_fmt_ranges(fail)}'
            self.final_report += f'\n\t\t  Genuine energy discontinuity / low overlap.'

        n_repaired = int(np.sum(self.repaired_mask)) if getattr(self, 'repaired_mask', None) is not None else 0
        if n_repaired > 0:
            self.final_report += f'\n\n\t{n_repaired} point(s) were reassigned by the a-posteriori energy-continuity repair'
            self.final_report += f'\n\tpass (extrapolated from trusted neighbouring trajectories). They were re-validated'
            self.final_report += f'\n\tby the energy-continuity criteria like any other point.'

        n_forced = int(np.sum(self.forced_mask)) if getattr(self, 'forced_mask', None) is not None else 0
        if n_forced > 0:
            self.final_report += f'\n\n\t{n_forced} point(s) were force-filled (FOR) by the completeness pass to guarantee a'
            self.final_report += f'\n\tband attribution everywhere. These are not genuine solves: they were assigned the'
            self.final_report += f'\n\tclosest available band in energy. A force-fill that lands inside a near-degenerate'
            self.final_report += f'\n\tdoublet is a gauge-level relabeling and keeps the band usable; only force-fills'
            self.final_report += f'\n\tacross a real energy gap (counted as "suspect") downgrade a band to CHECK/FAIL.'

        if len(degenerates) > 0:
            # contiguous run of usable bands from the bottom (the set that can be
            # safely fed to basis rotation)
            n_contig = 0
            for i in range(self.nbnd):
                if band_grade[i] not in ('CLEAN', 'USABLE', 'ROTATE'):
                    break
                n_contig += 1
            n_contig = max(n_contig, 1)
            self.final_report += f'\n\n\tTo use the program basis rotation, you must run the program for the first {n_contig} bands.'
            self.final_report += f'\n\n\t\t `$ berry basis {n_contig - 1}`'

        self.final_report += '\n\n*************************************************************************************************\n'

        return self.final_report

    def solve(self, step: float=0.5, alpha : float=1.0, min_alpha: float=0, alpha_patience: int=3) -> None:
        '''
        This method is the main algorithm which iterates between solutions
        trying to find the best result for the material.

        Parameters
            step : float
                It is the iteration value which is used to relax the alpha value.
                (default 0.5, i.e. the alpha sweep is 1.0 -> 0.5 -> 0.0)
            alpha : float
                Initial weight of the dot-product term in the sample-attachment
                score: score = alpha*<i|j> + (1-alpha)*f(E). The sweep descends
                from here by ``step`` down to ``min_alpha``.
                (default 1.0)
            min_alpha : float
                The minimum alpha.
                (default 0)
            alpha_patience : int
                Consecutive non-improving iterations (no new not-solved low) to
                tolerate at one alpha before descending to the next.
                (default 3)
        '''
        ###########################################################################
        # Initial preparation of data structures
        # The previous and best result are stored
        ###########################################################################
        # The main loop relies on `alpha -= step` to terminate (alpha sweeps down to
        # min_alpha).  A non-positive step would never lower alpha, so the loop could
        # only stop if the result happened to stabilise — otherwise it spins forever (H2).
        if step <= 0:
            raise ValueError(f"solve() requires step > 0 to converge (the alpha sweep would never advance); got step={step}.")
        self.alpha = alpha # The initial alpha is 1.0: alpha*<i|j> + (1-alpha)*f(E)
        COUNT = 0     # Counter iteration
        self.final_report = ''
        self.bands_final_prev = np.copy(self.bands_final)
        self.best_bands_final = np.copy(self.bands_final)
        self.best_score = np.zeros(self.total_bands, dtype=float)
        self.final_score = np.zeros(self.total_bands, dtype=float)
        self.signal_final = np.zeros((self.nks, self.total_bands), dtype=int)
        self.correct_signalfinal_best = np.full(self.signal_final.shape, -1, int)
        self.degenerate_best = None
        max_solved = 0  # The maximum number of solved bands

        self.repeat_communities = False
        # Convergence-paced alpha sweep: hold the current alpha while not_solved keeps
        # setting new lows here, and descend only after it stalls for `alpha_patience`
        # consecutive iterations. Any new best resets the stall counter, so brief
        # oscillations are absorbed -- alpha descends only when improvement at the
        # current alpha has genuinely stopped persisting.
        self.alpha_best_ns = np.inf      # lowest not_solved seen at the current alpha
        self.alpha_stall = 0             # consecutive iterations with no new best
        self.total_not_solved = np.inf   # set by correct_signal() each iteration

        ###########################################################################
        # Algorithm
        ###########################################################################
        # The 1e-9 slack keeps float accumulation in `alpha -= step` (e.g. ten
        # 0.1 steps summing to 0.9999999999) from dropping the final iteration.
        while self.alpha >= min_alpha - 1e-9:
            COUNT += 1
            start_time = time.time()
            self.logger.info()
            self.logger.info(f'\n\n\t* Iteration: {COUNT} - Clustering samples for Alpha: {self.alpha:.4f}')
            self.get_components(alpha=self.alpha, compute_communities= COUNT == 1 or self.repeat_communities)                    # Obtain components from a Graph

            self.logger.info('\n\t\tCalculating output')        
            self.obtain_output()                            # Compute the result
            self.print_report(self.signal_final, f'Report Number: {COUNT} considering dot-product information', header_text=EVALUATE_RESULT_HEADER)                           # Print result

            self.logger.info('\n\t\tValidating result using energy continuity criteria')     
            self.correct_signal()                           # Evaluate the energy continuity and perform a new Graph
            self._log_fE_outcome()                          # f(E) diagnostics, Tier 4: outcome over attached points
            self.print_report(self.correct_signalfinal, f'Validation Report Number: {COUNT} considering  energy continuity criteria', header_text=VALIDATE_RESULT_HEADER)     # Print result

            # This iteration's attempt, captured BEFORE the best-result revert below may
            # overwrite self.total_not_solved (the else branch re-runs correct_signal on
            # the reverted best). The convergence test uses the attempt's quality.
            attempt_not_solved = self.total_not_solved

            # An attempt identical to the previous one is a fixed point of the
            # (deterministic) region-growing partition at this alpha: repeating it can
            # change nothing. It used to end the whole solve here -- which under the
            # deterministic partitioner fired before the stall patience could ever
            # elapse and left the alpha sweep dead. Now it fast-forwards the alpha
            # descent below, so the sweep always completes down to min_alpha.
            result_stable = COUNT > 1 and np.array_equal(self.bands_final_prev, self.bands_final)
            self.bands_final_prev = np.copy(self.bands_final)

            # Verify and store the best result
            # To be a better result it has to be better score and all k points attributed for all the first max_solved bands
            solved = 0
            OTHER = 3

            # NOT_SOLVED-only counts (unlike self.total_not_solved from
            # correct_signal, which counts NOT + MIS) -- named apart on purpose;
            # the best-selection metric is kept exactly as tuned.
            n_not_best = np.sum(self.correct_signalfinal_best == NOT_SOLVED)
            n_not = np.sum(self.correct_signalfinal == NOT_SOLVED)

            total_best_score = np.sum(self.best_score)
            total_score = np.sum(self.final_score)

            first_max_bands_best_score = np.sum(self.best_score[:max_solved])
            first_max_bands_score = np.sum(self.final_score[:max_solved])
            for bn, score in enumerate(self.final_score):
                best_score = self.best_score[bn]
                mistake_best = np.sum(self.correct_signalfinal_best[:, bn] == MISTAKE)
                mistake = np.sum(self.correct_signalfinal[:, bn] == MISTAKE)
                other_best = np.sum(self.correct_signalfinal_best[:, bn] == OTHER)
                other = np.sum(self.correct_signalfinal[:, bn] == OTHER)
                not_solved = np.sum(self.correct_signalfinal[:, bn] == NOT_SOLVED)
                not_solved_best = np.sum(self.correct_signalfinal_best[:, bn] == NOT_SOLVED)

                solved_flag = not_solved == 0 
                solved_flag_points = True
                if COUNT > 1:
                    solved_flag_points = mistake < mistake_best and other < other_best or score >= best_score and mistake <= mistake_best and other <= other_best
                    solved_flag_points = solved_flag_points or (score > 0.99 and mistake + other < mistake_best + other_best)

                solved_flag = (solved_flag or not_solved < not_solved_best) and solved_flag_points


                self.logger.debug(f'\n\t\t\tPrev best result for band {bn}: {best_score} mistakes: {mistake_best} other: {other_best}')
                self.logger.debug(f'\n\t\t\tNew best result for band {bn}: {score} mistakes: {mistake} other: {other}. Solved: {solved_flag}')
                if score != 0 and not_solved == 0 and (score > best_score or solved_flag):
                    solved += 1
                else:
                    break
            
            self.logger.info(f'\n\t\t Iteration: {COUNT} - Clustered bands: {solved} - Max clustered bands: {max_solved}')
            self.logger.info('\t\t\t' + tempo(start_time, time.time(), name='iteration'))
                             
            n_bands = len(self.final_score)
            total_solved_flag = first_max_bands_score >= first_max_bands_best_score and total_score > total_best_score and n_not < n_not_best
            total_solved_flag = total_solved_flag or (total_score/n_bands > 0.9 and total_best_score/n_bands < 0.9)
            # A drastically cleaner attempt (under half the reigning best's not-solved
            # count) wins even when its contiguous solved-band count is lower: the
            # per-band chain above breaks at the first non-improving band, which let a
            # 10-band draw with 4566 not-solved hold the title against a 734-not-solved
            # one (phosphorene, Jul 10). The score guard keeps a low-not-solved but
            # low-overlap solution from displacing a genuinely solved one.
            much_cleaner = (n_not < 0.5 * n_not_best
                            and total_score >= 0.9 * total_best_score)
            if total_solved_flag or much_cleaner or solved >= max_solved or COUNT == 1:
                self.best_bands_final = np.copy(self.bands_final)
                self.best_score = np.copy(self.final_score)
                self.best_signal_final = np.copy(self.signal_final)
                self.degenerate_best = np.copy(self.degenerate_final)
                self.correct_signalfinal_best = np.copy(self.correct_signalfinal)
                max_solved = solved

                self.logger.debug(f'\n\t\t\tNew best result')

            else:
                self.bands_final = np.copy(self.best_bands_final)
                self.final_score = np.copy(self.best_score)
                self.degenerate_final = np.copy(self.degenerate_best)      
                self.signal_final = np.copy(self.best_signal_final)  

                self.correct_signal()

                self.logger.info(f'\n\t\t\tBest result: {max_solved} bands')
                self.print_report(self.correct_signalfinal, f'Validation Report: Best Iteration', header_text=VALIDATE_RESULT_HEADER)     # Print result

            # Pace the alpha sweep by convergence at the current alpha (not by the rebuild
            # flag). Hold alpha while not_solved keeps making new lows; descend once it
            # stalls for `alpha_patience` consecutive iterations. A new best resets the
            # stall counter, so brief oscillations are absorbed and alpha drops only when
            # the current alpha has genuinely stopped improving. The tolerance is relaxed
            # one notch per descent (bounded, floored at 0.10 -- unlike the old
            # per-iteration decay), and the alpha sweep still guarantees termination
            # (alpha eventually drops below min_alpha).
            if result_stable:
                # Fixed point at this alpha: skip the remaining patience and descend now.
                self.alpha_stall = alpha_patience
                self.logger.info(f'\n\t\tAlpha {self.alpha:.4f}: result repeated exactly '
                                 f'(fixed point) -- forcing the alpha descent')
            elif attempt_not_solved < self.alpha_best_ns:
                self.alpha_best_ns = attempt_not_solved
                self.alpha_stall = 0
            else:
                self.alpha_stall += 1
                self.logger.info(f'\n\t\tAlpha {self.alpha:.4f}: no improvement '
                                 f'({attempt_not_solved} vs best {self.alpha_best_ns}) -- '
                                 f'stall {self.alpha_stall}/{alpha_patience}')

            if self.alpha_stall >= alpha_patience:
                self.alpha -= step
                self.alpha_best_ns = np.inf
                self.alpha_stall = 0
                if self.alpha < min_alpha:
                    # The sweep has exhausted its range; the loop condition ends it.
                    # Don't log a bogus "descending to -0.5000".
                    self.logger.info(f'\n\t\tAlpha converged at the final value -- '
                                     f'alpha sweep finished')
                else:
                    self.tol = max(self.tol * 0.90, 0.10)
                    self.logger.info(f'\n\t\tAlpha converged -- descending to '
                                     f'{self.alpha:.4f} (tol={self.tol:.4f})')

        # The best result is maintained
        self.bands_final = np.copy(self.best_bands_final)
        self.final_score = np.copy(self.best_score)
        self.degenerate_final = np.copy(self.degenerate_best)
        self.signal_final = np.copy(self.best_signal_final)

        # Snapshot the in-loop best (bands + validation) BEFORE the post-loop passes
        # run, so the verbose provenance log can tell genuine in-loop solves apart from
        # a-posteriori repairs and force-fills, and list the resulting band swaps.
        _prov_inloop_bands = np.copy(self.bands_final)
        _prov_inloop_csf = np.copy(self.correct_signalfinal_best)

        ###########################################################################
        # A-posteriori energy-continuity repair: the points the validation flagged
        # (NOT/MIS) and the unattributed ones are locally reassigned from trusted
        # trajectories. Good points are never touched.
        ###########################################################################
        self.logger.info('\n\t\tRepairing energy discontinuities (a posteriori)')
        self.repaired_mask = self._repair_energy_discontinuities()

        ###########################################################################
        # Force a genuine band attribution everywhere: every slot still
        # unattributed is filled with the closest available band in energy,
        # using a continuity reference. These points are flagged FORCED so the
        # report and basisrotation can tell them apart from genuine solves.
        ###########################################################################
        self.forced_mask = self._force_complete_bands()
        self.signal_final[self.forced_mask] = FORCED

        self.correct_signal(last=True)
        self.correct_signalfinal[self.forced_mask] = FORCED_CONTINUITY

        ###########################################################################
        # Structural + wavefunction-evidence passes. First relocate whole
        # mis-slotted patches -- tier 1 seeded on the SILENT cliffs (gross
        # energy jumps), tier 1b the SUB-gate cliffs seeded on the same collision
        # measured at the local band gap (see _relocate_cliff_patches), tier 2
        # seeded on dp-refuted seams whose energy step hides below the gate (all
        # two-sided, internally perfect, validated: the one defect class the
        # cell-level passes below cannot arbitrate;
        # see _relocate_patches) -- then place the seams of the refuted
        # regions the greedy flood refuses at their exact optimum (min-cut,
        # see _mincut_seam_relocate) -- then relocate the near-degenerate
        # patches the strict passes exempt as gauge but the 0.8 flood map still
        # names (see _relocate_degenerate_patches) -- then make every CERTAIN
        # component single-slotted (wrong patches at cell scale, cycles of
        # any order and -1 holes), then exchange the remaining
        # zero-dp-support cells where the swap is dp-favoured (sub-certain
        # evidence), then demote whatever zero-support cells remain so the
        # report cannot call a silently swapped band clean.
        ###########################################################################
        self._log_refuted_census('post correct_signal')
        # The relocation/repair passes each have monotone acceptance, but they are
        # sensitive to their STARTING canvas: applied once to the raw in-loop
        # canvas (frontier-collision seams everywhere) they settle in a far worse
        # optimum than when re-applied to their own, cleaner output. A second
        # sweep from the cleaner canvas escapes that local optimum -- on
        # phosphorene the refuted census drops 176 -> 95 on the re-sweep (band 15
        # heals) and then converges. Iterate the sweep until the combined seam
        # census (refuted + cliffs + walls) stops decreasing; each pass is
        # monotone and the census is a non-negative integer, so this terminates.
        def _seam_census():
            return (len(self._dp_refuted_seam_edges())
                    + len(self._cliff_seam_edges()) + len(self._slot_wall_edges()))

        MAX_SWEEPS = 8
        prev_census = None
        for _sweep in range(MAX_SWEEPS):
            self._relocate_wall_patches()
            self._relocate_cliff_patches()
            self._relocate_dp_seam_patches()
            self._mincut_seam_relocate()
            self._relocate_degenerate_patches()
            self._repair_dp_cycles()
            self._repair_certain_links()
            census = _seam_census()
            self._log_refuted_census(f'post relocation sweep {_sweep + 1}')
            if prev_census is not None and census >= prev_census:
                break
            prev_census = census
        self._repair_dp_swaps()
        self._flag_dp_unsupported()
        self._log_refuted_census('final')
        self._verify_no_cliffs()
        self._handoff_dp_residue_to_rotation()
        self._verify_no_dp_cycles()

        self.logger.info(self.report())

        # Verbose (-v) post-mortem: per-slot provenance (in-loop / repaired / forced),
        # every post-loop band swap, and the silent energy discontinuities that pass
        # the validation flag. No-op unless the logger is at DEBUG level.
        self._log_solution_provenance(_prov_inloop_bands, _prov_inloop_csf)

class COMPONENT:
    '''
    This object contains the information that constructs a component,
    and also it has functions that are necessary to establish
    relations between components.

    Atributes
        GRAPH : Graph
            It is a graph object that contains nodes and edges defined as G = (V, E).
        N : int
            It is the number of nodes that the graph contains.
        m_shape : Tuple[int, int]
            The shape of the k-space representation.
        nks : int
            The number of k-points.
        kpoints_index : array_like
            Contains the k-points' indices on a k-space projection.
        matrix : array_like
            The matrix that contains in each position the k-point identification.
        dimensions : int
            The number of dimensions of the k-space.
        position_matrix : None
            Contains the k-points' projection on k-space for points that belong to the graph.
        nodes : array_like
            It contains the node id for all graph's k-points.
        __id__ : string
            It identifies the component of the graph.
        was_modified : bool
            The flag that says if the component was modified in the last iteration.
                default = True
        scores : dict
            Stores the last score for some component with __id__ key.
    
    Methods
        calculate_pointsMatrix() : None
            Calculate the k-points' projection on a matrix representation of k-space.
        get_bands() : array_like
            Get the bands associated with each node.
        validate() : bool
            Verify if the component received as an argument can join with the current component.
        join() : None
            Join the component received as a parameter with the current component.
        calc_boundary() : None
            Only the boundary nodes are necessary. Therefore this function computes
                these essential nodes and uses them to compare components.
        get_cluster_score() : float
            This function returns the similarity between components taking
                into account the dot product of all essential points and their
                energy value.
    '''
    def __init__(self, component: nx.Graph, kpoints_index:np.ndarray, matrix: np.ndarray, dimensions:int) -> None:
        '''
        Setup the component information.

        Parameters
            GRAPH : Graph
                It is a graph object that contains nodes and edges defined as G = (V, E).
            N : int
                It is the number of nodes that the graph contains.
            m_shape : Tuple[int, int]
                The shape of the k-space representation.
            nks : int
                The number of k-points.
            kpoints_index : array_like
                Contains the k-points' indices on a k-space projection.
            matrix : array_like
                The matrix that contains in each position the k-point identification.
            dimensions : int
                The number of dimensions of the k-space.
            position_matrix : None
                Contains the k-points' projection on k-space for points that belong to the graph.
            nodes : array_like
                It contains the node id for all graph's k-points.
            __id__ : string
                It identifies the component of the graph.
            was_modified : bool
                The flag that says if the component was modified in the last iteration.
                    default = True
            scores : dict
                Stores the last score for some component with __id__ key.
        '''
        self.GRAPH = component
        self.N = self.GRAPH.number_of_nodes()
        self.m_shape = matrix.shape
        self.nks = np.prod(self.m_shape)
        self.kpoints_index = np.array(kpoints_index)
        self.matrix = matrix
        self.dimensions = dimensions
        self.positions_matrix = None
        self.nodes = np.array(self.GRAPH.nodes)

        self.__id__ = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        self.was_modified = True
        self.scores = {}

    def calculate_values(self):
        self.nodes = np.array(self.GRAPH.nodes)
        self.k_points = self.nodes % self.nks                               # Transform node id to k-point notation
        self.bands_number = dict(zip(self.k_points,
                                     self.nodes//self.nks))                 # A dictionary that links the initial band to a k-point

    def calculate_pointsMatrix(self) -> None:
        '''
        Calculate the k-points' projection on a matrix representation of k-space.
        '''
        self.nodes = np.array(self.GRAPH.nodes)
        self.k_points = self.nodes % self.nks                               # Transform node id to k-point notation
        self.bands_number = dict(zip(self.k_points,
                                     self.nodes//self.nks))

        self.positions_matrix = np.zeros(self.m_shape, int)                 # Position matrix for k-points projection
        index_points = self.kpoints_index[self.nodes % self.nks]            # Get the k-points' indices
        if self.dimensions == 1:
            self.positions_matrix[index_points] = 1                          # Mark the k-point projection
        elif self.dimensions == 2:
            self.positions_matrix[index_points[:, 0], index_points[:, 1]] = 1   # Mark the k-point projection
        else:
            self.positions_matrix[index_points[:, 0], index_points[:, 1], index_points[:, 2]] = 1

    def get_bands(self) -> None:
        '''
        Get the bands associated with each node.

        Returns
            k_bands : array_like
                An array with bands information.
        '''
        self.k_points = self.nodes % self.nks                               # Get the k-point from node
        k_bands = self.nodes//self.nks                                      # Get the k-point's band from node
        self.bands_number = dict(zip(self.k_points,
                                     k_bands))                 # A dictionary that links the initial band to a k-point
        bands, counts = np.unique(k_bands, return_counts=True)              # Count the number of nodes with each band
        self.bands = bands[np.argsort(counts)[::-1]]                        # Save in decreasing order
        return k_bands

    def validate(self, component : COMPONENT) -> bool:
        '''
        Verify if the component received as a parameter can join with the current component.

        Parameters
            component : COMPONENT
                It is another COMPONENT object that is trying to join the current component.
        
        Returns
            validate() : bool
                If it is true, the component does not have overlaying between k-points,
                and the sum of nodes does not exceed the total number of k-points
        '''
        if self.positions_matrix is None:
            # Calculates the k-space projection if it does not exist
            self.calculate_pointsMatrix()
        # Computes the overlaying between components by XOR operation between k-space projections
        N = np.sum(self.positions_matrix ^ component.positions_matrix)
        return (component.N <= self.nks - self.N and N == self.N+component.N)

    def join(self, component : COMPONENT) -> None:
        '''
        Join the component received as a parameter with the current component.

        Parameter
            component : COMPONENT
                It is another COMPONENT object to be joined to the current component.
        '''
        del component.scores                            # Clear the scores' information of the component
        self.was_modified = True                        # The Component is marked as modified
        G = nx.Graph(self.GRAPH)                        # Copy the Graph
        G.add_nodes_from(component.GRAPH)               # Add the new nodes
        self.GRAPH = G                                  # Set the new Graph
        self.N = self.GRAPH.number_of_nodes()           # Atualize the number of nodes
        self.nodes = np.array(self.GRAPH.nodes)
        # Update the other attributes
        self.calculate_pointsMatrix()
        self.calc_boundary()

    def calc_boundary(self) -> None:
        '''
        Only the boundary nodes are necessary. Therefore this function computes
        these essential nodes and uses them to compare components.
        '''
        if self.positions_matrix is None:
            # Calculates the k-space projection if it does not exist
            self.calculate_pointsMatrix()

        if self.dimensions == 1:
            A = np.where(self.positions_matrix == 1)[0]
            self.boundary = np.zeros(self.positions_matrix.shape, dtype=int)
            if len(A) > 0:
                self.boundary[A[0]] = 1
                self.boundary[A[-1]] = 1
        
        elif self.dimensions == 2:
            Ax = sobel(self.positions_matrix, axis=0)    # Sobel operator for x-axis
            Ay = sobel(self.positions_matrix, axis=1)    # Sobel operator for y-axis
            self.boundary = np.sqrt(Ax**2 + Ay**2)         # Boundary points
        else:
            Ax = sobel(self.positions_matrix, axis=0)
            Ay = sobel(self.positions_matrix, axis=1)
            Az = sobel(self.positions_matrix, axis=2)
            self.boundary = np.sqrt(Ax**2 + Ay**2 + Az**2)

        # Maintain  all marked points
        self.boundary = self.boundary * self.positions_matrix
        self.boundary = (self.boundary > 0)
        self.k_edges = self.matrix[self.boundary]
        if len(self.k_edges) == 0:
            self.k_edges = self.nodes % self.nks

    def get_cluster_score(self, cluster : COMPONENT,
                          neighbors : np.ndarray, energies : np.ndarray, connections : np.ndarray, alpha : float = 0.5,
                          accept_E : float = None, disp_scale : float = None, fe_eweight : float = 1.0,
                          tol : float = None) -> float:
        '''
        This function returns the similarity between components taking
        into account the dot product of all essential points and their
        energy value.

        Parameters
            cluster : COMPONENT
                It is a component with which the similarity is calculated.
            neighbors : array_like
                It is an array that identifies the neighbors of each k point.
            energies : array_like
                It is an array of the energy values inside a matrix.
            connections : array_like
                It is an array with the dot product between k points
                    and his neighbors.
            alpha : float
                This is the weight given to the dotproduct value to compute the score.
                    score = alpha*<i|j> + (1-alpha)*f(E_i)
        Return
            score: float
                It is a float that represents the similarity between components.
        '''
        def difference_energy(bn1 : int, bn2 : int, iK1 : list[int], iK2: list[int], Ei : float = None) -> float:
            '''
            Compute the score for the k-point with their neighbor by comparing all possible energies with Ei energy.

            Parameters
                bn1 : int
                    k-point's band.
                bn2 : int
                    neighbor's band.
                iK1 : list[int]
                    K-point's indices on k-space.
                iK2 : list[int]
                    Neighbor's indices on k-space.
                Ei : float
                    Energy to comparison. The default is None
            
            Return
                score : float
                    If the neighbor's energy is the closest one to Ei the score is one, otherwise is between 0 and 1.
            '''
            ik1 = iK1[0]
            ik_n = iK2[0]
            if self.dimensions >= 2:
                jk1 = iK1[1]
                jk_n = iK2[1]
            if self.dimensions == 3:
                kk1 = iK1[2]
                kk_n = iK2[2]

            predicted = Ei is not None                                     # f(E) prediction path (Ei=Enew) vs default
            energies_candidates = lambda bn: energies[bn, ik_n] if self.dimensions == 1 else energies[bn, ik_n, jk_n] if self.dimensions == 2 else energies[bn, ik_n, jk_n, kk_n]

            if self.dimensions == 1:
                Ei = energies[bn1, ik1] if Ei is None else Ei              # The k-point Energy is used as default
            elif self.dimensions == 2:
                Ei = energies[bn1, ik1, jk1] if Ei is None else Ei
            else:
                Ei = energies[bn1, ik1, jk1, kk1] if Ei is None else Ei        

            Ecand = energies_candidates(bn2)                               # Candidate (neighbour band) energy
            delta_energy = np.abs(Ei - Ecand)                              # Directional residual: prediction vs THIS candidate
            if _FE_DEBUG and predicted:
                # f(E) diagnostics, Tier 2: residual = how far the extrapolated
                # energy (Ei=Enew) lands from the neighbour it is scored against.
                _FE_STATS['resid_n'] += 1
                _FE_STATS['resid_buckets'][_fe_resid_bucket(float(delta_energy), disp_scale)] += 1
            # Directional continuity-residual with a gross-jump gate, replacing the
            # old min(|Ei - E_all_bands|)/delta_energy ratio: score the candidate
            # band against the prediction along the join line ONLY, and veto any
            # jump beyond the gate (a real inter-group wall can never be crossed).
            # The gate stays at the GLOBAL scales on purpose: band_disp is a p95, so
            # ~5% of genuine one-step dispersions exceed it (the max is ~2.3x p95 on
            # phosphorene) and a locally sharpened gate vetoes real continuations in
            # the steep parts of a resolved manifold, fragmenting clusters (the
            # log1-vs-log2 phosphorene regression, and e90a33a before it). The local
            # gates (_local_accept) stay where a trusted trajectory exists to judge
            # against: seeds/growth in _partition_fused_component, the repair pass,
            # the outlier mask and the completeness fill.
            return _energy_continuity_score(float(Ei), float(Ecand),
                                            accept_E=accept_E, disp_scale=disp_scale)
        
        def fit_energy(bn1 : int, bn2 : int, iK1 : list[int], iK2: list[int]) -> float:
            '''
            Computes the best energy approximation for the neighbor's position.

            Parameters
                bn1 : int
                    k-point's band.
                bn2 : int
                    neighbor's band.
                iK1 : list[int]
                    K-point's indices on k-space.
                iK2 : list[int]
                    Neighbor's indices on k-space.
            
            Return
                score : float
                    Result of difference_energy function using the computed Energy.
            '''
            N = 4                                                               # Number of points to take account into the curve fitting

            ###########################################################################
            # New f(E): cliff-aware, dispersion-clipped one-step extrapolation,
            # applied here in the IN-LOOP score term (mirrors the post-loop
            # MATERIAL._extrapolate_energy). Enabled only when accept_E/disp_scale
            # are supplied; otherwise predict_energy reproduces the original plain
            # quadratic fit exactly, so the change is a no-op unless wired in.
            ###########################################################################
            use_new_fE = accept_E is not None and disp_scale is not None
            _accept_E  = accept_E if accept_E is not None else np.inf
            _disp      = disp_scale if disp_scale is not None else np.inf
            pol = lambda x, a, b, c: a*x**2 + b*x + c                            # Second order polynomial

            def predict_energy(X, Es, bands, new_x):
                '''
                One-step energy prediction at ``new_x`` from the band's own trajectory.
                ``X``/``Es``/``bands`` are ordered anchor-first: X[0] is the k-point
                nearest the neighbour. With the new f(E) enabled the walk is truncated
                at the first cliff -- an energy step exceeding the gross-jump scale
                ``accept_E`` OR a change of raw band index -- so the quadratic is never
                fitted across a discontinuity, and the prediction is clipped to a local
                dispersion window E(anchor) +/- ``disp_scale``. Returns None when fewer
                than 4 continuous points remain (caller falls back to the
                nearest-neighbour energy default). With the new f(E) disabled it is the
                original plain quadratic fit over all supplied points.
                '''
                X = np.asarray(X, dtype=float)
                Es = np.asarray(Es, dtype=float)
                bands = np.asarray(bands)
                if _FE_DEBUG:
                    _FE_STATS['calls'] += 1
                if use_new_fE:
                    Xt, Et = [X[0]], [Es[0]]
                    b_anchor = bands[0]
                    for x, e, b in zip(X[1:], Es[1:], bands[1:]):
                        ejump = abs(e - Et[-1]) > _accept_E
                        if ejump or b != b_anchor:
                            if _FE_DEBUG:
                                _FE_STATS['brk_ejump' if ejump else 'brk_band'] += 1
                            break                                               # cliff: stop the walk
                        Xt.append(x); Et.append(e)
                    if _FE_DEBUG:
                        _FE_STATS['kept_len_sum'] += len(Et)
                        if len(Et) < len(X):
                            _FE_STATS['trunc'] += 1
                    if len(Et) <= 3:
                        if _FE_DEBUG:
                            _FE_STATS['fallback'] += 1
                        return None                                             # too few continuous points
                else:
                    Xt, Et = X, Es
                    if _FE_DEBUG:
                        _FE_STATS['kept_len_sum'] += len(Et)
                popt, _ = curve_fit(pol, Xt, Et)
                Enew = float(pol(new_x, *popt))
                if use_new_fE:
                    E0 = float(Et[0])
                    clipped = min(max(Enew, E0 - _disp), E0 + _disp)            # local dispersion window
                    if _FE_DEBUG and clipped != Enew:
                        _FE_STATS['clip_hit'] += 1
                    Enew = clipped
                if _FE_DEBUG:
                    _FE_STATS['fit'] += 1
                return Enew
            ik1 = iK1[0]
            ik_n = iK2[0]
            if self.dimensions >= 2:
                jk1 = iK1[1]
                jk_n = iK2[1]
            if self.dimensions == 3:
                kk1 = iK1[2]
                kk_n = iK2[2]
            

            ###########################################################################
            # Preparation of the points' indices
            ###########################################################################

            if self.dimensions == 1:
                I = np.full(N+1,ik1)                                                # Repeat N+1 times the ik1 value  
                i = I + np.arange(0,N+1)*np.sign(ik1-ik_n)
                i = i[i >= 0]
                i = i[i < self.m_shape[0]]
                ###########################################################################
                # Computes the best energy using the points above defined
                ###########################################################################
                ks = self.matrix[i]                                                  # Select k-points on positions delimited by the (i,j) indices
                f = lambda e: e in self.k_points                                     # Auxiliar lambda function to verify if an e point is inside component's k-points
                exist_ks = list(map(f, ks))                                          # Apply f to all ks
                ks = ks[exist_ks]                                                    # Maintain only the existent k-points
                if len(ks) <= 3:
                    # It is necessary at least 3 points to fitting a second order curve
                    # If there are not enough points is used the difference_energy's default
                    return difference_energy(bn1, bn2, iK1, iK2)
                aux_bands = np.array([self.bands_number[kp] for kp in ks])
                bands = aux_bands
                # Use the existent k-points' indices
                i = i[exist_ks]
                Es = energies[bands, i]                                              # Get the ks' energies
                X = i                                                               # Obtain the x values for Es.
                new_x = ik_n                                                         # Get the position to approximate the energy

                Enew = predict_energy(X, Es, bands, new_x)                          # Cliff-aware, clipped prediction
                if Enew is None:
                    return difference_energy(bn1, bn2, iK1, iK2)                    # no continuous segment -> default
                return difference_energy(bn1, bn2, iK1, iK2, Ei = Enew)             # Score

            if self.dimensions == 2:

                I = np.full(N+1,ik1)                                                # Repeat N+1 times the ik1 value  
                J = np.full(N+1,jk1)                                                # Repeat N+1 times the jk1 value  
                flag = ik1 == ik_n                                                  # Identify the neighbor's direction
                # Take the (i,j) indices of N+1 points
                i = I if flag else I + np.arange(0,N+1)*np.sign(ik1-ik_n)
                j = J if not flag else J + np.arange(0,N+1)*np.sign(jk1-jk_n)
                
                if not flag:
                    # If the neighbor is in jk's direction then i is corrected to be inside boundaries.
                    # The shape of js indices is corrected
                    i = i[i >= 0]
                    i = i[i < self.m_shape[0]]
                    j = np.full(len(i), jk1)
                else:
                    # If the neighbor is in ik's direction then j is corrected to be inside boundaries.
                    # The shape of is indices is corrected
                    j = j[j >= 0]
                    j = j[j < self.m_shape[1]]
                    i = np.full(len(j), ik1)

                ###########################################################################
                # Computes the best energy using the points above defined
                ###########################################################################
                ks = self.matrix[i, j]                                                  # Select k-points on positions delimited by the (i,j) indices
                f = lambda e: e in self.k_points                                        # Auxiliar lambda function to verify if an e point is inside component's k-points
                exist_ks = list(map(f, ks))                                             # Apply f to all ks
                ks = ks[exist_ks]                                                       # Maintain only the existent k-points
                if len(ks) <= 3:
                    # It is necessary at least 3 points to fitting a second order curve
                    # If there are not enough points is used the difference_energy's default
                    return difference_energy(bn1, bn2, iK1, iK2)
                aux_bands = np.array([self.bands_number[kp] for kp in ks])              # Get the ks' bands
                bands = aux_bands
                # Use the existent k-points' indices
                i = i[exist_ks]
                j = j[exist_ks]
                Es = energies[bands, i, j]                                              # Get the ks' energies
                X = i if jk1 == jk_n else j                                             # Obtain the x values for Es.
                new_x = ik_n if jk1 == jk_n else jk_n                                   # Get the position to approximate the energy

                Enew = predict_energy(X, Es, bands, new_x)                              # Cliff-aware, clipped prediction
                if Enew is None:
                    return difference_energy(bn1, bn2, iK1, iK2)                        # no continuous segment -> default
                return difference_energy(bn1, bn2, iK1, iK2, Ei = Enew)                 # Score

            if self.dimensions == 3:
                    
                I = np.full(N+1,ik1)                                                    # Repeat N+1 times the ik1 value  
                J = np.full(N+1,jk1)                                                    # Repeat N+1 times the jk1 value  
                K = np.full(N+1,kk1)                                                    # Repeat N+1 times the kk1 value  
                flag_i = ik1 == ik_n                                                      # Identify the neighbor's direction
                flag_j = jk1 == jk_n                                                      # Identify the neighbor's direction
                # Take the (i,j, k) indices of N+1 points
                i = I if flag_i else I + np.arange(0,N+1)*np.sign(ik1-ik_n)
                j = J if flag_j else J + np.arange(0,N+1)*np.sign(jk1-jk_n)
                k = K if not flag_i and not flag_j else K + np.arange(0,N+1)*np.sign(kk1-kk_n)

                if not flag_i:   
                    # If the neighbor is in jk's direction then i is corrected to be inside boundaries.
                    # The shape of js indices is corrected
                    i = i[i >= 0]
                    i = i[i < self.m_shape[0]]
                    j = np.full(len(i), jk1)
                    k = np.full(len(i), kk1)
                elif not flag_j:
                    # If the neighbor is in ik's direction then j is corrected to be inside boundaries.
                    # The shape of is indices is corrected
                    j = j[j >= 0]
                    j = j[j < self.m_shape[1]]
                    i = np.full(len(j), ik1)
                    k = np.full(len(j), kk1)
                else:
                    # If the neighbor is in kk's direction then k is corrected to be inside boundaries.
                    # The shape of is indices is corrected
                    k = k[k >= 0]
                    k = k[k < self.m_shape[2]]
                    i = np.full(len(k), ik1)
                    j = np.full(len(k), jk1)

                ###########################################################################
                # Computes the best energy using the points above defined
                ###########################################################################
                ks = self.matrix[i, j, k]                                               # Select k-points on positions delimited by the (i,j) indices
                f = lambda e: e in self.k_points                                        # Auxiliar lambda function to verify if an e point is inside component's k-points
                exist_ks = list(map(f, ks))                                             # Apply f to all ks
                ks = ks[exist_ks]                                                       # Maintain only the existent k-points
                if len(ks) <= 3:
                    # It is necessary at least 3 points to fitting a second order curve
                    # If there are not enough points is used the difference_energy's default
                    return difference_energy(bn1, bn2, iK1, iK2)
                aux_bands = np.array([self.bands_number[kp] for kp in ks])              # Get the ks' bands
                bands = aux_bands
                # Use the existent k-points' indices
                i = i[exist_ks]
                j = j[exist_ks]
                k = k[exist_ks]
                Es = energies[bands, i, j, k]                                           # Get the ks' energies
                X = i if jk1 == jk_n and kk1 == kk_n else j if ik1 == ik_n and kk1 == kk_n else k # Obtain the x values for Es.
                new_x = ik_n if jk1 == jk_n and kk1 == kk_n else jk_n if ik1 == ik_n and kk1 == kk_n else kk_n # Get the position to approximate the energy

                Enew = predict_energy(X, Es, bands, new_x)                              # Cliff-aware, clipped prediction
                if Enew is None:
                    return difference_energy(bn1, bn2, iK1, iK2)                        # no continuous segment -> default
                return difference_energy(bn1, bn2, iK1, iK2, Ei = Enew)                 # Score
                

        ###########################################################################
        # Computes the final score between components
        ###########################################################################
        if not cluster.was_modified:
            # If the cluster was not modified the previous result is maintained
            return self.scores[cluster.__id__]

        score = 0
        count_k = 0
        for k in self.k_edges:
            # Each k-point is compared with his respective neighbor
            # that belongs to the comparison component
            bn1 = self.bands_number[k]                                               # k-point's band

            ik_point_index = []
            ik1 = self.kpoints_index[k, 0] if self.dimensions > 1 else self.kpoints_index[k]
            ik_point_index.append(ik1)
            if self.dimensions >= 2:
                jk1 = self.kpoints_index[k, 1]                                        # k-point idices
                ik_point_index.append(jk1)
            if self.dimensions == 3:
                kk1 = self.kpoints_index[k, 2]                                   # k-point idices
                ik_point_index.append(kk1)

            for i_neig, k_n in enumerate(neighbors[k]):
                # k-point's neighbors
                count_k += 1
                if k_n == -1 or k_n not in cluster.k_edges:
                    # If the neighbor is not a valid point the score for that point is 0
                    continue
                ik_n_point_index = []
                ik_n = cluster.kpoints_index[k_n, 0] if self.dimensions > 1 else cluster.kpoints_index[k_n]
                ik_n_point_index.append(ik_n)
                if self.dimensions >= 2:
                    jk_n = cluster.kpoints_index[k_n, 1]                             # neighbor's indices
                    ik_n_point_index.append(jk_n)
                if self.dimensions == 3:
                    kk_n = cluster.kpoints_index[k_n, 2]                             # neighbor's indices
                    ik_n_point_index.append(kk_n)
                bn2 = cluster.bands_number[k_n]                                     # neighbor's band
                connection = connections[k, i_neig, bn1, bn2]                       # Dot product between k-point and his neighbor
                energy_val = fit_energy(bn1, bn2, ik_point_index, ik_n_point_index)
                # Gross-jump gate: fit_energy returns exactly 0.0 only when the
                # candidate band sits across an energy wall (|dE| > accept_E).
                # Veto the WHOLE edge then -- a spuriously high dot product must
                # never pull a band across a genuine inter-group gap (the band-1
                # leak across the 0.46 Ry MoS2 wall). Loose gate (accept_E), so it
                # only ever cuts gross jumps, not intra-manifold continuations.
                gate = 0.0 if energy_val == 0.0 else 1.0
                # Cross-band dot-product veto (mirror of the make_connections
                # anti-bridge guard). A cross-band join (bn1 != bn2) is only
                # physical at a genuine crossing, where the sample band bn1 has
                # LOST its own continuity here. If bn1 still overlaps itself
                # strongly across this edge (<bn1@k|bn1@k_n> > tol), bn1 has its
                # own continuation and does NOT belong in the bn2 cluster: this is
                # the near-degenerate-partner swap (a band-15 fragment pulled into
                # the band-14 slot on the strength of f(E)~1 alone while the cross
                # overlap <15|14> = 0). Veto the edge so energy continuity can never
                # bridge two orthogonal (degenerate-pair) partners. Genuine
                # crossings -- where the same-band overlap has collapsed -- keep
                # same <= tol and are NOT vetoed.
                if tol is not None and bn1 != bn2 and \
                        connections[k, i_neig, bn1, bn1] > tol:
                    gate = 0.0
                # Down-weight f(E) when the gap has collapsed (fe_eweight < 1) and
                # renormalise so the dot product carries the freed weight. With
                # fe_eweight == 1 (well-separated bands) this is the original
                # alpha*conn + (1-alpha)*f(E) exactly.
                w_e = (1 - alpha) * fe_eweight
                w_c = 1 - w_e
                if _FE_DEBUG:
                    # f(E) diagnostics, Tier 1: relative weight each term carries in
                    # the blended score. If eng_sum << conn_sum, f(E) is drowned out.
                    _FE_STATS['pair_n'] += 1
                    _FE_STATS['conn_sum'] += float(gate * w_c * connection)
                    _FE_STATS['eng_sum'] += float(gate * w_e * energy_val)
                score += gate * (w_c*connection + w_e*energy_val)                   # Calculates the final k-point score
        score /= count_k if count_k > 0 else 1                                             # Get the final score
        self.scores[cluster.__id__] = score                                         # Store the score
        return score
