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
import random
import string
import textwrap
from typing import Tuple, Union, Callable

import logging

from scipy.ndimage import sobel, correlate
from scipy.optimize import curve_fit

import numpy as np
import networkx as nx
import time

from berry import log
from .write_k_points import _bands_numbers
from berry._subroutines.contatempo import tempo
try:
    import berry._subroutines.loadmeta as m
except:
    pass

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
_FE_DEBUG: bool = True
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


EVALUATE_RESULT_HELP = '''
    ---------------------- Report considering dot-product information ----------------------------
            C -> Mean dot-product |<i|j>| of each k-point
            ----------------------------------------------------------------------------
            Value | Abbreviation |                Description
            ----------------------------------------------------------------------------
            0     |     NOT      |      -                     The point is not solved
            1     |     MIS      |      MISTAKE               C <= 0.2
            2     |     DEG      |      DEGENERATE            It is a degenerate point
            3     |     PMI      |      POTENTIAL_MISTAKE     C <= 0.8
            4     |     PCO      |      POTENTIAL_CORRECT     0.8 < C < 0.9
            5     |     COR      |      CORRECT               C > 0.9
            6     |     FOR      |      FORCED                Force-filled by the completeness pass
    -----------------------------------------------------------------------------------------------
'''
EVALUATE_RESULT_HEADER = ['NOT', 'MIS', 'DEG', 'PMI', 'PCO', 'COR', 'FOR']

VALIDATE_RESULT_HELP = lambda N: f'''
    ----------------------- Report considering energy continuity criteria -------------------------
            N -> Number of directions that preserves energy continuity. ({N} is the maximum)
            C -> Mean dot-product |<i|j>| of each k-point. (1 is the maximum)
            ----------------------------------------------------------------------------
            Value | Abbreviation |                Description
            ----------------------------------------------------------------------------
            0     |     NOT      |      -                   The point is not solved
            1     |     MIS      |      MISTAKE             N = 0 and/or C <= 0.2
            2     |     DEG      |      DEGENERATE          It is a degenerate point
            3     |     OTH      |      OTHER               0 < N < {N} and 0.8 < C < 0.9
            4     |     COR      |      CORRECT             N = 4 and C > 0.9
            5     |     FOR      |      FORCED              Force-filled by the completeness pass
    ----------------------------------------------------------------------------------------------
'''
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
    'Status': 'CLEAN (pristine, nothing flagged) / USABLE (no genuine errors; may carry benign force-fills or a sub-pristine score) / ROTATE (needs basis rotation) / CHECK (a suspect force-fill across a real gap -> verify) / FAIL (genuine energy break, or many force-fills across a real gap)',
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
    f'''
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
    '''

    TOL = 0.9       # Tolerance for CORRECT output
    TOL_DEG = 0.8   # Tolerance for POTENTIAL_CORRECT output
    TOL_MIN = 0.2   # Tolerance for POTENTIAL_MISTAKE output

    value = np.mean(values) # Mean conection of each k point

    if value > TOL:
        return CORRECT

    if value > TOL_DEG:
        return POTENTIAL_CORRECT

    if value > TOL_MIN and value < TOL_DEG:
        return POTENTIAL_MISTAKE

    return MISTAKE

def evaluate_point(dimension:int, k: Kpoint, bn: Band, k_index: np.ndarray, k_matrix: np.ndarray,
                   signal: np.ndarray, bands: np.ndarray, energies: np.ndarray,
                   accept_E: float = None, disp_scale: float = None) -> Tuple[int, list[int]]:
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
        signal: array_like
            An array with the current signal value for each k point.
        bands: array_like
            An array with the information of current solution of band clustering.
        energies: array_like
            It contais the energy value for each k point.
    
    Returns
        (signal, scores): Tuple[int, list[int]]
            scores: list[int]
                Sinalize if exist continuity on each direction [Down, Right, up, Left].
                    1 --- This direction preserves energy continuity.
                    0 --- This direction does not preserves energy continuity.
            N -> Number of directions with energy continuity.
            signal: int
                Value :                              Description
                0     :                        The point is not solved
                1     :  MISTAKE               N = 0
                2     :  DEGENERATE            It is a degenerate point.
                3     :  OTHER                 0 < N < 4
                4     :  CORRECT               N = 4
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
    sig = signal[k, bn]             # signal

    if dimension == 1:
        ik = k_index[k]             # k point index on k-space
    elif dimension == 2:
        ik, jk = k_index[k]         # k point indices on k-space
    else:
        ik, jk, kk = k_index[k]     # k point indices on k-space

    Ek = energies[k, mach_bn]       # k point's Energy value

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
            i, j, k = kn_index[:, 0], kn_index[:, 1], kn_index[:, 2]
            flag_i = len(np.unique(i)) > 1
            flag_j = len(np.unique(j)) > 1

            if flag_i:
                i = i[i >= 0]
                i = i[i < k_matrix.shape[0]]
                j = np.full(len(i), j[0])
                k = np.full(len(i), k[0])
            elif flag_j:
                j = j[j >= 0]
                j = j[j < k_matrix.shape[1]]
                i = np.full(len(j), i[0])
                k = np.full(len(j), k[0])
            else:
                k = k[k >= 0]
                k = k[k < k_matrix.shape[2]]
                i = np.full(len(k), i[0])
                j = np.full(len(k), j[0])

            ks = k_matrix[i, j, k] if len(i) > 0 else []   # Identify the N k points
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
            X = i if flag_i else j if flag_j else k
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
                 connections: np.ndarray, neighbors: np.ndarray, logger: log, min_band: int, n_process: int=1) -> None:
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
            n_process : int
                Number of processes to use.
        '''
        self.dimensions = dimensions
        self.nkx, self.nky, self.nkz = nk_i
        self.nbnd = nbnd - min_band
        self.total_bands = nbnd - min_band
        self.nks = nks
        self.eigenvalues = eigenvalues[:, min_band:]
        self.connections = connections
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
        bands_final, _ = np.meshgrid(np.arange(0, self.nbnd),
                                     np.arange(0, self.nks))
        
        if self.dimensions == 1:
            BandsEnergy = np.empty((self.nbnd, self.nks), float)
            for bn in range(self.nbnd):
                BandsEnergy[bn] = self.eigenvalues[:, bn]
            return BandsEnergy
        
        if self.dimensions == 2:
            BandsEnergy = np.empty((self.nbnd, self.nkx, self.nky), float)
            for bn in range(self.nbnd):
                count = -1
                for j in range(self.nky):
                    for i in range(self.nkx):
                        count += 1
                        BandsEnergy[bn, i, j] = self.eigenvalues[count,
                                                        bands_final[count, bn]]
            return BandsEnergy
        
        BandsEnergy = np.empty((self.nbnd, self.nkx, self.nky, self.nkz), float)
        for bn in range(self.nbnd):
            count = -1
            for k in range(self.nkz):
                for j in range(self.nky):
                    for i in range(self.nkx):
                        count += 1
                        BandsEnergy[bn, i, j, k] = self.eigenvalues[count,
                                                                    bands_final[count, bn]]
        return BandsEnergy

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

    def make_vectors(self, min_band: int=0, max_band: int=-1) -> None:
        '''
        It transforms the information into more convenient data structures.

        Parameters
            min_band : int
                An integer that gives the minimum band that clustering will use.
                    default: 0
            max_band : int
                An integer that gives the maximum band that clustering will use.
                    default: All

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
        self.min_band = min_band
        self.max_band = m.final_band
        nbnd = self.nbnd if max_band == -1 else self.nbnd
        self.make_kpointsIndex()
        energies = self.make_BandsEnergy()
        self.logger.percent_complete(20, 100, title=process_name)

        ###########################################################################
        # Compute the vector representation of each k point
        ###########################################################################
        # n_vectors = (nbnd - min_band)*self.nks
        n_vectors = nbnd*self.nks
        # ik = np.tile(self.kpoints_index[:, 0], nbnd-min_band) if self.dimensions > 1 else np.tile(self.kpoints_index, nbnd-min_band)
        ik = np.tile(self.kpoints_index[:, 0], nbnd) if self.dimensions > 1 else np.tile(self.kpoints_index, nbnd)

        stack_aux = [ik]

        if self.dimensions >= 2:
            # jk = np.tile(self.kpoints_index[:, 1], nbnd-min_band)
            jk = np.tile(self.kpoints_index[:, 1], nbnd)
            stack_aux.append(jk)
        if self.dimensions == 3:
            # kk = np.tile(self.kpoints_index[:, 2], nbnd-min_band)
            kk = np.tile(self.kpoints_index[:, 2], nbnd)
            stack_aux.append(kk)

        # bands = np.arange(min_band, nbnd)
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

        # interband_scale: a robust INTER-band gap that ignores the near-zero
        # SOC/Kramers-pair splittings which dominate (and collapse) gap_scale.
        # Used only as a loose reference for diagnostics; the gross-jump gate
        # itself keys to accept_E (3*disp_scale), which is already wider than any
        # legitimate one-step dispersion and far below a genuine inter-band wall.
        nondeg = gaps[gaps > self.gap_scale] if gaps.size else gaps
        self.interband_scale = float(np.median(nondeg)) if nondeg.size else self.gap_scale

        disp_values = []
        neigh = np.asarray(self.neighbors)
        for j in range(neigh.shape[1]):
            nb = neigh[:, j]
            valid = nb != -1
            if np.any(valid):
                d = np.abs(self.eigenvalues[nb[valid]] - self.eigenvalues[valid])
                if d.size:
                    disp_values.append(d.ravel())
        disp_all = np.concatenate(disp_values) if disp_values else np.array([])
        disp_scale = float(np.percentile(disp_all, 99.9)) if disp_all.size else 0.0
        self.disp_scale = disp_scale if disp_scale > 0 else self.gap_scale
        self.sigma_min = 0.25 * self.gap_scale                          # min local-spread scale for _extrapolate_energy
        self.accept_E = max(3.0 * self.disp_scale, 0.5 * self.gap_scale)  # gross-jump guard for repair/forced fills

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

        # self.nbnd = nbnd-min_band
        self.bands_final = np.full((self.nks, self.total_bands), -1, dtype=int)
        # self.bands_final, _ = np.meshgrid(bands, np.arange(self.nks))

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
                tol = max(self.accept_E, sigma_mult * float(np.median(sigmas)))
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
        built from TRUSTED neighbours only, and the freed bands are redistributed to
        the slots nearest in energy to their predictions, respecting the per-k-point
        permutation constraint (no band used twice). A reassignment is only accepted
        if it lands within ``self.accept_E`` of the prediction (gross-jump guard).

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
                # Free all bad slots at k; bands kept by trusted slots stay used.
                ###########################################################################
                kept = self.bands_final[k][~bad[k]]
                used = np.unique(kept[kept != -1])
                available = list(np.setdiff1d(all_bands, used))
                avail_E = [float(self.eigenvalues[k, b]) for b in available]

                # Predicted (reference) energy per bad slot, from every direction
                # whose nearest neighbour is trusted in that slot; median for robustness.
                refs = []
                for s in bad_slots:
                    preds = [self._extrapolate_energy(k_nb, k, int(s), trusted=trusted)[0]
                             for k_nb in self.neighbors[k]
                             if k_nb != -1 and trusted[k_nb, s]]
                    refs.append(float(np.median(preds)) if preds else None)

                original = self.bands_final[k, bad_slots].copy()
                self.bands_final[k, bad_slots] = -1

                ###########################################################################
                # Assign predicted slots in energy order, each taking the nearest
                # still-available band (greedy nearest with removal == 1-D optimum),
                # accepted only within the gross-jump guard.
                ###########################################################################
                pred_idx = [i for i, r in enumerate(refs) if r is not None]
                pred_idx.sort(key=lambda i: refs[i])
                for i in pred_idx:
                    if not available:
                        break
                    j = int(np.argmin([abs(e - refs[i]) for e in avail_E]))
                    if abs(avail_E[j] - refs[i]) > self.accept_E:
                        continue                            # nothing continuous available: leave for later
                    s = int(bad_slots[i])
                    self.bands_final[k, s] = available.pop(j)
                    avail_E.pop(j)

                for i, s in enumerate(bad_slots):
                    s = int(s)
                    if self.bands_final[k, s] != -1:
                        # Repaired: trust it so the next layer can extrapolate through it.
                        bad[k, s] = False
                        trusted[k, s] = True
                        progress += 1
                        if self.bands_final[k, s] != original[i]:
                            repaired_mask[k, s] = True
                            n_repaired += 1
                            if self.logger.level <= logging.DEBUG:
                                nb_ = int(self.bands_final[k, s])
                                rE = refs[i]
                                self.logger.debug(
                                    f'{BODY_INDENT}  [repair] k={k} slot={s}: '
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

            self.logger.debug(f'\t\tRepair round {round_ + 1}: {progress} point(s) re-anchored')
            if progress == 0:
                break

        n_left = int(np.sum(bad))
        self.logger.info(f'{BODY_INDENT}Energy repair: {n_bad_initial} flagged point(s), '
                         f'{n_repaired} reassigned, {n_left} not repairable '
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
        attributed slot ("available") are assigned to the empty slots so that
        each slot takes the available band closest in energy to where its own
        trajectory predicts it should be (continuity reference). Slots are filled
        in reference-energy order taking the nearest still-available band, which
        is the conflict-free, 1-D optimal realisation of "closest band in energy".

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
            # Fill low-reference-energy slots first, each taking the nearest
            # still-available band (greedy nearest with removal == 1-D optimum).
            for idx in np.argsort(refs, kind='stable'):
                s = int(empty_slots[idx])
                if not available:
                    self.bands_final[k, s] = s          # last resort (permutation defect)
                    if self.logger.level <= logging.DEBUG:
                        self.logger.debug(f'{BODY_INDENT}  [forced] k={k} slot={s}: '
                                          f'no band available -> last-resort band {s}')
                    continue
                j = int(np.argmin([abs(e - refs[idx]) for e in avail_E]))
                chosen = int(available[j])
                dE = abs(avail_E[j] - refs[idx])
                self.bands_final[k, s] = available.pop(j)
                avail_E.pop(j)
                if self.logger.level <= logging.DEBUG:
                    self.logger.debug(f'{BODY_INDENT}  [forced] k={k} slot={s}: '
                                      f'-1->band {chosen}  E={float(self.eigenvalues[k, chosen]):.4f}  '
                                      f'refE={float(refs[idx]):.4f}  |dE|={dE:.4f}')
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
                default: 0.95
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
                    i_neig = np.where(self.neighbors[k1] == k2)[0]
                    connection = self.connections[k1, i_neig,
                                                    bn1, bn2]  # <i|j>
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
                        same = self.connections[k1, i_neig, bn1, bn1]   # <bn1@k1|bn1@k2>
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
                    if len(n2_idx) == 0 or len(n2_idx[0]) == 0:
                        continue
                    n2 = N2[n2_idx[0][0]]
                    N.append([n1, n2])
                if len(N) == 0:
                    continue
                flag = False
            else:
                if len(N1) == len(N2):
                    N = list(zip(N1, N2))
                else:
                    Ns = [N1, N2]
                    N_1 = Ns[np.argmin([len(N1), len(N2)])]
                    N_2 = Ns[np.argmax([len(N1), len(N2)])]
                    n2_idx = np.where(N_2 % NKS == N_1[0] % NKS)
                    if len(n2_idx) == 0 or len(n2_idx[0]) == 0:
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
        
        # calc_k_bn = lambda p: (p % self.nks, p // self.nks + self.min_band )
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
            if compute_communities:
                self.logger.info(f'\n\t\tResolution: {resolution:.2f}, Iteration: {iteration + 1}')
                communities = nx.community.louvain_communities(self.GRAPH, resolution=resolution, weight='weight')
            else:
                communities = nx.connected_components(self.GRAPH) 

            sub_graphs = []
            for c in communities:
                sub_graph = self.GRAPH.subgraph(c)
                if sub_graph.number_of_nodes() <= self.nks * 1.3:
                    sub_graphs.append(sub_graph)
                    continue
                new_communities = nx.community.louvain_communities(sub_graph, resolution=resolution, weight='weight')
                for new_c in new_communities:
                    sub_graphs.append(sub_graph.subgraph(new_c))


            clusters, samples, N_g_nks = communites2clusters(sub_graphs)

            total_bands_computed = len(self.solved) + len(clusters)
            if not compute_communities or np.abs(10 - len(clusters) * len(samples)) < 10:
                break

            if total_bands_computed == self.nbnd and len(self.solved) >= 0 and N_g_nks == 0:
                flag_resolution = False
                break

            if len(clusters) * len(samples) and (best_iteration is None or np.abs(self.nbnd - total_bands_computed) < best_score and len(self.solved) > max_solved and N_g_nks < N_g_nks_prev):
                best_iteration = sub_graphs
                best_score = np.abs(self.nbnd - total_bands_computed)
                max_solved = len(self.solved)

            iteration += 1
            if iteration == max_iter:
                sub_graphs = best_iteration
                self.logger.info(f'\n\t\tBest Iteration:')
                clusters, samples, N_g_nks = communites2clusters(sub_graphs)
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
                    if len(sample.k_edges) == 0:
                        # Compute the edges
                        sample.calculate_pointsMatrix()
                        sample.calc_boundary()
                    scores[j_s] = sample.get_cluster_score(cluster,
                                                           self.min_band,
                                                           self.max_band,
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
        
            #self.n_process = 10 # min(self.n_process, len(samples))
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

    def obtain_output(self, last=False) -> None:
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
            # bn1 = d1 // self.nks + self.min_band                            # band
            bn1 = d1 // self.nks                                            # band
            k2 = d2 % self.nks                                              # k point
            # bn2 = d2 // self.nks + self.min_band                            # band
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
            dps = self.connections[k1, i_neigs, bn1, bn2]        # Dot-product between the k-point and their neighbors
            if np.any(np.logical_and(dps >= 0.5, dps <= 0.8)):
                # It is considered degenerate if the k-point has some 
                # neighbor's dot-product between 0.5 and 0.8
                self.degenerate_final.append([k1, bn1, bn2])                # Storage the degenerate k-point



        # Otherwise, the program continues to the next step. Here it finds the degenerate points
        k_basis_rotation : list[Tuple[Kpoint, Kpoint, Band, list[Band]]] = []           # Storage pairs of points that are degenerates by dot product 0.5 < <i|j> < 0.8
        for bn in range(self.total_bands):
            # Search these degenerate points on each band
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
                if np.any(np.logical_and(dps >= 0.5, dps <= 0.8)):
                    # It is considered degenerate if the k-point has some 
                    # neighbor's dot-product between 0.5 and 0.8
                    dps_deg = self.connections[k, i_neigs, bn_k]                                # All k-point dot-products
                    k = k[0]                                                                    # K-point
                    i_deg, bn_deg = np.where(np.logical_and(dps_deg >= 0.5, dps_deg <= 0.8))    # Find where the k-point dot-product is considered degenerate
                    k_deg = self.neighbors[k][i_deg + np.min(i_neigs)]                            # Identify the degenerate neighbors
                    i_sort = np.argsort(k_deg)                                                  # Sort the degenerate neighbors
                    k_deg = k_deg[i_sort]                                                       # Sort the degenerate neighbors
                    bn_deg = bn_deg[i_sort]                                                     # Sort the degenerate bands    
                    k_unique, index_unique = np.unique(k_deg, return_index=True)                # Identify unique neighbors
                    bn_unique = np.split(bn_deg, index_unique[1:])                              # Classify the bands by unique neighbor
                    len_bn = np.array([len(k_len) for k_len in bn_unique])                      # Look how many bands have each neighbor
                    if np.any(len_bn > 1):
                        # If for some neighbor exist more than one unique band,
                        # the k-point is degenerate
                        i_deg = np.where(len_bn > 1)[0]                                             # Obtain the neighbors
                        k_deg = k_unique[i_deg]                                                     # Obtain the neighbors
                        bns_deg = [bn_unique[j_deg] for j_deg in i_deg]                             # Get the unique bands
                        k_basis_rotation.append([k, k_deg, bn, bns_deg])                            # Append the information of the degenerate k-point
                score += np.mean(dps)                                                       # Update the band score
            score /= self.nks                                                               # Compute the mean socore
            self.final_score[bn] = score                                                    # Storage the band score

#TODO: degenerates usa o total banda bn
        degenerates = []
        for i, (k, k_deg, bn, bns_deg) in enumerate(k_basis_rotation[:-1]):
            # For each possible degenerate point have to exist a pair
            for k_, k_deg_, bn_, bns_deg_ in k_basis_rotation[i+1:]:
                # Comparison between each possible degenerate point.
                k_deg = np.array(k_deg)
                k_deg_ = np.array(k_deg_)
                if k != k_ or k_deg.shape != k_deg_.shape or not np.all(k_deg == k_deg_):
                    # The k_ point is not the k's pair
                    continue
                if not np.all([np.all(np.isin(bns, bns_deg_[j])) for j, bns in enumerate(bns_deg)]):
                    # If they do not belong to the same bands, The k_ point is not the k's pair
                    continue
                degenerates.append([k, bn, bn_])


        if not last:
            # If it is not the last iteration, the program ends here
            self.degenerate_final = np.array(self.degenerate_final)
            return

        analyzed = []                                                              # K-points analyzed
        for i, (k, bn, bn_) in enumerate(degenerates):
            # For each possible degenerate point stored inside k_basis_rotation.
            # There are only a few that are true degenerates; the other ones are their neighbors
            if i in analyzed:
                # The degenerates[i] point was already analyzed
                continue
            analyzed.append(i)
            # It is necessary to search group of points that are degenerate
            # These points are stored in same_group
            same_group = [[k, bn, bn_]]
            for j, (k_, bn0, bn1) in enumerate(degenerates[i+1:]):
                # Comparison between each possible pair of degenerate point.

                if not np.all(np.isin([bn, bn_], [bn0, bn1])):
                    # If they do not belong to the same bands, the analysis can not be done
                    continue

                ik =  self.kpoints_index[k, 0] if self.dimensions > 1 else self.kpoints_index[k]       # Obtain the matrix indices of k-space projection (k-point)
                ik_ = self.kpoints_index[k_, 0] if self.dimensions > 1 else self.kpoints_index[k_]     # Obtain the matrix indices of k-space projection (k_-point)

                idif = np.abs(ik - ik_)                                                                 # Manhattan distance i axes

                if idif > 1:
                    # If the total Manhattan distance is more than 2,
                    # the analysis can not be done
                    continue

                if self.dimensions >= 2:
                    jk = self.kpoints_index[k, 1]                                                       # Obtain the matrix indices of k-space projection (k-point)
                    jk_ = self.kpoints_index[k_, 1]                                                     # Obtain the matrix indices of k-space projection (k_-point)
                    jdif = np.abs(jk - jk_)                                                             # Manhattan distance j axes
                    if jdif > 1:
                        # If the total Manhattan distance is more than 2,
                        # the analysis can not be done
                        continue
                
                if self.dimensions == 3:
                    kk = self.kpoints_index[k, 2]                                                       # Obtain the matrix indices of k-space projection (k-point)
                    kk_ = self.kpoints_index[k_, 2]                                                     # Obtain the matrix indices of k-space projection (k_-point)
                    kdif = np.abs(kk - kk_)                                                             # Manhattan distance k axes
                    if kdif > 1:
                        # If the total Manhattan distance is more than 2,
                        # the analysis can not be done
                        continue

                analyzed.append(j+i+1)                                                      # The point k_ was analyzed
                same_group.append([k_, bn0, bn1])                                           # The k_-point belongs to the same group of k

            same_group = np.array(same_group)
            ks = same_group[:, 0]                                                           # K-points
            neighs = self.neighbors[ks]                                                     # K-points' neighbors
            points = [np.sum(neighs == k) for k in ks]                                      # How many points the k-point is neighbor?
            self.degenerate_final.append(same_group[np.argmax(points)])                     # There is only one degenerate point
        
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

        error_directions = []                                                           # This array stores the k-point where the energy ccontinuity fails.     ! It is not used !
        directions = []                                                                 # This array stores the direction where the energy continuity fails.    ! It is not used !

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
                                                self.matrix, self.signal_final,
                                                self.bands_final, self.eigenvalues,
                                                accept_E=self.accept_E, disp_scale=self.disp_scale)  # Obtain the new signal
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
            if signal == OTHER:
                # If the point was not marked as a correct or mistake signal, It is stored
                error_directions.append([k, bn])
                directions.append(scores)
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
        if last:
            for k, bn in zip(k_error, bn_error):
                signal = self.correct_signalfinal[k, bn]                                       # The k-point's signal
                if self.final_score[bn] > 0.96 and signal == MISTAKE:
                    self.correct_signalfinal[k, bn] = OTHER
                    self.signal_final[k, bn] = POTENTIAL_CORRECT

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

        # k_ot = k_other[other_same]                                                  # Store these repeated k-points
        # bn_ot = bn_other[other_same]                                                # Save their bands
        # not_same = np.logical_not(other_same)                                       # Identify which points are different
        # k_other = k_other[not_same]                                                 # The different k-points
        # bn_other = bn_other[not_same]                                               # Their bands

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

        mean_fitler = np.ones((3,3))                                                # It is the kernel used to select the problems' boundary
        self.GRAPH = nx.Graph()                                                     # The new Graph
        self.GRAPH.add_nodes_from(np.arange(len(self.vectors)))                     # Set the nodes
        edges = []
        for bn, band in enumerate(bands_signaling):
            # For each band construct the new graph
            # bn += self.min_band                                                                     # Initial band correction
            #if self.dimensions == 2 and np.sum(band) > self.nks*0.20:
                # If there are more than 5% of marked points, the boundaries of 
                # the problem are considered a problem too.
            #    identify_points = correlate(band, mean_fitler, output=None,
            #                                mode='reflect', cval=0.0, origin=0) > 0                 # The mean kernel is applied
            #else:
                # Otherwise, just the marked points are considered
            #    identify_points = band > 0
        
            identify_points = band > 0          # The marked points are considered
        
            if self.dimensions == 1:
                directions = np.array([1])                                                  # Auxiliary array with the directions to evaluate the edges' existence
                # If the problem is 1D, the graph is built by the neighbors
                for k, need_correction in enumerate(identify_points):
                    # For each not identified point the graph is built
                    kp = self.matrix[k]                                                         # The k-point on position k in k-space
                    if need_correction and kp not in self.degenerate_final:
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
                            p = kp + b_p*self.nks
                            pn = kneig + b_pn*self.nks
                            edges.append([p, pn])                                               # Establish an edge between nodes p (k-point) and pn (neighbor)

            if self.dimensions == 2:
                directions = np.array([[1, 0], [0, 1]])                                     # Auxiliary array with the directions to evaluate the edges' existence
                for ik, row in enumerate(identify_points):
                    for jk, need_correction in enumerate(row):
                        # For each not identified point the graph is built
                        kp = self.matrix[ik, jk]                                                        # The k-point on position ik, jk in k-space
                        if need_correction and kp not in self.degenerate_final:
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
                                p = kp + b_p*self.nks                                   # The kpoint's node id
                                pn = kneig + b_pn*self.nks                              # The neighbor's node id
                                edges.append([p, pn])                                                   # Establish an edge between nodes p (k-point) and pn (neighbor)

            if self.dimensions == 3:
                directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])                        # Auxiliary array with the directions to evaluate the edges' existence
                for ik, plane in enumerate(identify_points):
                    for jk, row in enumerate(plane):
                        for kk, need_correction in enumerate(row):
                            # For each not identified point the graph is built
                            kp = self.matrix[ik, jk, kk]                                                        # The k-point on position ik, jk in k-space
                            if need_correction and kp not in self.degenerate_final:
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
                                    p = kp + b_p*self.nks                                     # The kpoint's node id
                                    pn = kneig + b_pn*self.nks                                # The neighbor's node id
                                    edges.append([p, pn])                                                       # Establish an edge between nodes p (k-point) and pn (neighbor)

        self.correct_signalfinal_prev = np.copy(self.correct_signalfinal)                       # Save the currect result

        total_not_solved = np.sum(self.correct_signalfinal == NOT_SOLVED) + np.sum(self.correct_signalfinal == MISTAKE)
        self.logger.info(f'\n{BODY_INDENT}Total not solved: ' + str(total_not_solved) + '\n')
        self.total_not_solved = total_not_solved        # exposed so solve() can pace the alpha sweep by convergence

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
            for edge in edges:
                # For each edge, the graph is built
                p, pn = edge
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
            edges = np.array(edges)
            self.GRAPH.add_edges_from(edges)                                                        # Build the identified edges
            # self.correct_signalfinal[k_ot, bn_ot] = CORRECT-1                                       # Signaling as CORRECT the repeated k-points

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

        Only broken slots are (re)assigned: a slot already holding a distinct in-group
        band keeps its dp-chosen value; empty/duplicate/out-of-group slots are filled,
        preferring identity (slot == band, i.e. energy order). Filled cells are marked
        FORCED_CONTINUITY in ``correct_signalfinal`` so the band no longer counts as
        not-solved while the fills stay visible (FOR column) for audit.

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
                    if hi - lo + 1 > 2:
                        grown += 1
                    band_set = set(range(lo, hi + 1))                   # slots == bands owned by this group
                    seen, held = set(), set()
                    for s in range(lo, hi + 1):                         # keep already-correct in-group slots
                        v = int(bf[k, s])
                        if v in band_set and v not in seen:
                            seen.add(v); held.add(s)
                    miss = [bd for bd in range(lo, hi + 1) if bd not in seen]
                    free = [s for s in range(lo, hi + 1) if s not in held]
                    if miss:
                        miss_set = set(miss)
                        for s in [s for s in free if s in miss_set]:    # identity first (slot == band)
                            bf[k, s] = s; csf[k, s] = FORCED_CONTINUITY
                            miss_set.discard(s); free.remove(s); filled += 1
                            resolved.setdefault(s, []).append(k)
                        for s, bd in zip(free, sorted(miss_set)):       # pair any leftovers in energy order
                            bf[k, s] = bd; csf[k, s] = FORCED_CONTINUITY; filled += 1
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
                for v in dup_vals:
                    cs = where[v]
                    keep = max(cs, key=lambda c: (sig[k, c], -self._slot_energy_penalty(k, c, v)))
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
                                    self.matrix, self.signal_final, bf, self.eigenvalues,
                                    accept_E=self.accept_E, disp_scale=self.disp_scale)
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
        ethr = getattr(self, 'degen_ethr', 0.005)
        close = np.zeros(self.eigenvalues.shape, bool)              # (k, band) near-degenerate w/ a neighbour band
        dE = np.diff(self.eigenvalues, axis=1) < ethr
        close[:, 1:] |= dE
        close[:, :-1] |= dE
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
            if i >= ncols:
                return 0, np.empty(0, dtype=int)
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
            ncols = self.bands_final.shape[1]
            if i >= ncols:
                return self.final_score[i]
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

        header_line = ' Band | Failed | Degen | Benign | Susp | Score | Status'
        table = f'\n{TITLE_INDENT}====== Final Report ======\n'
        table += _format_legend(['Failed', 'Degen', 'Benign', 'Susp', 'Score', 'Status'])
        table += f'\n{BODY_INDENT}{header_line}'
        table += f'\n{BODY_INDENT}' + '-' * len(header_line)
        band_grade = []
        for i in range(self.nbnd):
            failed = int(report_a2[i, NOT_SOLVED] + report_a2[i, MISTAKE])
            degen = int(report_a2[i, DEGENERATE])
            forced = int(report_a2[i, FORCED_CONTINUITY])
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
            if failed > 0 or suspect > FORCED_FAIL_FRAC * self.nks:
                status = 'FAIL'
            elif suspect > 0:
                status = 'CHECK'                # force-fill across a real gap -> verify
            elif degen > 0:
                status = 'ROTATE'               # usable after basis rotation
            elif forced == 0 and score >= TOL_CLEAN:
                status = 'CLEAN'                # pristine, nothing flagged
            else:
                status = 'USABLE'               # nothing flagged (benign fills and/or score < clean)
            band_grade.append(status)
            table += (f'\n{BODY_INDENT} {i+self.min_band:<4d} | {failed:^6d} | {degen:^5d} | '
                      f'{benign:^6d} | {suspect:^4d} | {score:>5.2f} | {status}')
            if status in ('FAIL', 'CHECK'):
                bands_attention.append((i, failed, degen, benign, suspect, status))
        table += '\n'
        self.final_report += table

        ###########################################################################
        # The points that failed and why.
        ###########################################################################
        if bands_attention:
            self.final_report += f'\n{BODY_INDENT}Bands needing attention:'
            for i, failed, degen, benign, suspect, status in bands_attention:
                if failed > 0:
                    reason = f'{failed} failed (energy discontinuity / low overlap)'
                elif suspect > 0:
                    reason = f'{suspect} of {benign+suspect} force-fill(s) across a real energy gap'
                else:
                    reason = f'score {self.final_score[i]:.2f}'
                verdict = 'NOT usable' if status == 'FAIL' else 'verify before use'
                self.final_report += (f'\n{BODY_INDENT}  Band {i+self.min_band:>3d} : {reason} - {verdict}')
            self.final_report += '\n'

        p_report, problems = self.solved_problems_info

        self.final_report += '\n\t{:-^30}\n\t{: <30}\n\t{:-^30}\n'.format('', 'Summary:', '')
        self.final_report += p_report

        # point2k_bn = lambda p: (p % self.nks, p // self.nks + self.min_band)
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
            min_alpha : float
                The minimum alpha.
                (default 0)
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
        self.step = step
        self.alpha = alpha # The initial alpha is 1.0: alpha*<i|j> + (1-alpha)*f(E)
        self.init_alpha = alpha
        COUNT = 0     # Counter iteration
        bands_final_flag = True
        self.final_report = ''
        self.bands_final_prev = np.copy(self.bands_final)
        self.best_bands_final = np.copy(self.bands_final)
        self.best_score = np.zeros(self.total_bands, dtype=float)
        self.final_score = np.zeros(self.total_bands, dtype=float)
        self.signal_final = np.zeros((self.nks, self.total_bands), dtype=int)
        self.correct_signalfinal_prev = np.full(self.signal_final.shape, -1, int)
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
        while bands_final_flag and self.alpha >= min_alpha:
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

            # Verification if the result is similar to the previous one
            bands_final_flag = np.sum(np.abs(self.bands_final_prev - self.bands_final)) != 0
            self.bands_final_prev = np.copy(self.bands_final)

            # Verify and store the best result
            # To be a better result it has to be better score and all k points attributed for all the first max_solved bands
            solved = 0
            OTHER = 3

            total_not_solved_best = np.sum(self.correct_signalfinal_best == NOT_SOLVED)
            total_not_solved = np.sum(self.correct_signalfinal == NOT_SOLVED)

            total_best_score = np.sum(self.best_score)
            total_score = np.sum(self.final_score)

            first_max_bands_best_score = np.sum(self.best_score[:max_solved])
            first_max_bands_score = np.sum(self.final_score[:max_solved])
#TODO: ver best_score e bn
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
            total_solved_flag = first_max_bands_score >= first_max_bands_best_score and total_score > total_best_score and total_not_solved < total_not_solved_best
            total_solved_flag = total_solved_flag or (total_score/n_bands > 0.9 and total_best_score/n_bands < 0.9)
            if total_solved_flag or solved >= max_solved or COUNT == 1:
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
            if attempt_not_solved < self.alpha_best_ns:
                self.alpha_best_ns = attempt_not_solved
                self.alpha_stall = 0
            else:
                self.alpha_stall += 1
                self.logger.info(f'\n\t\tAlpha {self.alpha:.4f}: no improvement '
                                 f'({attempt_not_solved} vs best {self.alpha_best_ns}) -- '
                                 f'stall {self.alpha_stall}/{alpha_patience}')

            if self.alpha_stall >= alpha_patience:
                self.alpha -= step
                self.tol = max(self.tol * 0.90, 0.10)
                self.alpha_best_ns = np.inf
                self.alpha_stall = 0
                self.logger.info(f'\n\t\tAlpha converged -- descending to '
                                 f'{self.alpha:.4f} (tol={self.tol:.4f})')

        # The best result is maintained
        self.bands_final = np.copy(self.best_bands_final)
        self.final_score = np.copy(self.best_score)
        self.degenerate_final = np.copy(self.degenerate_best)
        self.signal_final = np.copy(self.best_signal_final)
        self.max_solved = max_solved

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
        # Wavefunction-evidence pass. The energy machinery above cannot see a label
        # swap between near-parallel bands (a few-mHa error passes every energy
        # gate), so first exchange the zero-dp-support cells where the swap is
        # dp-favoured (and re-grade the failed cells on the final canvas), then
        # demote whatever zero-support cells remain so the report cannot call a
        # silently swapped band clean.
        ###########################################################################
        self._repair_dp_swaps()
        self._flag_dp_unsupported()

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

    def get_cluster_score(self, cluster : COMPONENT, min_band : int, max_band : int,
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
            min_band : int
                It is an integer that gives the minimum band used for clustering.
            max_band : int
                It is an integer that gives the maximum band used for clustering.
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
            # jump beyond accept_E (a real inter-group wall can never be crossed).
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
                # bands = aux_bands + min_band
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
                # bands = aux_bands + min_band                                            # Initial band correction
                bands = aux_bands                                                       # Initial band correction
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
                # bands = aux_bands + min_band                                            # Initial band correction
                bands = aux_bands                                                       # Initial band correction
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
            # bn1 = self.bands_number[k] + min_band                                   # k-point's band
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
                # bn2 = cluster.bands_number[k_n]+min_band                            # neighbor's band
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
