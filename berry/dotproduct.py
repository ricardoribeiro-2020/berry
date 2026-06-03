from multiprocessing import Pool, Array
from typing import Tuple

import os
from time import time
import ctypes
import logging

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # graceful degradation if threadpoolctl is not installed
    threadpool_limits = None

from berry import log

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def _load_wfc(k: int, band: int, comp: str) -> np.ndarray:
    """Load a single wavefunction (one spinor component for the noncolinear
    case, ``comp=""`` otherwise), transparently handling compressed .npz."""
    wfc = np.load(os.path.join(m.wfcdirectory, f"k0{k}b0{band}{comp}.wfc"))
    if isinstance(wfc, np.lib.npyio.NpzFile):  # compressed wfc
        wfc = wfc['arr_0']
    return wfc


def _stack_block(k: int, bands, comp: str) -> np.ndarray:
    """Stack a block of bands of k-point ``k`` into a (n_block, nr) matrix."""
    return np.stack([_load_wfc(k, band, comp) for band in bands])


def dot(nk: int, j: int, neighbor: int, jNeighbor: int) -> None:
    """Dot products of Bloch factors between k-point ``nk`` and ``neighbor``
    for every pair of bands.

    The whole band x band loop is one matrix product
    ``dpc[b0, b1] = sum_k dphase[k] * wfc0[b0, k] * conj(wfc1[b1, k])`` which
    is dispatched to BLAS.  To keep memory bounded it is tiled in band blocks
    of ``BAND_BLOCK`` rows, so peak memory per worker is ~``2 * BAND_BLOCK *
    nr * ncomp * 16`` bytes regardless of the total number of bands.  COMPS is
    ``("",)`` in the colinear case and ``("-0", "-1")`` (the two spinor
    components, summed) in the noncolinear case.
    """
    start = time()

    dphase = d_phase[:, nk] * d_phase[:, neighbor].conj()
    nb = len(BANDS)

    # Outer loop: block of reference (nk) bands, loaded once and reused against
    # every neighbor block.  Inner loop: block of neighbor bands, streamed.
    for i0 in range(0, nb, BAND_BLOCK):
        ref_bands = BANDS[i0:i0 + BAND_BLOCK]
        A = [_stack_block(nk, ref_bands, c) for c in COMPS]          # ncomp x (bi, nr)

        for k0 in range(0, nb, BAND_BLOCK):
            nbr_bands = BANDS[k0:k0 + BAND_BLOCK]

            block = None
            for ci, c in enumerate(COMPS):
                # neighbor block pre-conjugated and phase-weighted: (bk, nr)
                B = _stack_block(neighbor, nbr_bands, c).conj() * dphase
                term = A[ci] @ B.T                                   # (bi, bk) BLAS gemm
                block = term if block is None else block + term

            b0, b1 = i0 + len(ref_bands), k0 + len(nbr_bands)
            dpc[nk, j, i0:b0, k0:b1] = block
            dpc[neighbor, jNeighbor, k0:b1, i0:b0] = block.conj().T

    logger.debug(f"\tFinished nk: {nk:>4}\tneighbor: {neighbor:>4}\tin: {(time() - start):>4.2f} seconds")


def get_point_neighbors(nk: int, j: int) -> None:
    """Generates the arguments for the dot function."""
    neighbor = d.neighbors[nk, j]
    if neighbor != -1 and neighbor > nk:
        # index of nk in the neighbor's own neighbor list (a plain int so the
        # dpc block assignment uses basic slicing rather than advanced indexing)
        jNeighbor = int(np.where(d.neighbors[neighbor] == nk)[0][0])

        return (nk, j, neighbor, jNeighbor)
    return None


def _init_worker(blas_threads: int) -> None:
    """Pool initializer: cap BLAS/OpenMP to ``blas_threads`` in each worker so
    the ``npr`` processes and their BLAS threads together stay at ~os.cpu_count()
    instead of every process spawning a full set of BLAS threads."""
    if threadpool_limits is not None:
        threadpool_limits(blas_threads)


def _available_ram() -> int:
    """Bytes of currently-available physical RAM (0 if it cannot be queried)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        try:  # POSIX fallback
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            return 0


def _auto_band_block(npr: int, bytes_per_band: int, nb: int) -> int:
    """Pick a band-block size so that ~2 blocks per worker fit in a fraction of
    the available RAM.  Falls back to ``nb`` (no tiling) when memory info is
    unavailable."""
    avail = _available_ram()
    if avail <= 0:
        return nb
    budget = 0.4 * avail / max(npr, 1)            # 40% of the per-worker share
    bs = int(budget // (2 * max(bytes_per_band, 1)))  # two blocks live at once
    return max(1, min(nb, bs))


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS for readable progress lines."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{h:02d}:{minutes:02d}:{sec:02d}"


def _dot_star(args: Tuple) -> None:
    """Unpack the argument tuple for use with ``Pool.imap_unordered`` (which,
    unlike ``starmap``, passes a single argument)."""
    dot(*args)

def run_dot(npr: int = 1, logger_name: str = "dot", logger_level: logging = logging.INFO, compress: bool = False, flush: bool = False, band_block: int = 0):
    global dpc, logger, d_phase, BANDS, COMPS, BAND_BLOCK
    logger = log(logger_name, "DOT PRODUCT", level=logger_level, flush=flush)

    if not 0 < npr <= os.cpu_count():
        raise ValueError(f"npr must be between 1 and {os.cpu_count()}")

    logger.header()


    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    DPC_SIZE = m.nks * 2 * m.dimensions * m.number_of_bands * m.number_of_bands
    DPC_SHAPE = (m.nks, 2 * m.dimensions, m.number_of_bands, m.number_of_bands)

    # Bands actually used, spinor components, and the RAM-bounded band-block size
    BANDS = list(range(m.initial_band, m.nbnd))
    COMPS = ("-0", "-1") if m.noncolin else ("",)
    bytes_per_band = m.nr * len(COMPS) * np.dtype(np.complex128).itemsize
    if band_block and band_block > 0:
        BAND_BLOCK = min(band_block, len(BANDS))
    else:
        BAND_BLOCK = _auto_band_block(npr, bytes_per_band, len(BANDS))

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tNumber of processors to use: {npr}")
    logger.info(f"\tTotal number of k-points: {m.nks}")
    logger.info(f"\tTotal number of points in real space: {m.nr}")
    logger.info(f"\tWill use bands from {m.initial_band} to {m.final_band}")
    logger.info(f"\tTotal number of bands to be used: {m.number_of_bands}")
    logger.info(f"\tBand-block size: {BAND_BLOCK} (~{2 * BAND_BLOCK * bytes_per_band / 1e9:.2f} GB/worker peak)\n")
    logger.info(f"\tDirectory where the wfc are: {m.wfcdirectory}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    logger.info(f"\tAllocating shared dpc array: shape {DPC_SHAPE} "
                f"(~{2 * DPC_SIZE * ctypes.sizeof(ctypes.c_double) / 1e9:.2f} GB)")
    dpc_base = Array(ctypes.c_double, 2 * DPC_SIZE, lock=False)
    dpc = np.frombuffer(dpc_base, dtype=np.complex128).reshape(DPC_SHAPE)
    dp = np.zeros(DPC_SHAPE, dtype=np.float64)
    logger.info(f"\tLoading phase array from {os.path.join(m.data_dir, 'phase.npy')}")
    d_phase = np.load(os.path.join(m.workdir, os.path.join(m.data_dir, "phase.npy")))

    ###########################################################################
    # 4. CALCULATE
    ###########################################################################
    logger.info("\n\tBuilding the list of neighbouring k-point pairs to process...")
    tasks = [
        args
        for nk in range(m.nks)
        for j in range(2 * m.dimensions)
        if (args := get_point_neighbors(nk, j)) is not None
    ]
    n_tasks = len(tasks)
    # Hand each worker a contiguous batch of tasks to cut per-task IPC overhead.
    chunksize = max(1, n_tasks // (npr * 4))
    # Share the cores between the npr processes: each gets cpu_count // npr BLAS
    # threads so process-level and BLAS-level parallelism don't oversubscribe.
    blas_threads = max(1, os.cpu_count() // npr)
    if threadpool_limits is None:
        logger.info("\tthreadpoolctl not installed: BLAS thread count is left at its "
                    "default and may oversubscribe the cores (pip install threadpoolctl).")
    else:
        logger.info(f"\tBLAS threads per worker: {blas_threads}")
    logger.info(f"\t{n_tasks} k-point pairs to process, "
                f"{len(BANDS)}x{len(BANDS)} band pairs each, "
                f"chunksize {chunksize}.\n")

    # Report progress roughly every 5% of the workload (at least every pair).
    report_every = max(1, n_tasks // 20)
    width = len(str(n_tasks))
    t0 = time()
    with Pool(npr, initializer=_init_worker, initargs=(blas_threads,)) as pool:
        # imap_unordered yields as each pair finishes, so we can track how many
        # are done and estimate the time remaining.
        for done, _ in enumerate(pool.imap_unordered(_dot_star, tasks, chunksize=chunksize), 1):
            if done % report_every == 0 or done == n_tasks:
                elapsed = time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (n_tasks - done) / rate if rate > 0 else 0.0
                logger.info(f"\t  {done:>{width}}/{n_tasks} pairs "
                            f"({100.0 * done / n_tasks:5.1f}%)  "
                            f"elapsed {_fmt_hms(elapsed)}  "
                            f"ETA {_fmt_hms(eta)}  "
                            f"({rate:.1f} pairs/s)")
    logger.info(f"\n\tAll {n_tasks} k-point pairs computed in {_fmt_hms(time() - t0)}.")

    logger.info("\tNormalizing dot products and computing modulus...")
    dpc /= m.nr         # To normalize the dot product
    dp = np.abs(dpc)    # Calculate the modulus of the dot product

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    if compress:
        np.savez_compressed(os.path.join(m.data_dir, "dpc.npz"), dpc)
        np.savez_compressed(os.path.join(m.data_dir, "dp.npz"), dp)
        logger.info(f"\n\tDot products saved to file dpc.npz")
        logger.info(f"\tDot products modulus saved to file dp.npz")
    else:
        np.save(os.path.join(m.data_dir, "dpc.npy"), dpc)
        np.save(os.path.join(m.data_dir, "dp.npy"), dp)
        logger.info(f"\n\tDot products saved to file dpc.npy")
        logger.info(f"\tDot products modulus saved to file dp.npy")

    ###########################################################################
    # Finished
    ###########################################################################
    logger.footer()

if __name__ == "__main__":
    run_dot(log("dotproduct", "DOT PRODUCT", "version"), 20)
