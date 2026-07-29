from multiprocessing import Pool
from typing import Tuple

import os
import json
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
    """Dot products of the periodic parts u_nk between k-point ``nk`` and
    ``neighbor`` for every pair of bands.

    The .wfc files hold u_nk(r) (the periodic part; wfck2r.x output), so the
    overlap is computed directly between them with NO Bloch-phase weighting:
    ``dpc[b0, b1] = sum_r wfc0[b0, r] * conj(wfc1[b1, r])`` = <u_b1(k')|u_b0(k)>.

    The whole band x band loop is one matrix product dispatched to BLAS.  To
    keep memory bounded it is tiled in band blocks of ``BAND_BLOCK`` rows, so
    peak memory per worker is ~``2 * BAND_BLOCK * nr * ncomp * 16`` bytes
    regardless of the total number of bands.  COMPS is ``("",)`` in the
    colinear case and ``("-0", "-1")`` (the two spinor components, summed) in
    the noncolinear case.
    """
    start = time()

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
                # neighbor block pre-conjugated: (bk, nr)
                B = _stack_block(neighbor, nbr_bands, c).conj()
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


def _dot_star(args: Tuple) -> int:
    """Unpack the argument tuple for use with ``Pool.imap_unordered`` (which,
    unlike ``starmap``, passes a single argument) and return the task index so
    the parent can mark it done in the checkpoint mask."""
    idx, dot_args = args
    dot(*dot_args)
    return idx


CKPT_INTERVAL = 600  # seconds between checkpoint flushes
PROGRESS_INTERVAL = 30  # seconds between progress/ETA log lines


def _flush_logfile() -> None:
    """Force buffered log records out to the .log file so the progress/ETA lines
    show up live during the run rather than only at the end."""
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass


def _ckpt_paths() -> Tuple[str, str, str]:
    """Paths of the three checkpoint files, alongside the final dpc.npy."""
    base = m.data_dir
    return (os.path.join(base, "dpc.ckpt.npy"),
            os.path.join(base, "dpc.ckpt.mask.npy"),
            os.path.join(base, "dpc.ckpt.meta.json"))


def _save_mask(path: str, done: np.ndarray) -> None:
    """Atomically persist the per-task done-mask (write to .tmp, then replace)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, done)
    os.replace(tmp, path)


def _cleanup_ckpt(*paths: str) -> None:
    """Remove checkpoint files (and any leftover .tmp) ignoring missing ones."""
    for p in paths:
        for q in (p, p + ".tmp"):
            try:
                os.remove(q)
            except OSError:
                pass

def run_dot(npr: int = 1, logger_name: str = "dot", logger_level: logging = logging.INFO, compress: bool = False, flush: bool = False, band_block: int = 0):
    global dpc, logger, BANDS, COMPS, BAND_BLOCK
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
    # 3. BUILD THE TASK LIST
    ###########################################################################
    logger.info("\n\tBuilding the list of neighbouring k-point pairs to process...")
    tasks = [
        args
        for nk in range(m.nks)
        for j in range(2 * m.dimensions)
        if (args := get_point_neighbors(nk, j)) is not None
    ]
    n_tasks = len(tasks)

    ###########################################################################
    # 4. CHECKPOINT / RESUME SETUP
    ###########################################################################
    # The full dpc result lives in a memmapped .npy, so every worker write is
    # persisted to disk as it happens.  A tiny per-task done-mask records which
    # pairs are finished, and a meta file ties the checkpoint to this exact run
    # so a stale checkpoint from a different calculation is never reused.
    ckpt_path, mask_path, meta_path = _ckpt_paths()
    meta = {
        "refname": str(m.refname),
        "shape": list(DPC_SHAPE),
        "noncolin": bool(m.noncolin),
        "n_tasks": n_tasks,
    }
    logger.info(f"\tResult array dpc: shape {DPC_SHAPE} "
                f"(~{2 * DPC_SIZE * ctypes.sizeof(ctypes.c_double) / 1e9:.2f} GB), "
                f"memmapped to {ckpt_path}")

    resume = all(os.path.exists(p) for p in (ckpt_path, mask_path, meta_path))
    if resume:
        try:
            with open(meta_path) as fh:
                saved_meta = json.load(fh)
        except (OSError, ValueError):
            saved_meta = None
        if saved_meta != meta:
            resume = False
            logger.info("\tFound a checkpoint from a different run; ignoring it and starting fresh.")

    if resume:
        dpc = np.lib.format.open_memmap(ckpt_path, mode="r+")
        done = np.load(mask_path)
        if dpc.shape != DPC_SHAPE or done.shape[0] != n_tasks:
            resume = False           # header/mask inconsistent: rebuild from scratch

    if resume:
        n_already = int(done.sum())
        logger.info(f"\tResuming from checkpoint: {n_already}/{n_tasks} pairs already done, "
                    f"{n_tasks - n_already} to compute.")
    else:
        dpc = np.lib.format.open_memmap(ckpt_path, mode="w+",
                                        dtype=np.complex128, shape=DPC_SHAPE)
        done = np.zeros(n_tasks, dtype=bool)
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        _save_mask(mask_path, done)
        n_already = 0

    ###########################################################################
    # 5. CALCULATE
    ###########################################################################
    # Only the pairs not already in the checkpoint are dispatched; each carries
    # its index in ``tasks`` so the worker can report it back for the done-mask.
    indexed_tasks = [(i, t) for i, t in enumerate(tasks) if not done[i]]
    n_remaining = len(indexed_tasks)
    # Keep the chunks small (aim for a few hundred per run) so imap_unordered
    # yields results steadily and the progress/ETA lines refresh throughout the
    # run instead of only at chunk boundaries.  The per-task payload is a few
    # ints and the per-pair work is BLAS-heavy, so the extra IPC is negligible.
    chunksize = max(1, n_remaining // 500)
    # Share the cores between the npr processes: each gets cpu_count // npr BLAS
    # threads so process-level and BLAS-level parallelism don't oversubscribe.
    blas_threads = max(1, os.cpu_count() // npr)
    if threadpool_limits is None:
        logger.info("\tthreadpoolctl not installed: BLAS thread count is left at its "
                    "default and may oversubscribe the cores (pip install threadpoolctl).")
    else:
        logger.info(f"\tBLAS threads per worker: {blas_threads}")
    logger.info(f"\t{n_remaining} k-point pairs to process this run, "
                f"{len(BANDS)}x{len(BANDS)} band pairs each, "
                f"chunksize {chunksize}.\n")

    width = len(str(n_tasks))
    t0 = time()
    last_ckpt = t0
    last_log = t0
    if n_remaining:
        with Pool(npr, initializer=_init_worker, initargs=(blas_threads,)) as pool:
            # imap_unordered yields each task's index as it finishes, so we can
            # mark the done-mask, checkpoint periodically and estimate the ETA.
            processed = 0
            for idx in pool.imap_unordered(_dot_star, indexed_tasks, chunksize=chunksize):
                done[idx] = True
                processed += 1
                done_count = n_already + processed
                now = time()
                if now - last_ckpt >= CKPT_INTERVAL:
                    dpc.flush()
                    _save_mask(mask_path, done)
                    last_ckpt = now
                # Log a progress/ETA line at a steady interval (and on the last
                # pair), flushing it straight to the .log file.
                if now - last_log >= PROGRESS_INTERVAL or done_count == n_tasks:
                    elapsed = now - t0
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    eta = (n_remaining - processed) / rate if rate > 0 else 0.0
                    logger.info(f"\t  {done_count:>{width}}/{n_tasks} pairs "
                                f"({100.0 * done_count / n_tasks:5.1f}%)  "
                                f"elapsed {_fmt_hms(elapsed)}  "
                                f"ETA {_fmt_hms(eta)}  "
                                f"({rate:.1f} pairs/s)")
                    _flush_logfile()
                    last_log = now
        dpc.flush()
        _save_mask(mask_path, done)
        logger.info(f"\n\tAll {n_tasks} k-point pairs computed in {_fmt_hms(time() - t0)}.")
        _flush_logfile()
    else:
        logger.info("\tAll k-point pairs were already computed in a previous run; "
                    "reusing the checkpoint.")

    ###########################################################################
    # 6. NORMALIZE AND SAVE OUTPUT
    ###########################################################################
    # Normalize on a RAM copy so the on-disk checkpoint stays the raw (un-normalized)
    # dot products: if a crash happens here, a resume reloads raw data and
    # normalizes exactly once.
    logger.info("\tNormalizing dot products and computing modulus...")
    dpc_final = np.array(dpc, dtype=np.complex128)
    dpc_final /= m.nr          # To normalize the dot product
    dp = np.abs(dpc_final)     # Calculate the modulus of the dot product

    if compress:
        np.savez_compressed(os.path.join(m.data_dir, "dpc.npz"), dpc_final)
        np.savez_compressed(os.path.join(m.data_dir, "dp.npz"), dp)
        logger.info(f"\n\tDot products saved to file dpc.npz")
        logger.info(f"\tDot products modulus saved to file dp.npz")
    else:
        np.save(os.path.join(m.data_dir, "dpc.npy"), dpc_final)
        np.save(os.path.join(m.data_dir, "dp.npy"), dp)
        logger.info(f"\n\tDot products saved to file dpc.npy")
        logger.info(f"\tDot products modulus saved to file dp.npy")

    ###########################################################################
    # 7. REMOVE CHECKPOINT
    ###########################################################################
    del dpc                    # close the memmap before deleting its backing file
    _cleanup_ckpt(ckpt_path, mask_path, meta_path)

    ###########################################################################
    # Finished
    ###########################################################################
    logger.footer()

if __name__ == "__main__":
    run_dot(log("dotproduct", "DOT PRODUCT", "version"), 20)
