import os
from time import time
import logging
import threading
from math import prod

import numpy as np
from findiff import Gradient

from berry import log
from berry.shg import load_berry_connections

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def deriv(berryConnection, s, sprime, alpha1, alpha2, dk):
    """ Derivative of the Berry connection."""
    if m.dimensions == 1:
        grad = Gradient(h=[dk], acc=4)  # Defines gradient function in 1D
    elif m.dimensions == 2:
        grad = Gradient(h=[dk, dk], acc=4)  # Defines gradient function in 2D
    else:
        grad = Gradient(h=[dk, dk, dk], acc=4)  # Defines gradient function in 3D

    a = grad(berryConnection[s][sprime][alpha1])

    return a[alpha2]

def loadz(pathz, path, mmap_mode=None):
    if os.path.exists(pathz): # reading .npz compressed file if exists
        r = np.load(pathz, mmap_mode=mmap_mode)
        r = r['arr_0']
    else: # otherwise read .npy file
        r = np.load(path, mmap_mode=mmap_mode)

    return r


###############################################################################
# Streaming computation of the Berry connection and curvature
#
# The Berry connection A_ab = i<u_a|d_alpha u_b> and the curvature are sums
# over the real-space grid of products of per-band arrays of shape (nr, ...k).
# The per-pair RESULTS are only k-mesh sized, so instead of re-reading the
# (huge) per-band files once per band pair, a single pass reads a tile of
# nr-rows of ALL bands at once and accumulates every pair's contribution;
# each wfcpos file is read from disk exactly once.  The k-space gradient
# (formerly precomputed by r2k and stored as wfcgra files) is applied on the
# fly to each tile: it acts only along the k axes, so tiling nr does not
# change it (same findiff stencil as r2k, acc=2).  wfcgra files are no longer
# read (r2k does not write them by default anymore).
###############################################################################

def _kshape():
    if m.dimensions == 1:
        return (m.nkx,)
    if m.dimensions == 2:
        return (m.nkx, m.nky)
    return (m.nkx, m.nky, m.nkz)


CHERN_METHODS = ("chern", "chern_curl", "chern_bp", "chern_bp_bz")


def _parse_props(prop) -> set:
    """Expand the -prop value into the set of requested properties.

    Accepts a single token or a comma-separated list of tokens among
    'both', 'conn', 'curv' and the four Chern methods, plus the aliases
    'chern_all' (all four Chern methods) and 'all' (connection, curvature
    and all four Chern methods)."""
    valid = {"both", "conn", "curv"} | set(CHERN_METHODS)
    props = set()
    for token in str(prop).split(","):
        token = token.strip()
        if token == "all":
            props |= {"both"} | set(CHERN_METHODS)
        elif token == "chern_all":
            props |= set(CHERN_METHODS)
        elif token in valid:
            props.add(token)
        elif token:
            raise ValueError(f"invalid property {token!r}; choose from "
                             f"{sorted(valid)} or the aliases 'chern_all', 'all'")
    if not props:
        raise ValueError("no property requested")
    return props


def _geometry_files_complete(bands, kind: str, logger) -> bool:
    """True iff the berryConn ('conn') or berryCur ('curv') .npy file exists in
    m.geometry_dir for every pair of the requested bands (existence only; no
    shape or staleness check)."""
    prefix = {"conn": "berryConn", "curv": "berryCur"}[kind]
    missing = sum(1 for a in bands for b in bands
                  if not os.path.exists(os.path.join(m.geometry_dir, f"{prefix}{a}_{b}.npy")))
    if missing:
        logger.debug(f"\t{missing} of {len(bands)**2} {prefix} files missing "
                     f"for bands {bands[0]}-{bands[-1]}")
    return missing == 0


def _pos_file(band: int, comp: int = None):
    """Path of a wfcpos .npy file; refuses compressed runs (an .npz cannot be
    read in slices, which the streaming pass needs)."""
    name = f"wfcpos{band}" if comp is None else f"wfcpos{band}-{comp}"
    path = os.path.join(m.data_dir, name + ".npy")
    if not os.path.exists(path):
        if os.path.exists(os.path.join(m.data_dir, name + ".npz")):
            raise RuntimeError(
                f"{name}.npz is compressed (r2k was run with -c); the streaming "
                "geometry needs plain .npy files. Decompress it first, e.g.: "
                f"python3 -c \"import numpy as np; np.save('{name}.npy', "
                f"np.load('{name}.npz')['arr_0'])\" in {m.data_dir}."
            )
        raise FileNotFoundError(f"{path} not found; run r2k for this band first.")
    return path


def _pread_exact(fd: int, buf, offset: int) -> None:
    """Read len(buf) bytes at offset into buf (loops over short reads)."""
    view = memoryview(buf).cast("B")
    done, total = 0, len(view)
    while done < total:
        got = os.preadv(fd, [view[done:]], offset + done)
        if got <= 0:
            raise IOError(f"short read (fd {fd}, offset {offset + done})")
        done += got


class _PosStream:
    """Streams nr-tiles of all bands' wfcpos files with one prefetch thread.

    Yields, per tile, one (nk-flat, nbands, tile) complex array per spinor
    component (one for colinear, two for noncolinear) -- k-major, so the
    per-k batched matmuls and the k-axis gradient stencil act on it directly.
    Reads are positioned (os.preadv), sequential within each file, and each
    file is read once; the (tile, k) -> (k, tile) transpose happens in the
    prefetch thread, overlapped with the compute of the previous tile.
    """

    def __init__(self, bands, mem_gb: float, logger):
        self.bands = bands
        self.logger = logger
        self.kshape = _kshape()
        self.K = prod(self.kshape)
        self.ncomp = 2 if m.noncolin else 1
        N = len(bands)

        # validate the files and record data offsets
        pos_shape = (m.nr,) + self.kshape
        self.handles = []  # per component: list of (fd, data_offset)
        for comp in (range(2) if m.noncolin else (None,)):
            comp_handles = []
            for band in bands:
                path = _pos_file(band, comp)
                mm = np.lib.format.open_memmap(path, mode="r")
                if mm.shape != pos_shape or mm.dtype != np.complex128:
                    raise RuntimeError(f"{path}: expected {pos_shape} complex128, "
                                       f"got {mm.shape} {mm.dtype}")
                offset = mm.offset
                del mm
                comp_handles.append((os.open(path, os.O_RDONLY), offset))
            self.handles.append(comp_handles)

        # tile size from the memory budget: live tile-sized arrays are the two
        # ping-pong k-major buffers, the conjugated copy, and the gradient
        # (+ its conjugate for the curvature): ~(3 + 2*dim) tile-sized arrays
        row_bytes = self.ncomp * N * self.K * 16 * (3 + 2 * m.dimensions)
        self.tile = max(64, min(m.nr, int(mem_gb * 2**30 // max(1, row_bytes))))
        self.n_tiles = -(-m.nr // self.tile)

        # ping-pong buffers so each tile read reuses memory
        self._bufs = [np.empty((self.ncomp, self.K, N, self.tile), np.complex128)
                      for _ in range(2)]
        self._staging = np.empty((self.tile, self.K), np.complex128)

    def _read_tile(self, i_tile: int):
        r0 = i_tile * self.tile
        t = min(self.tile, m.nr - r0)
        buf = self._bufs[i_tile % 2]
        stag = self._staging
        row = self.K * 16
        for ci, comps in enumerate(self.handles):
            for ni, (fd, off) in enumerate(comps):
                _pread_exact(fd, stag[:t].reshape(-1), off + r0 * row)
                buf[ci, :, ni, :t] = stag[:t].T
        return [c[..., :t] for c in buf]

    def tiles(self):
        nxt = {"res": None, "err": None}

        def prefetch(i):
            try:
                nxt["res"] = self._read_tile(i)
            except BaseException as e:
                nxt["err"] = e

        th = threading.Thread(target=prefetch, args=(0,))
        th.start()
        for i in range(self.n_tiles):
            th.join()
            if nxt["err"] is not None:
                raise nxt["err"]
            cur = nxt["res"]
            if i + 1 < self.n_tiles:
                th = threading.Thread(target=prefetch, args=(i + 1,))
                th.start()
            yield i * self.tile, cur

    def close(self):
        for comps in self.handles:
            for fd, _ in comps:
                os.close(fd)


def _tile_gradient(pos_tile: np.ndarray, kshape) -> np.ndarray:
    """k-space gradient of one k-major tile (K-flat, N, t) -> (D, K, N, t).

    Same operator r2k used to build the wfcgra files -- findiff's first
    derivative with spacing m.step at acc=2: central stencil (f[i+1] -
    f[i-1])/2h inside, the one-sided 3-point acc=2 stencils at the mesh
    edges.  Applied here by direct slicing (findiff's generic machinery is
    ~10x slower on this layout).  The gradient only couples k points, so
    applying it to an nr-tile is exactly what r2k computed row by row.
    """
    K, N, t = pos_tile.shape
    gra = np.empty((m.dimensions, K, N, t), np.complex128)
    inv2h = 1.0 / (2.0 * m.step)
    src = pos_tile.reshape(*kshape, N, t)
    for axis in range(m.dimensions):
        f = np.moveaxis(src, axis, 0)
        g = np.moveaxis(gra[axis].reshape(*kshape, N, t), axis, 0)
        g[1:-1] = (f[2:] - f[:-2]) * inv2h
        g[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) * inv2h
        g[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) * inv2h
    return gra


def _broken_kpoint_masks(bands, enabled: bool, logger):
    """Per-band k-mesh masks of the wavefunction-discontinuity points to regularize.

    At a mislabelled seam the band assignment switches to a near-orthogonal state
    between neighbouring k-points, so the central finite-difference gradient
    (u(k+1) - u(k-1))/2h jumps by O(1) instead of O(h): the Berry
    connection/curvature spikes ~1/h there (a lattice image of a divergence).
    These points are a tiny, isolated minority, yet a divergent integrand can
    dominate a BZ sum, so they are interpolated over before the geometry files
    are written (see docs/berry_geometry_seam_regularization.md).

    The set of points is NOT recomputed here -- it is exactly the one the
    clustering signals: the endpoints of its dp-broken seam edges (the report's
    dpBrk column) on non-FAIL bands, written by the solve as ``dp_interp_mask.npy``
    (shape ``(nks, nslots)``, column ``a`` = band ``a``, the same index the
    wfcpos{a} files use; see MATERIAL.dp_interp_mask). Returns ``{band: mask}``
    with each mask of shape ``_kshape()``; an empty dict (with a warning) if the
    mask file is missing -- e.g. a run whose clustering predates this output, in
    which case re-run the clustering step to regenerate it.
    """
    if not enabled:
        return {}
    try:
        flag = np.load(os.path.join(m.data_dir, "dp_interp_mask.npy"))    # (nks,nslots) bool
        nktoijl = np.asarray(d.nktoijl)                                   # (nks,>=dim)
    except Exception as exc:
        logger.warning(f"\tSeam regularization skipped (missing dp_interp_mask.npy -- "
                       f"re-run the clustering step to regenerate it: {exc})")
        return {}
    nks = flag.shape[0]
    kshape = _kshape()
    dim = len(kshape)
    masks = {}
    total_flagged = 0
    for a in bands:
        if a >= flag.shape[1]:
            continue
        idx = np.where(flag[:, a])[0]
        mask = np.zeros(kshape, bool)
        if idx.size:
            ijl = nktoijl[idx]
            if dim == 1:
                mask[ijl[:, 0]] = True
            elif dim == 2:
                mask[ijl[:, 0], ijl[:, 1]] = True
            else:
                mask[ijl[:, 0], ijl[:, 1], ijl[:, 2]] = True
        masks[a] = mask
        total_flagged += int(mask.sum())
        if mask.any():
            logger.info(f"\t  band {a}: {int(mask.sum())} dp-broken seam k-point(s) "
                        f"({100 * mask.sum() / nks:.2f}%) flagged for regularization")
    logger.info(f"\tSeam regularization: {total_flagged} signalled (band, k) cell(s) "
                f"interpolated across bands {bands[0]}-{bands[-1]}")
    return masks


def _inpaint_periodic(arr: np.ndarray, mask: np.ndarray, max_iter: int = 16) -> np.ndarray:
    """Replace the masked k-mesh cells of ``arr`` by the mean of their unmasked
    mesh neighbours, iterating so pockets fill from the boundary inward. The
    mesh is a torus, so neighbours wrap (np.roll). ``arr`` has shape
    ``(*lead, *kshape)`` (e.g. (D, nkx, nky) for a connection component or
    (nkx, nky) for a scalar curvature); ``mask`` has shape ``kshape``. Non-mesh
    leading axes are carried along untouched. Returns a copy; ``arr`` unchanged.
    """
    if mask is None or not mask.any():
        return arr
    kdim = mask.ndim
    ksh = mask.shape
    lead = arr.shape[:arr.ndim - kdim]
    a = arr.reshape((-1,) + ksh).astype(arr.dtype, copy=True)   # (L, *ksh)
    known = ~mask
    for _ in range(max_iter):
        todo = mask & ~known
        if not todo.any():
            break
        num = np.zeros_like(a)
        den = np.zeros(ksh, dtype=float)
        for ax in range(kdim):
            for sh in (1, -1):
                kn = np.roll(known, sh, axis=ax)               # neighbour currently known?
                num += np.roll(a, sh, axis=ax + 1) * kn[None]  # +1: skip the L axis
                den += kn
        fill = todo & (den > 0)
        if not fill.any():
            break
        safe = np.where(den > 0, den, 1.0)
        avg = num / safe[None]
        a[:, fill] = avg[:, fill]
        known = known | fill
    return a.reshape(lead + ksh)


def _stream_connection_curvature(bands, do_conn: bool, do_curv: bool,
                                 mem_gb: float, logger,
                                 regularize: bool = True) -> None:
    """One pass over the wfcpos files; writes all berryConn / berryCur pairs.

    ``regularize`` interpolates the connection/curvature over the
    clustering-signalled dp-broken seam k-points (``dp_interp_mask.npy``) before
    writing -- the wavefunction-discontinuity points where the finite-difference
    gradient spikes (see ``_broken_kpoint_masks`` /
    docs/berry_geometry_seam_regularization.md). Set to ``False`` to disable.
    """
    N = len(bands)
    kshape = _kshape()
    K = prod(kshape)
    D = m.dimensions

    stream = _PosStream(bands, mem_gb, logger)
    logger.info(f"\tStreaming {stream.n_tiles} tile(s) of {stream.tile} r-points, "
                f"{N} bands, {stream.ncomp} spinor component(s)")

    # accumulators are k-mesh sized only: a few MB for every pair at once
    acc_conn = np.zeros((D, K, N, N), np.complex128) if do_conn else None
    # curvature needs M[d,d'][a,b,k] = sum_r grad_d u_a * conj(grad_d' u_b);
    # only one of each (d,d') pair is accumulated, the transpose-conjugate
    # gives the other (exact identity M[d',d] = conj(M[d,d']).swap(a,b))
    curv_pairs = [] if not do_curv else ([(0, 1)] if D == 2 else [(2, 1), (0, 2), (1, 0)])
    acc_M = {dd: np.zeros((K, N, N), np.complex128) for dd in curv_pairs}

    t_read = t_comp = 0.0
    t0 = time()
    for r0, comps in stream.tiles():
        t_read += time() - t0
        t0 = time()
        for pos in comps:  # one pass per spinor component, contributions add
            gra = _tile_gradient(pos, kshape)              # (D, K, N, t)
            if do_conn:
                # acc[d,k] += P_k^H (G_d)_k^T: batched (N,t)@(t,N) over k
                Pc = pos.conj()                            # (K, N, t)
                for dd in range(D):
                    acc_conn[dd] += Pc @ gra[dd].transpose(0, 2, 1)
            if curv_pairs:
                grac = gra.conj()
                for dd in curv_pairs:
                    acc_M[dd] += gra[dd[0]] @ grac[dd[1]].transpose(0, 2, 1)
        t_comp += time() - t0
        logger.debug(f"\ttile at r={r0} done (read {t_read:.1f}s, compute {t_comp:.1f}s)")
        t0 = time()
    stream.close()
    logger.info(f"\tstreaming pass finished: {t_read:.1f}s reading, {t_comp:.1f}s computing")

    # Seam regularization: the clustering-signalled dp-broken seam k-points
    # (dp_interp_mask.npy) spike the FD gradient. Interpolate over them below;
    # a pair (a, b) is regularized at every k where EITHER band is broken (both
    # are differentiated in the curvature, and grad u_b spikes the connection A_ab).
    masks = _broken_kpoint_masks(bands, regularize, logger) if (do_conn or do_curv) else {}

    def _pair_mask(a, b):
        ma = masks.get(bands[a]); mb = masks.get(bands[b])
        if ma is None and mb is None:
            return None
        if ma is None:
            return mb
        if mb is None:
            return ma
        return ma | mb

    if do_conn:
        ##  normalization convention: (1/nr) * sum_r |u|^2 = 1
        ##  (the .wfc files hold the periodic parts u_nk)
        conn = 1j * acc_conn.transpose(2, 3, 0, 1) / m.nr        # (N,N,D,K)
        # A is Hermitian in the band indices up to FD truncation and gauge
        # quality; the residual is a free diagnostic of both
        res = np.abs(conn - conn.conj().transpose(1, 0, 2, 3))
        worst = np.unravel_index(np.argmax(res), res.shape)
        logger.info(f"\tHermiticity residual max |A_ab - A_ba*| = {res.max():.3e} "
                    f"(pair {bands[worst[0]]},{bands[worst[1]]}; "
                    f"large values flag bad gauge/attribution k-points)")
        for a in range(N):
            for b in range(N):
                arr = conn[a, b].reshape((D,) + kshape)
                arr = _inpaint_periodic(arr, _pair_mask(a, b))
                np.save(os.path.join(m.geometry_dir, f"berryConn{bands[a]}_{bands[b]}.npy"), arr)
        logger.info(f"\tberryConn files written for bands {bands[0]}-{bands[-1]}"
                    + (" (seam-regularized)" if masks else ""))

    if do_curv:
        def kernel(M_ddp):  # 1j*(M[d,d'] - M[d',d]) / nr, M[d',d] from the identity
            return 1j * (M_ddp - M_ddp.conj().transpose(0, 2, 1)) / m.nr
        if D == 2:
            comps_std = [kernel(acc_M[(0, 1)])]                  # Omega_z
        else:
            comps_std = [kernel(acc_M[dd]) for dd in curv_pairs] # Omega_x,y,z
        # standard sign convention Omega = +curl(A): the kernels
        # above carry the old orientation; -conj converts, as before
        for a in range(N):
            for b in range(N):
                if D == 2:
                    # NOTE shape change: the scalar Omega_z(k) is saved as
                    # (nkx, nky) -- the old code broadcast it into (2, nkx, nky)
                    # with two identical planes, which double-counted in
                    # prop="chern" and misled vis into quiver-plotting it
                    bcr = -np.conj(comps_std[0][:, a, b]).reshape(kshape)
                else:
                    bcr = -np.conj(np.stack([c[:, a, b] for c in comps_std])).reshape((3,) + kshape)
                bcr = _inpaint_periodic(bcr, _pair_mask(a, b))
                np.save(os.path.join(m.geometry_dir, f"berryCur{bands[a]}_{bands[b]}.npy"), bcr)
        logger.info(f"\tberryCur files written for bands {bands[0]}-{bands[-1]}"
                    + (" (seam-regularized)" if masks else ""))


def chern_number(curv):
    """Flux of the Berry curvature through the k-mesh, divided by 2*pi.

    The mesh is square with spacing m.step in every direction (its actual
    construction), so the Riemann area element is step^2 -- NOT |b|/nk, which
    is only equal to step when the mesh happens to tile the BZ exactly.  The
    result is the Chern number when the mesh spans the full BZ; for a partial
    patch it is the (gauge-invariant) patch flux / 2*pi.

    3D: returns the three-component Chern vector.  C_alpha is a 2D integral
    over a slice perpendicular to axis alpha, averaged over the n_alpha
    slices.  (The old form summed each component over the whole 3D mesh times
    a single 1D spacing -- dimensionally inconsistent.)
    """
    if m.dimensions == 1:
        logger.warning("\tChern number is not defined for 1D materials; returning 0.")
        return 0
    if m.dimensions == 2:
        if curv.ndim == 3:  # legacy file: scalar duplicated into (2, nkx, nky)
            curv = curv[0]
        return np.sum(curv) * m.step**2 / (2 * np.pi)
    # 3D: per-slice average of the flux of each curvature component
    if curv.ndim == 5:      # legacy file: components duplicated into (3, 3, ...)
        curv = curv[:, 0]
    return np.array([np.sum(curv[0]) * m.step**2 / m.nkx,
                     np.sum(curv[1]) * m.step**2 / m.nky,
                     np.sum(curv[2]) * m.step**2 / m.nkz]) / (2 * np.pi)

def berry_phase(pos, x, y):
    """Gauge-invariant Berry phase of the counterclockwise plaquette at (x, y),
    with the standard sign: bp = +Omega * dk^2 in the continuum limit.

    The discrete link <u_k|u_{k+dk}> has angle -A.dk (since <u|du> = -iA with
    A = i<u|grad u>), so the counterclockwise product of THESE links gives
    -flux; the minus sign restores the standard orientation.  Verified against
    the gauge-invariant perturbation formula on a C=+1 model."""
    bp = (np.sum(pos[:, x, y].conj()     * pos[:, x+1, y])
       *  np.sum(pos[:, x+1, y].conj()   * pos[:, x+1, y+1])
       *  np.sum(pos[:, x+1, y+1].conj() * pos[:, x, y+1])
       *  np.sum(pos[:, x, y+1].conj()   * pos[:, x, y])) / (m.nr ** 4)

    bp = -np.angle(bp)

    return bp

def chern_number_bp(pos) -> np.complex128:
    """Fukui-Hatsugai-Suzuki lattice Chern number: sum of plaquette Berry
    phases / 2*pi.  Each plaquette phase is gauge invariant.  Note: the loops
    cover the open (nkx-1)x(nky-1) rectangle of plaquettes -- the wrap-around
    row/column is not included, so the total is exactly integer only when the
    mesh tiles the closed BZ torus (for a partial patch it is the patch flux)."""
    chern = 0
    if m.dimensions == 2:
        for i in range(m.nkx -1):
            for j in range(m.nky -1):
                chern += berry_phase(pos, i, j)
        chern /= (2*np.pi)
    else:
        logger.warning("\tchern_bp is only implemented for 2D materials; returning 0.")

    return chern

def chern_number_bp_bz(pos) -> np.complex128:
    """Berry phase of the counterclockwise Wilson loop around the mesh
    boundary, / 2*pi, with the standard sign (= +flux of Omega through the
    mesh, mod 1 -- Arg is only defined mod 2*pi, so this cannot distinguish C
    from C+-1; use chern_bp for the integer).  Each link is normalized to unit
    modulus before entering the product: the result stays fully gauge
    invariant (intermediate gauge phases cancel in the product) and the float
    underflow of a long product of |overlap|<1 factors is gone (the old code
    needed np.complex256 for that, which no longer exists in NumPy 2)."""
    chern = 0.0
    if m.dimensions == 2:
        def _link(z):
            return z / abs(z)
        prod_ = 1.0 + 0.0j
        for i in range(m.nkx-1):
            prod_ *= _link(np.sum(pos[:, i, 0].conj() * pos[:, i+1, 0]))
            prod_ *= _link(np.sum(pos[:, i, m.nky-1] * pos[:, i+1, m.nky-1].conj()))
        for j in range(m.nky-1):
            prod_ *= _link(np.sum(pos[:, m.nkx-1, j].conj() * pos[:, m.nkx-1, j+1]))
            prod_ *= _link(np.sum(pos[:, 0, j] * pos[:, 0, j+1].conj()))
        # links <u_k|u_next> carry angle -A.dk (see berry_phase): negate for
        # the standard orientation
        chern = -np.angle(prod_) / (2*np.pi)
    else:
        logger.warning("\tchern_bp_bz is only implemented for 2D materials; returning 0.")

    return chern


def berry_curvature_curl(idx: int, idx_: int, berry_connection, offset: int = 0) -> None:
    """
    Calculates the Berry curvature using the curl of Berry connections.

    ``idx``/``idx_`` are absolute band numbers (used in the output file name);
    ``offset`` is the first band held in ``berry_connection``, whose band axes
    are indexed relative to it (load_berry_connections shifts by initial_band).
    """
    if m.dimensions == 1:
        logger.warning("\tBerry curvature (curl) is not defined for 1D materials; skipping.")
        return

    s, s_ = idx - offset, idx_ - offset

    # deriv(bc, s, s', alpha1, alpha2, dk) = d_{alpha2} A_{alpha1}, so the curl
    # Omega_c = d_a A_b - d_b A_a is deriv(..., b, a, ...) - deriv(..., a, b, ...).
    # (The old code had TWO bugs here: the second term was a dangling expression
    # statement -- never subtracted -- and the term order gave -Omega.)
    if m.dimensions == 2:                # 2D case
        def aux_curvature() -> np.ndarray:
            """
            Auxiliary function to calculate the Berry curvature.
            Attention: this is valid for 2D and 3D materials.
            """

            bcr = (deriv(berry_connection, s, s_, 1, 0, m.step)
                   - deriv(berry_connection, s, s_, 0, 1, m.step))

            return bcr

    else:                                # 3D case
        def aux_curvature():
            """
            Auxiliary function to calculate the Berry curvature.
            Attention: this is valid for 2D and 3D materials.
            """

            bcr0 = (deriv(berry_connection, s, s_, 2, 1, m.step)
                    - deriv(berry_connection, s, s_, 1, 2, m.step))

            bcr1 = (deriv(berry_connection, s, s_, 0, 2, m.step)
                    - deriv(berry_connection, s, s_, 2, 0, m.step))

            bcr2 = (deriv(berry_connection, s, s_, 1, 0, m.step)
                    - deriv(berry_connection, s, s_, 0, 1, m.step))

            return bcr0, bcr1, bcr2

    start = time()

    bcr = aux_curvature() if m.dimensions == 2 else np.array(aux_curvature())
    logger.info(f"\tberry_curvature{idx}_{idx_}_curl calculated in {time() - start:.2f} seconds")

    np.save(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx_}_curl.npy"), bcr)

    return bcr


def run_berry_geometry(max_band: int, min_band: int = 0, npr: int = 0, prop: str = "both", digits: int = 0, mem_gb: float = 32.0, regularize: bool = True, force: bool = False, logger_name: str = "geometry", logger_level: int = logging.INFO, flush: bool = False):
    """``prop`` is a token or comma-separated list among 'both', 'conn', 'curv',
    'chern', 'chern_curl', 'chern_bp', 'chern_bp_bz', plus the aliases
    'chern_all' (the four Chern methods) and 'all' (everything).  The
    connection/curvature pass is skipped when all its output files already
    exist, unless ``force``; Chern methods compute their missing prerequisite
    files automatically."""
    global logger
    logger = log(logger_name, "BERRY GEOMETRY", level=logger_level, flush=flush)

    logger.header()

    props = _parse_props(prop)

    if npr <= 0:
        npr = os.cpu_count()

    ###########################################################################
    # 1. STDOUT THE PARAMETERS
    ###########################################################################
    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tProperties to calculate: {', '.join(sorted(props))}")
    logger.info(f"\tMinimum band: {min_band}")
    logger.info(f"\tMaximum band: {max_band}")
    logger.info(f"\tNumber of threads: {npr}")
    logger.info(f"\tMemory budget: {mem_gb} GB")
    logger.info(f"\tSeam regularization: {'on (dp-broken seams)' if regularize else 'off'}")
    logger.info(f"\tForce recomputation: {force}\n")
    logger.info(f"\t{m.dimensions} dimensions calculation.\n")

    ###########################################################################
    # 2. CALCULATE BERRY GEOMETRY
    ###########################################################################
    bands = list(range(min_band, max_band + 1))
    want_conn = bool(props & {"both", "conn"})
    want_curv = bool(props & {"both", "curv"}) and m.dimensions > 1
    if props & {"both", "curv"} and m.dimensions == 1:
        logger.info(f"\tBerry curvature is not defined for 1D materials.")

    conn_done = _geometry_files_complete(bands, "conn", logger)
    curv_done = _geometry_files_complete(bands, "curv", logger)
    # prerequisites: 'chern' integrates the berryCur files, 'chern_curl'
    # differentiates the berryConn files -- compute whichever is missing
    need_conn = "chern_curl" in props and not conn_done
    need_curv = "chern" in props and not curv_done and m.dimensions > 1
    do_conn = need_conn or (want_conn and (force or not conn_done))
    do_curv = need_curv or (want_curv and (force or not curv_done))
    if want_conn and not do_conn:
        logger.info(f"\tberryConn files for bands {bands[0]}-{bands[-1]} already "
                    f"present; skipping (use -force to recompute)")
    if want_curv and not do_curv:
        logger.info(f"\tberryCur files for bands {bands[0]}-{bands[-1]} already "
                    f"present; skipping (use -force to recompute)")

    if do_conn or do_curv:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            from contextlib import nullcontext
            threadpool_limits = lambda limits: nullcontext()
        start = time()
        with threadpool_limits(limits=npr):
            _stream_connection_curvature(bands, do_conn, do_curv, mem_gb, logger,
                                         regularize=regularize)
        logger.info(f"\tconnection/curvature pass took {time() - start:.2f} seconds")

    def _new_chern_array():
        # 2D: one Chern number per band; 3D: a three-component Chern vector per band
        if m.dimensions == 3:
            return np.zeros((max_band + 1, 3), dtype=np.complex128)
        return np.zeros((max_band + 1), dtype=np.complex128)

    def _log_chern_numbers(method, values):
        logger.info(f"\n\t{method}:")
        logger.info(f"{'Band:' : >13} Chern Number")
        for idx in bands:
            logger.info(f"{f'{idx}:' : >13} {values[idx]}")

    if m.dimensions > 1:
        if "chern" in props:
            chern_num = _new_chern_array()
            for idx in bands:
                curv = np.load(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx}.npy"))
                chern_num[idx] = chern_number(curv)

            # save the chern number
            chern_num = np.round(np.real(chern_num), decimals=digits)
            np.save(os.path.join(m.geometry_dir, "chern_number.npy"), chern_num)
            logger.info(f"\tchern_number.npy saved")
            _log_chern_numbers("chern", chern_num)

        if "chern_curl" in props:
            chern_num = _new_chern_array()
            number_of_bands = max_band + 1 - min_band
            if m.dimensions == 2:
                berry_conn_size  = m.dimensions * m.nkx * m.nky * (number_of_bands) ** 2
                berry_conn_shape = (number_of_bands, number_of_bands, m.dimensions, m.nkx, m.nky)
            else:
                berry_conn_size  = m.dimensions * m.nkx * m.nky * m.nkz * (number_of_bands) ** 2
                berry_conn_shape = (number_of_bands, number_of_bands, m.dimensions, m.nkx, m.nky, m.nkz)
            # (the old call passed the arguments in a stale order -- crashed
            # since load_berry_connections moved to (initial, conduction, ...))
            berry_connections = load_berry_connections(min_band, max_band, berry_conn_size, berry_conn_shape)
            for idx in bands:
                curv = berry_curvature_curl(idx, idx, berry_connections, offset=min_band)
                chern_num[idx] = chern_number(curv)
            np.save(os.path.join(m.geometry_dir, "chern_number_curl.npy"), chern_num)
            logger.info(f"\tchern_number_curl.npy saved")
            _log_chern_numbers("chern_curl", np.real(chern_num))

        if props & {"chern_bp", "chern_bp_bz"}:
            # both methods read the same (large) wfcpos files: load once per band
            chern_bp_num = _new_chern_array() if "chern_bp" in props else None
            chern_bz_num = _new_chern_array() if "chern_bp_bz" in props else None
            for idx in bands:
                pos = loadz(os.path.join(m.data_dir, f"wfcpos{idx}.npz"), os.path.join(m.data_dir, f"wfcpos{idx}.npy"))
                if chern_bp_num is not None:
                    chern_bp_num[idx] = chern_number_bp(pos)
                if chern_bz_num is not None:
                    chern_bz_num[idx] = chern_number_bp_bz(pos)
            if chern_bp_num is not None:
                np.save(os.path.join(m.geometry_dir, "chern_number_bp.npy"), chern_bp_num)
                logger.info(f"\tchern_number_bp.npy saved")
                _log_chern_numbers("chern_bp", np.real(chern_bp_num))
            if chern_bz_num is not None:
                np.save(os.path.join(m.geometry_dir, "chern_number_bp_bz.npy"), chern_bz_num)
                logger.info(f"\tchern_number_bp_bz.npy saved")
                _log_chern_numbers("chern_bp_bz", np.real(chern_bz_num))

    ###########################################################################
    # Finished
    ###########################################################################

    logger.footer()

if __name__ == "__main__":
    run_berry_geometry(9, prop="both")
