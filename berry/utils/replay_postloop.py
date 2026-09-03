"""Offline replay of the cluster0 POST-LOOP pass chain on a saved final canvas.

Purpose: validate changes to the post-loop passes (patch relocation, min-cut
seam placement, certain-link repair, dp swaps) without re-running the full
in-loop solve.  The saved canvas (e.g. ``data/run5_backup``) is a FINAL
canvas -- the passes already ran on it once -- so the replay measures the
ADDITIONAL effect of the changed passes on the residue the old passes left.

Run FROM A RUN'S WORKING DIRECTORY (where ``data/`` lives -- loadmeta and
loaddata hard-code relative ``data/`` paths), on the machine holding the
arrays:

    python3 -m berry.utils.replay_postloop --canvas data/run5_backup

Approximations (documented, harmless for the replayed passes):
  * ``forced_mask`` is reconstructed as ``signal_final == FORCED``; the live
    ``_repair_certain_links`` can clear mask bits post-hoc, so a cell it
    un-forced in the ORIGINAL run is not recoverable here.  This only affects
    the FORCED re-grade branch of the relocation passes.
  * ``repaired_mask`` is left unset (every consumer is getattr-guarded).
  * solve-time bookkeeping that only decorates the report (degenerate-problem
    listing, max_solved) is stubbed empty.
"""

import argparse
import logging
import os
import sys

import numpy as np


def _census(material, tag: str) -> dict:
    """Print + return the seam/grade census of the current canvas."""
    from berry._subroutines.clustering_libs import (
        NOT_SOLVED, MISTAKE, POTENTIAL_MISTAKE, DEGENERATE)

    refuted = len(material._dp_refuted_seam_edges())
    walls = len(material._slot_wall_edges())
    cliffs = len(material._cliff_seam_edges()) if hasattr(material, '_cliff_seam_edges') else 0
    csf = material.correct_signalfinal
    failed = int(np.sum((csf == NOT_SOLVED) | (csf == MISTAKE)))
    # SILENT: validated cells (OTHER/CORRECT on the csf scale) sitting on a
    # gross energy jump -- same expression as report()'s _silent_mask.
    jumps = material._nn_energy_jump_map()
    silent = int(np.sum((material.bands_final >= 0) &
                        (np.nan_to_num(jumps, nan=0.0) > material.accept_E) &
                        np.isin(csf, (3, 4))))
    print(f'[{tag:>26s}]  refuted={refuted:5d}  walls={walls:5d}  cliffs={cliffs:5d}  '
          f'failed={failed:6d}  silent={silent:4d}')
    per_slot = [int(np.sum((csf[:, s] == NOT_SOLVED) | (csf[:, s] == MISTAKE)))
                for s in range(csf.shape[1])]
    bad = {s: n for s, n in enumerate(per_slot) if n}
    if bad:
        print(f'{"":29s}failed per slot: {bad}')
    return {'refuted': refuted, 'walls': walls, 'cliffs': cliffs,
            'failed': failed, 'silent': silent}


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Replay the cluster0 post-loop pass chain on a saved canvas.')
    ap.add_argument('--canvas', default='data/run5_backup',
                    help='directory holding the saved final arrays '
                         '(bandsfinal.npy, signalfinal.npy, correct_signalfinal.npy, '
                         'degeneratefinal.npy, final_score.npy)')
    ap.add_argument('--max-band', type=int, default=-1,
                    help='max band (default: m.final_band)')
    ap.add_argument('--npr', type=int, default=1, help='processes for make_vectors')
    ap.add_argument('--passes', default='walls,cliff,dp,mincut,degen,dpcycle,certain,swaps,flag',
                    help='comma list, chain order is fixed: '
                         'walls,cliff,dp,mincut,degen,dpcycle,certain,swaps,flag')
    ap.add_argument('--skip-greedy-dp', action='store_true',
                    help='A/B: drop the greedy dp-seam relocation so the min-cut '
                         'pass runs as its replacement')
    ap.add_argument('--save-out', default=None,
                    help='optional directory to save the modified arrays '
                         '(never writes into data/)')
    ap.add_argument('--report', action='store_true',
                    help='print the full final report table at the end')
    ap.add_argument('--sweeps', type=int, default=1,
                    help='iterate the relocation chain N times (solve() runs it to a '
                         'fixed point; >1 replays that, terminal swaps/flag run once after)')
    args = ap.parse_args()

    if not os.path.isdir('data'):
        sys.exit('No data/ directory here -- run from a run working directory.')
    if not os.path.isdir(args.canvas):
        sys.exit(f'Canvas directory not found: {args.canvas}')

    from berry import log
    from berry._subroutines.clustering_libs import MATERIAL, FORCED
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m

    logger = log('replay_postloop', 'REPLAY', level=logging.INFO)

    max_band = args.max_band if args.max_band != -1 else m.final_band
    min_band = m.initial_band
    connections = np.load(os.path.join(m.data_dir, 'dp.npy'))

    material = MATERIAL(m.dimensions, [m.nkx, m.nky, m.nkz], m.nbnd, m.nks,
                        d.eigenvalues, connections, d.neighbors, logger,
                        min_band, max_band=max_band, n_process=args.npr)
    # Recomputes matrix/kpoints_index and every energy scale (accept_E,
    # disp_scale, band_disp, local_gap, ...) deterministically from
    # eigenvalues + neighbors -- nothing canvas-dependent.
    material.make_vectors()

    def canvas(name):
        return np.load(os.path.join(args.canvas, name))

    material.bands_final = canvas('bandsfinal.npy')
    material.signal_final = canvas('signalfinal.npy')
    material.correct_signalfinal = canvas('correct_signalfinal.npy')
    material.degenerate_final = canvas('degeneratefinal.npy')
    material.final_score = canvas('final_score.npy')
    material.forced_mask = material.signal_final == FORCED   # approximation, see docstring
    material.final_report = ''
    material.degenerates = np.empty((0, 2), dtype=int)       # report stubs
    material.solved_problems_info = ['', []]
    material.max_solved = 0

    selected = [p.strip() for p in args.passes.split(',') if p.strip()]
    if args.skip_greedy_dp and 'dp' in selected:
        selected.remove('dp')

    chain = [
        ('walls',   lambda: material._relocate_wall_patches()),
        ('cliff',   lambda: material._relocate_cliff_patches()
                    if hasattr(material, '_relocate_cliff_patches') else
                    print('  (cliff pass not present in this code version -- skipped)')),
        ('dp',      lambda: material._relocate_dp_seam_patches()),
        ('mincut',  lambda: material._mincut_seam_relocate()
                    if hasattr(material, '_mincut_seam_relocate') else
                    print('  (mincut pass not present in this code version -- skipped)')),
        ('degen',   lambda: material._relocate_degenerate_patches()
                    if hasattr(material, '_relocate_degenerate_patches') else
                    print('  (degen pass not present in this code version -- skipped)')),
        ('dpcycle', lambda: material._repair_dp_cycles()
                    if hasattr(material, '_repair_dp_cycles') else
                    print('  (dpcycle pass not present in this code version -- skipped)')),
        ('certain', lambda: material._repair_certain_links()),
        ('swaps',   lambda: material._repair_dp_swaps()),
        ('flag',    lambda: material._flag_dp_unsupported()),
    ]

    base = _census(material, 'baseline')
    # solve() iterates the relocation chain to a fixed point (the passes settle in
    # a better optimum on the re-sweep from a cleaner canvas); --sweeps replays
    # that.  swaps/flag are terminal (they only touch zero-support cells), so they
    # run once after the sweeps, matching solve().
    loop_passes = [c for c in chain if c[0] not in ('swaps', 'flag')]
    tail_passes = [c for c in chain if c[0] in ('swaps', 'flag')]
    prev = None
    for sweep in range(max(1, args.sweeps)):
        for name, fn in loop_passes:
            if name not in selected:
                continue
            fn()
            if args.sweeps == 1:
                _census(material, f'post {name}')
        cur = _census(material, f'post sweep {sweep + 1}') if args.sweeps > 1 else None
        if cur is not None and prev is not None and cur['refuted'] + cur['cliffs'] \
                + cur['walls'] >= prev['refuted'] + prev['cliffs'] + prev['walls']:
            break
        prev = cur
    for name, fn in tail_passes:
        if name not in selected:
            continue
        fn()
        _census(material, f'post {name}')
    final = _census(material, 'final')

    print(f'\ndelta: refuted {base["refuted"]} -> {final["refuted"]}, '
          f'walls {base["walls"]} -> {final["walls"]}, '
          f'cliffs {base["cliffs"]} -> {final["cliffs"]}, '
          f'failed {base["failed"]} -> {final["failed"]}, '
          f'silent {base["silent"]} -> {final["silent"]}')

    if args.report:
        print(material.report())

    if args.save_out:
        os.makedirs(args.save_out, exist_ok=True)
        for name, arr in (('bandsfinal.npy', material.bands_final),
                          ('signalfinal.npy', material.signal_final),
                          ('correct_signalfinal.npy', material.correct_signalfinal),
                          ('degeneratefinal.npy', material.degenerate_final),
                          ('final_score.npy', material.final_score)):
            np.save(os.path.join(args.save_out, name), arr)
        print(f'modified arrays saved to {args.save_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
