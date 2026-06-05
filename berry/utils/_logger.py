from pathlib import Path

import time
import logging

from berry import __version__
from berry._subroutines.headerfooter import header, footer
from berry._subroutines.contatempo import tempo

def prepare_message(method):
    def wrapper(ref, *messages):
        message = ''
        if len(messages) == 0:
            method(ref, message)
            return
        for m in messages:
            message += str(m) + ' '
        method(ref, message)
    return wrapper

class log:
    def __init__(self, program, title, level=logging.INFO, flush: bool = False):
        Path(program).parent.mkdir(parents=True, exist_ok=True)

        self.program = program
        self.title = title
        self.version = __version__
        self.level = level
        self.flush = flush
        logging.basicConfig(filename='log/'+program+'.log',
                            filemode='w',
#                            encoding='utf-8',
                            format='%(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=level)
        self.logger = logging.getLogger(program)

        self.STARTTIME = time.time()
        self._progress = {}   # per-title state for percent_complete ETA logging

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)
    
    def header(self):
        H = header(self.title, self.version, time.asctime())
        self.info(H)

    @prepare_message
    def debug(self, message):
        if self.flush:
            print(message, flush=True)
        self.logger.debug(message)

    @prepare_message
    def info(self, message):
        if self.flush:
            print(message, flush=True)
        self.logger.info(message)

    @prepare_message
    def error(self, message):
        if self.flush:
            print(message, flush=True)
        self.logger.error(message)

    @prepare_message
    def warning(self, message):
        if self.flush:
            print(message, flush=True)
        self.logger.warning(message)
    
    def footer(self):
        ENDTIME = time.time()
        F = footer(tempo(self.STARTTIME, ENDTIME))
        self.info(F)

    def percent_complete(self, step, total_steps, bar_width=60, title="", print_perc=True, min_interval=10.0):
        '''
        Log a progress line with elapsed time and an estimated time remaining (ETA),
        instead of drawing a terminal progress bar.

        Per `title`, intermediate updates are rate-limited to at most one every
        `min_interval` seconds; the first update for a title and the final update
        (step >= total_steps) always log.  The ETA is a linear extrapolation from the
        progress made since this title started.

        `bar_width` and `print_perc` are accepted for call-site compatibility and are
        unused.
        '''
        if total_steps is None or total_steps <= 0:
            return

        def _fmt(seconds):
            seconds = int(max(0, seconds))
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{h:d}:{m:02d}:{s:02d}"

        now = time.time()
        key = title or '_'
        state = self._progress.get(key)
        # (Re)start tracking on a new title or when the counter restarts.
        if state is None or step < state['last_step']:
            state = {'start_time': now, 'start_step': step, 'last_emit': 0.0, 'last_step': step}
            self._progress[key] = state
        state['last_step'] = step

        perc = min(100.0, 100.0 * float(step) / float(total_steps))
        is_first = step <= state['start_step']
        is_last = step >= total_steps
        if not (is_first or is_last) and (now - state['last_emit'] < min_interval):
            return
        state['last_emit'] = now

        elapsed = now - state['start_time']
        done_since = step - state['start_step']
        msg = f"{title + ': ' if title else ''}{perc:6.2f}% ({step}/{total_steps}) | elapsed {_fmt(elapsed)}"
        if is_last:
            msg += " | done"
            self._progress.pop(key, None)
        elif done_since > 0:
            remaining = elapsed * (total_steps - step) / done_since
            msg += f" | ETA {_fmt(remaining)}"
        self.info(msg)