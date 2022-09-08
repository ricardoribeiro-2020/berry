import sys
import logging
import time
from headerfooter import header
from headerfooter import footer
from contatempo import tempo

class log:
    def __init__(self, program, title, version):
        self.program = program
        self.title = title
        self.version = version
        logging.basicConfig(filename=program+'.log',
                            filemode='w',
                            encoding='utf-8',
                            format='%(asctime)s   %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.DEBUG)
        self.logger = logging.getLogger(program)

        self.STARTTIME = time.time()

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s: %(message)s'))
        self.logger.addHandler(ch)
    
    def header(self):
        H = header(self.title, self.version, time.asctime())
        self.info(H)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        print(message)
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def footer(self):
        ENDTIME = time.time()
        F = footer(tempo(self.STARTTIME, ENDTIME))
        self.info(F)

    def percent_complete(self, step, total_steps, bar_width=60, title="", print_perc=True):
        '''
        author: WinEunuuchs2Unix
        url: https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage
        '''
        # UTF-8 left blocks: 1, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8
        utf_8s = ["█", "▏", "▎", "▍", "▌", "▋", "▊", "█"]
        perc = 100 * float(step) / float(total_steps)
        max_ticks = bar_width * 8
        num_ticks = int(round(perc / 100 * max_ticks))
        full_ticks = num_ticks / 8      # Number of full blocks
        part_ticks = num_ticks % 8      # Size of partial block (array index)
        
        disp = bar = ""                 # Blank out variables
        bar += utf_8s[0] * int(full_ticks)  # Add full blocks into Progress Bar
        
        # If part_ticks is zero, then no partial block, else append part char
        if part_ticks > 0:
            bar += utf_8s[part_ticks]
        
        # Pad Progress Bar with fill character
        bar += "▒" * int((max_ticks/8 - float(num_ticks)/8.0))
        
        if len(title) > 0:
            disp = title + ": "         # Optional title to progress display
        
        # Print progress bar in green: https://stackoverflow.com/a/21786287/6929343
        disp += "\x1b[0;32m"            # Color Green
        disp += bar                     # Progress bar to progress display
        disp += "\x1b[0m"               # Color Reset
        if print_perc:
            # If requested, append percentage complete to progress display
            if perc > 100.0:
                perc = 100.0            # Fix "100.04 %" rounding error
            disp += " {:6.2f}".format(perc) + " %"
        
        # Output to terminal repetitively over the same line using '\r'.
        sys.stdout.write("\r" + disp)
        sys.stdout.flush()