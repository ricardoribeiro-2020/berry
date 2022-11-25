from typing import Callable, Dict, Any

import os
import re
import sys
import logging
import subprocess
import argparse, argcomplete

from _version import __version__

#TODO: Talk about np.savez
#NOTE: np.savez could help with backwards compatibility 


class CustomParser(argparse.ArgumentParser):
    def _check_value(self, action, value):
        if isinstance(action.choices, dict):
            if 'preprocess' in action.choices.keys() and value not in action.choices.keys():
                msg = f"""invalid program choice: {value}.
This error probably means you are trying to run a program in the incorrect order and therefore do not have the required files.
Try the program in the following order: 'preprocess', 'wfcgen', 'dot', 'cluster', 'r2k', 'geometry', 'condutivity', 'shg'."""
                raise argparse.ArgumentTypeError(msg)
            elif 'both' in action.choices.keys() and value not in action.choices.keys():
                msg = f"""invalid program choice in geometry program: {value}. Please choose from the following: {action.choices.keys()}"""

        elif isinstance(action.choices, range) and action.choices is not None and value not in action.choices:
            first, last = action.choices[0], action.choices[-1]
            msg = f"invalid choice: {value}. Choose from {first} up to (and including) {last}."
            raise argparse.ArgumentError(action, msg)
        else:
            super()._check_value(action, value)

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


##################################################################################################
# MAIN PROGRAMS
##################################################################################################
WFCGEN = DOT = CLUSTER = BASIS = R2K = GEOMETRY = CONDUCTIVITY = SHG = 0
try:
    import berry._subroutines.loaddata as d

    # NOTE: Changed 'd.workdir' to 'os.getcwd()' because one might not want to run from the start
    d.workdir = os.getcwd()

    if os.path.exists(os.path.join(d.workdir, "datafile.npy")):
        WFCGEN = 1
    if os.path.exists(os.path.join(d.workdir, "wfc")):
        DOT = 1
    if os.path.exists(os.path.join(d.workdir, "dpc.npy")):
        CLUSTER = 1
    if os.path.exists(os.path.join(d.workdir, "signalfinal.npy")):
        BASIS = 1
    if os.path.exists(os.path.join(d.workdir, "final.report")):
        R2K = 1
    if os.path.exists(os.path.join(d.workdir, "wfcgra0.npy")):
        GEOMETRY = 1
    if os.path.exists(os.path.join(d.workdir, "berryConn0_0.npy")):
        CONDUCTIVITY = 1
        SHG = 1
except:
    pass

def master_cli():
    ###########################################################################
    # 1. DEFINING MASTER CLI ARGS
    ###########################################################################
    try:
        parser = CustomParser(formatter_class=argparse.RawTextHelpFormatter,
        description="""This is the Master Command Line Interface (CLI) for the berry suite.
The berry suite extracts the Bloch wavefunctions from DFT calculations in an ordered way. 
This CLI is an interface to run the different scripts of the berry suite.
The scripts have to be run in order because each script uses results from the previous ones.
Command line is of the form:

berry [package options] script parameter [script options]
""")
        parser.add_argument("--version", action="store_true", help="Displays current Berry version.")
        parser.add_argument("--enable-autocomplete", action="store_true", help="Enables autocomplete for the berry CLI.")
        parser.add_argument("--disable-autocomplete", action="store_true", help="Disables autocomplete for the berry CLI.")

        sub_parser = parser.add_subparsers(dest="program", help="Choose the program to run.")

        preprocess_parser = sub_parser.add_parser("preprocess", help="Run and extract data from DFT calculations. This should be the first script to run.", description="Run and extract data from DFT calculations. This should be the first script to run.")
        preprocess_parser.add_argument("input_file", type=str, help="Path to input file with the run parameters.")
        preprocess_parser.add_argument("-flush", type=str, help="Flushes output into stdout.")
        preprocess_parser.add_argument("-o", default="preprocess", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
        preprocess_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")

        if WFCGEN:
            wfc_parser = sub_parser.add_parser("wfcgen", help="Extracts wavefunctions from DFT calculations.", description="Extracts wavefunctions from DFT calculations.")
        if DOT:
            dot_parser = sub_parser.add_parser("dot", help="Calculates the dot product of Bloch factors of nearby wavefunctions.", description="Calculates the dot product of Bloch factors of nearby wavefunctions.")
        if CLUSTER:
            cluster_parser = sub_parser.add_parser("cluster", help="Classifies the eigenstates in bands.", description="Classifies the eigenstates in bands.")
        if BASIS:
            basis_parser = sub_parser.add_parser("basis", help="Finds problematic cases and make a local basis rotation of the wavefunctions.", description="Finds problematic cases and make a local basis rotation of the wavefunctions.")
        if R2K:
            r2k_parser = sub_parser.add_parser("r2k", help="Converts wavefunctions from r-space to k-space.", description="Converts wavefunctions from r-space to k-space.")
        if GEOMETRY:
            geometry_parser = sub_parser.add_parser("geometry", help="Calculates the Berry connections and curvatures.", description="Calculates the Berry connections and curvatures.")
        if CONDUCTIVITY:
            conductivity_parser = sub_parser.add_parser("conductivity", help="Calculates the optical linear conductivity of the system.", description="Calculates the optical linear conductivity of the system.")
        if SHG:
            shg_parser = sub_parser.add_parser("shg", help="Calculates the second harmonic generation conductivity of the system.", description="Calculate the second harmonic generation conductivity of the system.")
        argcomplete.autocomplete(parser)

        if WFCGEN:
            wfc_parser.add_argument("-nk"  , type=int, metavar=f"[0-{d.nks-1}]"  , default=None, choices=range(d.nks)  , help="k-point where wavefunctions will be generated (all bands) (default: All).")
            wfc_parser.add_argument("-band", type=int, metavar=f"[0-{d.nbnd-1}]", default=None, choices=range(d.nbnd), help="Band where wavefunction will be generated (on k-point -nk) (default: All).")
            wfc_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            wfc_parser.add_argument("-o", default="wfc", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            wfc_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if DOT:
            dot_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            dot_parser.add_argument("-flush", type=str, help="Flushes output into stdout.")
            dot_parser.add_argument("-o", default="dot", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            dot_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if CLUSTER:
            cluster_parser.add_argument("Mb" , type=int, nargs='?',   default=-1,   metavar=f"Mb (0-{d.nbnd-1})",    choices=range(d.nbnd) ,             help="Maximum band to consider.")
            cluster_parser.add_argument("-mb", type=int,              default=0,    metavar=f"[0-{d.nbnd-1}]"      , choices=range(d.nbnd)             , help="Minimum band to consider (default: 0).")
            cluster_parser.add_argument("-np", type=int,              default=1,    metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1).")
            cluster_parser.add_argument("-t",  type=restricted_float, default=0.95, metavar="[0.0-1.0]",  help="Tolerance used for graph construction (default: 0.95).")
            cluster_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            cluster_parser.add_argument("-o", default="cluster", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            cluster_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if BASIS:
            basis_parser.add_argument("Mb" , type=int           , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)             , help="Maximum band to consider.")
            basis_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1).")
            basis_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            basis_parser.add_argument("-o", default="basis", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            basis_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if R2K:
            r2k_parser.add_argument("Mb" , type=int           , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)             , help="Maximum band to consider.")
            r2k_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1).")
            r2k_parser.add_argument("-mb", type=int, default=0, metavar=f"[0-{d.nbnd-1}]"      , choices=range(d.nbnd)             , help="Minimum band to consider (default: 0).")
            r2k_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            r2k_parser.add_argument("-o", default="r2k", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            r2k_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if GEOMETRY:
            geometry_parser.add_argument("Mb"   , type=int                , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)                      , help="Maximum band to consider.")
            geometry_parser.add_argument("-np"  , type=int, default=1     , metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1)         , help="Number of processes to use (default: 1).")
            geometry_parser.add_argument("-mb"  , type=int, default=0     , metavar=f"[0-{d.nbnd-1}]"      , choices=range(d.nbnd)                      , help="Minimum band to consider (default: 0).")
            geometry_parser.add_argument("-prop", type=str, default="both"                                 , choices=["both", "connection", "curvature"], help="Specify which proprety to calculate. (default: both)")
            geometry_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            geometry_parser.add_argument("-o", default="geometry", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            geometry_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if CONDUCTIVITY:
            conductivity_parser.add_argument("cb" , type=int                        ,metavar=f"cb ({d.vb+1}-{d.nbnd-1})", choices=range(d.vb+1, d.nbnd)     , help="Index of the highest conduction band to consider.")
            conductivity_parser.add_argument("-np"       , type=int  , default=1    , metavar=f"[1-{os.cpu_count()}]"   , choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1).")
            conductivity_parser.add_argument("-eM"  , type=float, default=2.5                                                                          , help="Maximum energy in Ry units (default: 2.5).")
            conductivity_parser.add_argument("-eS" , type=float, default=0.001                                                                        , help="Energy step in Ry units (default: 0.001).")
            conductivity_parser.add_argument("-brd", type=float, default=0.01j                                                                        , help="Energy broading in Ry units (default: 0.01).")
            conductivity_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            conductivity_parser.add_argument("-o", default="conductivity", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            conductivity_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if SHG:
            shg_parser.add_argument("cb" , type=int                        ,metavar=f"cb ({d.vb+1}-{d.nbnd-1})", choices=range(d.vb+1, d.nbnd)     , help="Index of the highest conduction band to consider.")
            shg_parser.add_argument("-np"       , type=int  , default=1    , metavar=f"[1-{os.cpu_count()}]"   , choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            shg_parser.add_argument("-eM"  , type=float, default=2.5                                                                          , help="Maximum energy in Ry units (default: 2.5).")
            shg_parser.add_argument("-eS" , type=float, default=0.001                                                                        , help="Energy step in Ry units (default: 0.001).")
            shg_parser.add_argument("-brd", type=float, default=0.01j                                                                        , help="Energy broading in Ry units (default: 0.01).")
            shg_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            shg_parser.add_argument("-o", default="shg", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            shg_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
    except NameError as err:
        parser = CustomParser(formatter_class=argparse.RawTextHelpFormatter, description=
        """Master CLI for the berry suite. Other programs will become available after running the 'preprocess' command.
For more information add the '-h' flag to the 'preprocess' subcommand.""")
        sub_parser = parser.add_subparsers(dest="program", help="Run the 'preprocess' program.")

        preprocess_parser = sub_parser.add_parser("preprocess", help="Extract DFT calculations from specific program.", description="Extract DFT calculations from specific program.")
        preprocess_parser.add_argument("input_file", type=str, help="Path to input file from where to extract the run parameters.")
        preprocess_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
        preprocess_parser.add_argument("-o", default="preprocess", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
        preprocess_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
    finally:
        args = parser.parse_args()

    ###########################################################################
    # HANDLE BERRY SUITE OPTIONAL ARGUMENTS
    ###########################################################################
    if args.version:
        print(f"berry suite version: {__version__}")
        sys.exit(0)
    
    if args.enable_autocomplete:
        autocomplete()
        sys.exit(0)
    
    if args.disable_autocomplete:
        autocomplete(disable=True)
        sys.exit(0)

    if args.program is None:
        parser.print_help()
        sys.exit(0)

    ###########################################################################
    # PROCESSING ARGS
    ###########################################################################

    program_dict: Dict[str, Callable] = {
        "preprocess": preprocessing_cli,
        "wfcgen": generatewfc_cli,
        "dot": dotproduct_cli,
        "cluster": clustering_cli,
        "basis": basisrotation_cli,
        "r2k": r2k_cli,
        "geometry": berry_props_cli,
        "conductivity": conductivity_cli,
        "shg": shg_cli
    }

    program_dict[args.program](args)




##################################################################################################
# MAIN PROGRAMS
##################################################################################################
def autocomplete(disable: bool = False):
    """Enable or disable autocomplete for the CLI.

    This program checks if the bashrc file exists and then based on the disable flag it either adds or
    removes the autocomplete function from the bashrc file.

    Args:
        disable (bool, optional): if True, disable autocomplete. Defaults to False.
    """
    # Get bashrc file path
    bashrc_path = os.path.join(os.path.expanduser("~"), ".bashrc")

    # If disbale is True, remove the autocomplete function from the bashrc file if it exists
    if disable:
        if os.path.exists(bashrc_path):
            with open(bashrc_path, "r") as f:
                content = f.read()
            if re.search(r'eval "\$\(register-python-argcomplete berry\)\n?"', content):
                # Ask for confirmation
                while True:
                    ans = input("Are you sure you want to disable autocomplete? [y/n]: ")
                    if ans.lower() == "y":
                        break
                    elif ans.lower() == "n":
                        return
                    else:
                        print("Please answer with 'y' or 'n'")
                with open(bashrc_path, "w") as f:
                    f.write(re.sub(r'eval "\$\(register-python-argcomplete berry\)\n?"', "", content))
                    print("Autocomplete disabled!")
                    print("Please restart your terminal for the changes to take effect.")
            else:
                print("Autocomplete already disabled!")
        else:
            print("No bashrc file found. Autocomplete is already disabled.")
    else:
        # If bashrc file does not exist, create it
        if not os.path.exists(bashrc_path):
            with open(bashrc_path, "w") as f:
                pass
        # If autocomplete function is not in the bashrc file, add it
        with open(bashrc_path, "r") as f:
            lines = f.readlines()
        if not any(re.search(r'eval "\$\(register-python-argcomplete berry\)\n?"', line) for line in lines):
            # Ask for confirmation
            print("Autocomplete will be enabled. This will add the following line to your bashrc file:")
            print('eval "$(register-python-argcomplete berry)"')
            while True:
                print("Do you want to continue? [y/n]", end=" ")
                ans = input()
                if ans.lower() == "y":
                    break
                elif ans.lower() == "n":
                    print("Autocomplete not enabled.")
                    return
                else:
                    print("Please answer with 'y' or 'n'")
            with open(bashrc_path, "a") as f:
                f.write('eval "$(register-python-argcomplete berry)"\n')
                print("Autocomplete enabled.")
                print("Please restart your terminal for the changes to take effect.")
        else:
            print("Autocomplete already enabled.")

    return


def preprocessing_cli(args: argparse.Namespace):
    from berry import Preprocess
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert os.path.isfile(args.input_file), f"Input file {args.input_file} does not exist."
    with open(args.input_file, "r") as f:
        lines = f.readlines()

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    for line in lines:
        ii = line.split()
        if len(ii) == 0:
            continue
        if ii[0] == "k0":
            args_dict["k0"] = [float(ii[1]), float(ii[2]), float(ii[3])]
        if ii[0] == "nkx":
            args_dict["nkx"] = int(ii[1])
        if ii[0] == "nky":
            args_dict["nky"] = int(ii[1])
        if ii[0] == "nkz":
            args_dict["nkz"] = int(ii[1])
        if ii[0] == "step":
            args_dict["step"] = float(ii[1])
        if ii[0] == "nbnd":
            args_dict["nbnd"] = int(ii[1])
        if ii[0] == "npr":
            args_dict["npr"] = int(ii[1])
        if ii[0] == "dftdirectory":
            args_dict["dft_dir"] = os.path.abspath(ii[1])
        if ii[0] == "name_scf":
            args_dict["scf"] = ii[1]
        if ii[0] == "wfcdirectory":
            args_dict["wfc_dir"] = ii[1]
        if ii[0] == "point":
            args_dict["point"] = float(ii[1])
        if ii[0] == "program":
            args_dict["program"] = str(ii[1])
        if ii[0] == "refname":
            args_dict["ref_name"] = ii[1]

    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["flush"] = args.flush

    Preprocess(**args_dict).run() 

def generatewfc_cli(args: argparse.Namespace):
    from berry import WfcGenerator
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    if args.nk is None and args.band is not None:
        raise ValueError("If you want to generate the wavefunctions for a specific band, you must specify the number of k-points.")

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["nk_points"] = args.nk
    args_dict["bands"] = args.band
    args_dict["flush"] = args.flush

    WfcGenerator(**args_dict).run()

    return 

def dotproduct_cli(args: argparse.Namespace):
    from berry import run_dot
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["npr"] = args.np
    args_dict["flush"] = args.flush

    run_dot(**args_dict)

def clustering_cli(args: argparse.Namespace):
    from berry import run_clustering
    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["npr"] = args.np
    args_dict["max_band"] = args.Mb
    args_dict["min_band"] = args.mb
    args_dict["tol"] = args.t
    args_dict["flush"] = args.flush

    run_clustering(**args_dict)

def basisrotation_cli(args: argparse.Namespace):
    from berry import run_basis_rotation
    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["npr"] = args.np
    args_dict["max_band"] = args.Mb
    args_dict["flush"] = args.flush

    run_basis_rotation(**args_dict)


def r2k_cli(args: argparse.Namespace):
    from berry import run_r2k
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["npr"] = args.np
    args_dict["min_band"] = args.mb
    args_dict["max_band"] = args.Mb
    args_dict["flush"] = args.flush

    run_r2k(**args_dict)

def berry_props_cli(args: argparse.Namespace):
    from berry import run_berry_geometry
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["npr"] = args.np
    args_dict["min_band"] = args.mb
    args_dict["max_band"] = args.Mb
    args_dict["prop"] = args.prop
    args_dict["flush"] = args.flush

    run_berry_geometry(**args_dict)

def conductivity_cli(args: argparse.Namespace):
    from berry import run_conductivity
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    
    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["npr"] = args.np
    args_dict["conduction_band"] = args.cb
    args_dict["energy_max"] = args.eM
    args_dict["energy_step"] = args.eS
    args_dict["broadning"] = args.brd
    args_dict["flush"] = args.flush

    run_conductivity(**args_dict)


def shg_cli(args: argparse.Namespace):
    from berry import run_shg
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["npr"] = args.np
    args_dict["conduction_band"] = args.cb
    args_dict["energy_max"] = args.eM
    args_dict["energy_step"] = args.eS
    args_dict["broadning"] = args.brd
    args_dict["flush"] = args.flush

    run_shg(**args_dict)

##################################################################################################
# VIZUALIZATION PROGRAMS
##################################################################################################

def viz_berry_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="Visualizes the Berry connection and curvature vectors.")
    sub_parser = parser.add_subparsers(help="Available visualizations.", dest="viz_prog")

    # 1.1. BERRY CONNECTION
    bcc_parser = sub_parser.add_parser("bcc", help="Visualizes the Berry connection vectors.")
    bcc_parser.add_argument("band", type=int, metavar=f"band (0-{d.nbnd-1})", choices=range(d.nbnd), help="Band to visualize.")
    bcc_parser.add_argument("grad", type=int, metavar=f"grad (0-{d.nbnd-1})", choices=range(d.nbnd), help="Gradient to visualize.")
    bcc_parser.add_argument("-space", default="all", choices=["all", "real", "imag", "complex"], help="Space to visualize (default: all).")

    # 1.2. BERRY CURVATURE
    bcr_parser = sub_parser.add_parser("bcr", help="Visualizes the Berry curvature vectors.")
    bcr_parser.add_argument("band", type=int, metavar=f"band (0-{d.nbnd-1})", choices=range(d.nbnd), help="Band to visualize.")
    bcr_parser.add_argument("grad", type=int, metavar=f"grad (0-{d.nbnd-1})", choices=range(d.nbnd), help="Gradient to visualize.")
    bcr_parser.add_argument("-space", default="all", choices=["all", "real", "imag", "complex"], help="Space to visualize (default: all).")

    args = parser.parse_args()
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert args.viz_prog is not None, parser.error("No visualization program was selected.")

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################

    return args


def viz_debug_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="Visualizes the information of the calculation. It is useful for debugging.")
    sub_parser = parser.add_subparsers(help="Available visualizations.", dest="viz_prog")

    # 1.1 view data
    data_parser = sub_parser.add_parser("data", help="Visualizes the data of the system.")
    
    # 1.2 view data
    dot1_parser = sub_parser.add_parser("dot1", help="Dot product visualization. (Option 1)")
    
    # 1.2 view data
    dot2_parser = sub_parser.add_parser("dot2", help="Dot product visualization. (Option 2)")
    dot2_parser.add_argument("band", type=int, metavar=f"band (0-{d.nbnd-1})", choices=range(d.nbnd), help="Band to consider.")

    # 1.3 view eigenvalues
    eigen_parser = sub_parser.add_parser("eigen", help="Visualizes the eigenvalues of the system.")
    eigen_parser.add_argument("band", type=int, metavar=f"band (0-{d.nbnd-1})", choices=range(d.nbnd), help="Band to consider.")
    eigen_parser.add_argument("-acc", type=int, default=0, help="Precision of the eigenvalues.")

    # 1.4 view neighbors
    neighors_parser = sub_parser.add_parser("neig", help="Visualizes the neighbors in system.")

    # 1.5 view occupations
    occ_parser = sub_parser.add_parser("occ", help="Visualizes the occupations of each k point.")

    # 1.5 view real space
    real_parser = sub_parser.add_parser("r-space", help="Visualizes the real space of the system.")

    args = parser.parse_args()
    
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert args.viz_prog is not None, parser.error("No visualization program was selected.")

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################

    return args

if __name__ == "__main__":
    master_cli()
