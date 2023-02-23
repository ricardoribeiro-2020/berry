from typing import Callable, Dict, Tuple

import os
import re
import sys
import logging
import subprocess
import argparse, argcomplete

from berry import __version__

#TODO: Talk about np.savez
#NOTE: np.savez could help with backwards compatibility 


class CustomParser(argparse.ArgumentParser):
    def _check_value(self, action, value):
        if isinstance(action.choices, dict):
            if 'preprocess' in action.choices.keys() and value not in action.choices.keys():
                msg = f"""invalid program choice: {value}.
This error probably means you are trying to run a program in the incorrect order and therefore do not have the required files.
Try the program in the following order: 'preprocess', 'wfcgen', 'dot', 'cluster', 'r2k', 'geometry', 'conductivity', 'shg'."""
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


WFCGEN = DOT = CLUSTER = BASIS = R2K = GEOMETRY = CONDUCTIVITY = SHG = 0
try:
    import berry._subroutines.loadmeta as m

    # NOTE: Changed 'm.workdir' to 'os.getcwd()' because one might not want to run from the start
    m.workdir = os.getcwd()

    if os.path.exists(os.path.join(m.workdir, "datafile.npy")):
        WFCGEN = 1
    if os.path.exists(os.path.join(m.workdir, "wfc")):
        DOT = 1
    if os.path.exists(os.path.join(m.workdir, "dpc.npy")):
        CLUSTER = 1
    if os.path.exists(os.path.join(m.workdir, "signalfinal.npy")):
        BASIS = 1
    if os.path.exists(os.path.join(m.workdir, "final.report")):
        R2K = 1
    if os.path.exists(os.path.join(m.workdir, "wfcgra0.npy")):
        GEOMETRY = 1
    if os.path.exists(os.path.join(m.workdir, "berryConn0_0.npy")):
        CONDUCTIVITY = 1
        SHG = 1
except:
    pass

##################################################################################################
# MAIN PROGRAMS
##################################################################################################
def berry_cli():
    ###########################################################################
    # 1. DEFINING BERRY CLI ARGS
    ###########################################################################
    parser = CustomParser(formatter_class=argparse.RawTextHelpFormatter,
        description="""This is the Master Command Line Interface (CLI) for the berry suite.
The berry suite extracts the Bloch wavefunctions from DFT calculations in an ordered way. 
This CLI is an interface to run the different scripts of the berry suite.
The scripts have to be run in order because each script uses results from the previous ones.
Command line is of the form:

berry [package options] script parameter [script options]
""")
    parser.add_argument("--version", action="store_true", help="Displays current Berry version.")
    # parser.add_argument("--enable-autocomplete", action="store_true", help="Enables autocomplete for the berry CLI.")
    # parser.add_argument("--disable-autocomplete", action="store_true", help="Disables autocomplete for the berry CLI.")

    main_sub_parser = parser.add_subparsers(metavar="MAIN_PROGRAMS" ,dest="main_programs", help="Choose the program to run.")
    preprocess_parser = main_sub_parser.add_parser("preprocess", help="Run and extract data from DFT calculations. This should be the first script to run.", description="Run and extract data from DFT calculations. This should be the first script to run.")

    try:
        if WFCGEN:
            wfc_parser = main_sub_parser.add_parser("wfcgen", help="Extracts wavefunctions from DFT calculations.", description="Extracts wavefunctions from DFT calculations.")
        if DOT:
            dot_parser = main_sub_parser.add_parser("dot", help="Calculates the dot product of Bloch factors of nearby wavefunctions.", description="Calculates the dot product of Bloch factors of nearby wavefunctions.")
        if CLUSTER:
            cluster_parser = main_sub_parser.add_parser("cluster", help="Classifies the eigenstates in bands.", description="Classifies the eigenstates in bands.")
        if BASIS:
            basis_parser = main_sub_parser.add_parser("basis", help="Finds problematic cases and make a local basis rotation of the wavefunctions.", description="Finds problematic cases and make a local basis rotation of the wavefunctions.")
        if R2K:
            r2k_parser = main_sub_parser.add_parser("r2k", help="Converts wavefunctions from r-space to k-space.", description="Converts wavefunctions from r-space to k-space.")
        if GEOMETRY:
            geometry_parser = main_sub_parser.add_parser("geometry", help="Calculates the Berry connections and curvatures.", description="Calculates the Berry connections and curvatures.")
        if CONDUCTIVITY:
            conductivity_parser = main_sub_parser.add_parser("conductivity", help="Calculates the optical linear conductivity of the system..", description="Calculates the optical linear conductivity of the system.")
        if SHG:
            shg_parser = main_sub_parser.add_parser("shg", help="Calculates the second harmonic generation conductivity of the system.", description="Calculate the second harmonic generation conductivity of the system.")
        argcomplete.autocomplete(parser)

        if WFCGEN:
            wfc_parser.add_argument("-nk"  , type=int, metavar=f"[0-{m.nks-1}]"  , default=None, choices=range(m.nks)  , help="k-point where wavefunctions will be generated (all bands) (default: All).")
            wfc_parser.add_argument("-band", type=int, metavar=f"[0-{m.nbnd-1}]", default=None, choices=range(m.nbnd), help="Band where wavefunction will be generated (on k-point -nk) (default: All).")
            wfc_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            wfc_parser.add_argument("-o", default="wfc", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            wfc_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if DOT:
            dot_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            dot_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            dot_parser.add_argument("-o", default="dot", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            dot_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if CLUSTER:
            cluster_parser.add_argument("Mb" , type=int, nargs='?',   default=-1,   metavar=f"Mb (0-{m.nbnd-1})",    choices=range(m.nbnd) ,             help="Maximum band to consider.")
            cluster_parser.add_argument("-mb", type=int,              default=0,    metavar=f"[0-{m.nbnd-1}]"      , choices=range(m.nbnd)             , help="Minimum band to consider (default: 0).")
            cluster_parser.add_argument("-np", type=int,              default=1,    metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1).")
            cluster_parser.add_argument("-t",  type=restricted_float, default=0.95, metavar="[0.0-1.0]",  help="Tolerance used for graph construction (default: 0.95).")
            cluster_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            cluster_parser.add_argument("-o", default="cluster", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            cluster_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if BASIS:
            basis_parser.add_argument("Mb" , type=int           , metavar=f"Mb (0-{m.nbnd-1})"   , choices=range(m.nbnd)             , help="Maximum band to consider.")
            basis_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1).")
            basis_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            basis_parser.add_argument("-o", default="basis", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            basis_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if R2K:
            r2k_parser.add_argument("Mb" , type=int           , metavar=f"Mb (0-{m.nbnd-1})"   , choices=range(m.nbnd)             , help="Maximum band to consider.")
            r2k_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1).")
            r2k_parser.add_argument("-mb", type=int, default=0, metavar=f"[0-{m.nbnd-1}]"      , choices=range(m.nbnd)             , help="Minimum band to consider (default: 0).")
            r2k_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            r2k_parser.add_argument("-o", default="r2k", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            r2k_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if GEOMETRY:
            geometry_parser.add_argument("Mb"   , type=int                , metavar=f"Mb (0-{m.nbnd-1})"   , choices=range(m.nbnd)                      , help="Maximum band to consider.")
            geometry_parser.add_argument("-np"  , type=int, default=1     , metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1)         , help="Number of processes to use (default: 1).")
            geometry_parser.add_argument("-mb"  , type=int, default=0     , metavar=f"[0-{m.nbnd-1}]"      , choices=range(m.nbnd)                      , help="Minimum band to consider (default: 0).")
            geometry_parser.add_argument("-prop", type=str, default="both", metavar="",choices=["both", "conn", "curv"], help="Specify which proprety to calculate. Possible choices are 'both', 'conn' and 'curv' (default: both)")
            geometry_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            geometry_parser.add_argument("-o", default="geometry", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            geometry_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if CONDUCTIVITY:
            conductivity_parser.add_argument("cb" , type=int                        ,metavar=f"cb ({m.vb+1}-{m.nbnd-1})", choices=range(m.vb+1, m.nbnd)     , help="Index of the highest conduction band to consider.")
            conductivity_parser.add_argument("-np"       , type=int  , default=1    , metavar=f"[1-{os.cpu_count()}]"   , choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1).")
            conductivity_parser.add_argument("-eM", metavar="", type=float, default=2.5, help="Maximum energy in Ry units (default: 2.5).")
            conductivity_parser.add_argument("-eS", metavar="", type=float, default=0.001, help="Energy step in Ry units (default: 0.001).")
            conductivity_parser.add_argument("-brd", metavar="", type=float, default=0.01j, help="Energy broading in Ry units (default: 0.01).")
            conductivity_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            conductivity_parser.add_argument("-o", default="conductivity", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            conductivity_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
        if SHG:
            shg_parser.add_argument("cb" , type=int,metavar=f"cb ({m.vb+1}-{m.nbnd-1})", choices=range(m.vb+1, m.nbnd), help="Index of the highest conduction band to consider.")
            shg_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            shg_parser.add_argument("-eM", metavar="", type=float, default=2.5, help="Maximum energy in Ry units (default: 2.5).")
            shg_parser.add_argument("-eS", metavar="", type=float, default=0.001, help="Energy step in Ry units (default: 0.001).")
            shg_parser.add_argument("-brd", metavar="", type=float, default=0.01j, help="Energy broading in Ry units (default: 0.01).")
            shg_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
            shg_parser.add_argument("-o", default="shg", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
            shg_parser.add_argument("-v"        , action="store_true", help="Increases output verbosity.")
    except NameError as err:
        #TODO: Should we display a helpfull message indicating that the user should run the missing berry programs?
        pass
    finally:
        preprocess_parser.add_argument("input_file", type=str, help="Path to input file with the run parameters.")
        preprocess_parser.add_argument("-flush", action="store_true", help="Flushes output into stdout.")
        preprocess_parser.add_argument("-o", default="preprocess", type=str, metavar="file_path", help="Name of output log file. Extension will be .log regardless of user input.")
        preprocess_parser.add_argument("-v", action="store_true", help="Increases output verbosity")

        args = parser.parse_args()

    ###########################################################################
    # HANDLE BERRY SUITE OPTIONAL ARGUMENTS
    ###########################################################################
    if args.version:
        print(f"berry suite version: {__version__}")
        sys.exit(0)
    
    # if args.enable_autocomplete:
    #     autocomplete("berry")
    #     sys.exit(0)
    
    # if args.disable_autocomplete:
    #     autocomplete("berry", disable=True)
    #     sys.exit(0)

    if args.main_programs is None:
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

    program_dict[args.main_programs](args)

def preprocessing_cli(args: argparse.Namespace):
    from berry.preprocessing import Preprocess
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
    from berry.generatewfc import WfcGenerator
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
    from berry.dotproduct import run_dot
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
    from berry.clustering_bands import run_clustering
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
    from berry.basisrotation import run_basis_rotation
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
    from berry.r2k import run_r2k
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
    from berry.berry_geometry import run_berry_geometry
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["logger_level"] = logging.DEBUG if args.v else logging.INFO
    args_dict["logger_name"] = args.o
    args_dict["npr"] = 1 #NOTE: UNTIL NOW, ONLY ONE PROCESS IS SUPPORTED
    args_dict["min_band"] = args.mb
    args_dict["max_band"] = args.Mb
    args_dict["prop"] = args.prop
    args_dict["flush"] = args.flush

    run_berry_geometry(**args_dict)

def conductivity_cli(args: argparse.Namespace):
    from berry.conductivity import run_conductivity
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
    from berry.shg import run_shg
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

#TODO: Make the entire process of adding new commands more automatic
#IDEA: Maybe using a global dict of commands with a tree structure will be a good idea to easily implement the same cli structure
##################################################################################################
# VISUALIZATION PROGRAMS
##################################################################################################
def berry_vis_cli():
    sub_programs = {
        "debug": "debug_vis",
        "geometry": "geometry_vis",
        "wave": "wave_vis",
    }
    ###########################################################################
    # 1. DEFINING BERRY VIS CLI ARGS
    ###########################################################################
    parser = CustomParser(description="Berry Visualization Program")
    # parser.add_argument("--enable-autocomplete", action="store_true", help="Enables autocomplete for the berry-vis CLI.")
    # parser.add_argument("--disable-autocomplete", action="store_true", help="Disables autocomplete for the berry-vis CLI.")

    vis_sub_parser = parser.add_subparsers(metavar="VIS_PROGRAMS", dest="vis_programs", help="Choose a visualization program.")
    debug_parser = vis_sub_parser.add_parser("debug", help="Prints data for debugging.")
    geometry_parser = vis_sub_parser.add_parser("geometry", help="Draws Berry connection and curvature vectors.")
    wave_parser = vis_sub_parser.add_parser("wave", help="Shows the electronic band structure.")
    
    debug_dot2_parser, debug_eigen_parser = handle_debug_parser(debug_parser)
    bcc_parser, bcr_parser = handle_geometry_parser(geometry_parser)
    wave_corrected_parser, wave_machine_parser = handle_wave_parser(wave_parser)

    argcomplete.autocomplete(parser)

    try:
        debug_dot2_parser.add_argument("band", type=int, metavar=f"band (0-{m.nbnd-1})", choices=range(m.nbnd), help="Band to consider.")

        debug_eigen_parser.add_argument("band", type=int, metavar=f"band (0-{m.nbnd-1})", choices=range(m.nbnd), help="Band to consider.")
        debug_eigen_parser.add_argument("-acc", type=int, default=0, help="Precision of the eigenvalues.")

        bcc_parser.add_argument("band", type=int, metavar=f"band (0-{m.nbnd-1})", choices=range(m.nbnd), help="Band to visualize.")
        bcc_parser.add_argument("grad", type=int, metavar=f"grad (0-{m.nbnd-1})", choices=range(m.nbnd), help="Gradient to visualize.")
        bcc_parser.add_argument("-space", default="all", choices=["all", "real", "imag", "complex"], help="Space to visualize (default: all).")
    
        bcr_parser.add_argument("band", type=int, metavar=f"band (0-{m.nbnd-1})", choices=range(m.nbnd), help="Band to visualize.")
        bcr_parser.add_argument("grad", type=int, metavar=f"grad (0-{m.nbnd-1})", choices=range(m.nbnd), help="Gradient to visualize.")
        bcr_parser.add_argument("-space", default="all", choices=["all", "real", "imag", "complex"], help="Space to visualize (default: all).")

        wave_corrected_parser.add_argument("Mb", type=int, metavar=f"Mb (0-{m.nbnd-1})", choices=range(m.nbnd), help="Maximum band to consider")
        wave_corrected_parser.add_argument("-mb", type=int, default=0, metavar=f"(0-{m.nbnd-1})", choices=range(m.nbnd), help="Minimum band to consider (default: 0)")
        
        wave_machine_parser.add_argument("Mb", type=int, metavar=f"Mb (0-{m.nbnd-1})", choices=range(m.nbnd), help="Maximum band to consider")
        wave_machine_parser.add_argument("-mb", type=int, default=0, metavar=f"(0-{m.nbnd-1})", choices=range(m.nbnd), help="Minimum band to consider (default: 0)")

    except NameError:
        print(f"berry-vis cannot display any output because no berry calculations can be found in the directory: {os.getcwd()}")
        print("For more information, please run: 'berry preprocess -h'")
        sys.exit(1)

    args = parser.parse_args()
    ###########################################################################
    # HANDLE BERRY VIS SUITE OPTIONAL ARGUMENTS
    ###########################################################################
    # if args.enable_autocomplete:
    #     autocomplete("berry-vis")
    #     sys.exit(0)
    
    # if args.disable_autocomplete:
    #     autocomplete("berry-vis", disable=True)
    #     sys.exit(0)
    
    if args.vis_programs is None:
        parser.print_help()
        sys.exit(0)

    for sub_program in sub_programs:
        if args.vis_programs == sub_program:
            if getattr(args, sub_programs[sub_program]) is None:
                eval(f"{sub_program}_parser.print_help()")
                sys.exit(0)

    ###########################################################################
    # HANDLE BERRY VIS SUITE SUB PROGRAMS
    ###########################################################################
    from berry.vis import _debug, _geometry, _wave

    program_dict: Dict[str, Callable] = {
        "debug": _debug.debug,
        "geometry": _geometry.geometry,
        "wave": _wave.wave,
    }

    program_dict[args.vis_programs](args)

def handle_wave_parser(debug_parser: CustomParser) -> Tuple[argparse.Namespace]:
    wave_programs_parser = debug_parser.add_subparsers(metavar="WAVE_PROGRAMS", dest="wave_vis", help="Choose how to visualize the bands.")

    # VIEW CORRECTED
    wave_corrected_parser = wave_programs_parser.add_parser("corrected", help="Shows the corrected bands.")

    # VIEW MACHINE
    wave_machine_parser = wave_programs_parser.add_parser("machine", help="Shows the original bands.")

    return wave_corrected_parser, wave_machine_parser

def handle_debug_parser(debug_parser: CustomParser) -> Tuple[argparse.Namespace]:
    debug_programs_parser = debug_parser.add_subparsers(metavar="DEBUG_PROGRAMS", dest="debug_vis", help="Choose a debug program.")

    # VIEW DATA
    debug_data_parser = debug_programs_parser.add_parser("data", help="Visualizes the data of the system.")
    
    # VIEW DOT 1
    debug_dot1_parser = debug_programs_parser.add_parser("dot1", help="Dot product visualization. (Option 1)")
    
    # VIEW DOT 2
    debug_dot2_parser = debug_programs_parser.add_parser("dot2", help="Dot product visualization. (Option 2)")

    # VIEW EIGENVALUES
    debug_eigen_parser = debug_programs_parser.add_parser("eigen", help="Visualizes the eigenvalues of the system.")

    # VIEW NEIGHBORS
    debug_neighors_parser = debug_programs_parser.add_parser("neig", help="Visualizes the neighbors in system.")

    # VIEW OCCUPATIONS
    debug_occ_parser = debug_programs_parser.add_parser("occ", help="Visualizes the occupations of each k point.")

    # VIEW REAL SPACE
    debug_real_parser = debug_programs_parser.add_parser("r-space", help="Visualizes the real space of the system.")

    return debug_dot2_parser, debug_eigen_parser

def handle_geometry_parser(geometry_parser: CustomParser) -> Tuple[argparse.Namespace]:
    geometry_vis_parser= geometry_parser.add_subparsers(metavar="GEOMETRY_PROGRAMS", dest="geometry_vis", help="Choose which berry property to show.")

    # 1.1. BERRY CONNECTION
    bcc_parser = geometry_vis_parser.add_parser("bcc", help="Graph of the Berry connection vectors.")

    # 1.2. BERRY CURVATURE
    bcr_parser = geometry_vis_parser.add_parser("bcr", help="Graph of the Berry curvature vectors.")

    return bcc_parser, bcr_parser

###########################################################################
# SUB PROCEDURES
###########################################################################

def autocomplete(cli_tool_name: str, disable: bool = False):
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
            if re.search(r'eval "\$\(register-python-argcomplete '+cli_tool_name+r'\)\n?"', content):
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
                    f.write(re.sub(r'eval "\$\(register-python-argcomplete '+cli_tool_name+r'\)\n?"', "", content))
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
        if not any(re.search(r'eval "\$\(register-python-argcomplete '+cli_tool_name+r'\)\n?"', line) for line in lines):
            # Ask for confirmation
            print("Autocomplete will be enabled. This will add the following line to your bashrc file:")
            print(f'eval "$(register-python-argcomplete {cli_tool_name})"')
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
                f.write(f'eval "$(register-python-argcomplete {cli_tool_name})"\n')
                print("Autocomplete enabled.")
                print("Please restart your terminal for the changes to take effect.")
        else:
            print("Autocomplete already enabled.")