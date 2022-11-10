from typing import Callable, Dict, Any

import os
import sys
import logging
import subprocess
import argparse, argcomplete

#TODO: Talk to professor about np.savez
#NOTE: np.savez could help with backwards compatability 


class CustomParser(argparse.ArgumentParser):
    def _check_value(self, action, value):
        if not isinstance(action.choices, range):
            super()._check_value(action, value)
        elif action.choices is not None and value not in action.choices:
            first, last = action.choices[0], action.choices[-1]
            msg = f"invalid choice: {value}. Choose from {first} up to (and including) {last}."
            raise argparse.ArgumentError(action, msg)

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
        parser = CustomParser(description="""Master Command Line Interface (CLI) for the Berry Suite.
The Berry Suite is a collection of programs that have to be run in a specific order. 
This CLI is meant to help the user run the programs in the correct order
and to help the user understand what each program does. 
When running this CLI, the user must specify the program to run and the arguments for that program.""")
        sub_parser = parser.add_subparsers(dest="program", help="Choose the program to run.")
        sub_parser.add_parser("enable_autocomplete", help="Enable autocomplete for the CLI.")

        preprocess_parser = sub_parser.add_parser("preprocess", help="Run and extract data from DFT calculations. This should be the first to run.", description="Run and extract data from DFT calculations. This should be the first to run.")
        preprocess_parser.add_argument("input_file", type=str, help="Path to input file with the run parameters.")
        preprocess_parser.add_argument("-o", default="preprocess", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
        preprocess_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
        
        if WFCGEN:
            wfc_parser = sub_parser.add_parser("wfcgen", help="Extracts wavefunctions from DFT calculations.", description="Extracts wavefunctions from DFT calculations.")
        if DOT:
            dot_parser = sub_parser.add_parser("dot", help="Calculates the dot product of Bloch factors of nearby wavefunctions.", description="Calculates the dot product of Bloch factors of nearby wavefunctions.")
        if CLUSTER:
            cluster_parser = sub_parser.add_parser("cluster", help="Classify the eigenstates in bands.", description="Classify the eigenstates in bands.")
        if BASIS:
            basis_parser = sub_parser.add_parser("basis", help="Finds problematic cases and make a local basis rotation of the wavefunctions.", description="Finds problematic cases and make a local basis rotation of the wavefunctions.")
        if R2K:
            r2k_parser = sub_parser.add_parser("r2k", help="Calculates the grid of points in the k-space", description="Calculates the grid of points in the k-space")
        if GEOMETRY:
            geometry_parser = sub_parser.add_parser("geometry", help="Calculates the Berry connections and the Berry curvature.", description="Calculates the Berry connections and the Berry curvature.")
        if CONDUCTIVITY:
            conductivity_parser = sub_parser.add_parser("conductivity", help="Calculates the optical linear conductivity of the system..", description="Calculates the optical linear conductivity of the system.")
        if SHG:
            shg_parser = sub_parser.add_parser("shg", help="Calculates the second harmonic generation conductivity of the system.", description="Calculate the second harmonic generation conductivity of the system.")
        argcomplete.autocomplete(parser)

        if WFCGEN:
            wfc_parser.add_argument("-nk"  , type=int, metavar=f"[0-{d.nks-1}]"  , default=None, choices=range(d.nks)  , help="K-point to generate the wavefunction for all bands (default: All).")
            wfc_parser.add_argument("-band", type=int, metavar=f"[0-{d.nbnd-1}]", default=None, choices=range(d.nbnd), help="Band to generate the wavefunction for a single k-point (default: All).")
            wfc_parser.add_argument("-o", default="wfc", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
            wfc_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
        if DOT:
            dot_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            dot_parser.add_argument("-o", default="dot", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
            dot_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
        if CLUSTER:
            cluster_parser.add_argument("Mb" , type=int           , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)             , help="Maximum band to consider")
            cluster_parser.add_argument("-mb", type=int, default=0, metavar=f"[0-{d.nbnd-1}]"      , choices=range(d.nbnd)             , help="Minimum band to consider (default: 0)")
            cluster_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            cluster_parser.add_argument("-t",  type=restricted_float, default=0.95, metavar="[0.0-1.0]", help="Tolerance used for graph construction (default: 0.95)")
            cluster_parser.add_argument("-o", default="cluster", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
            cluster_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
        if BASIS:
            basis_parser.add_argument("Mb" , type=int           , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)             , help="Maximum band to consider")
            basis_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            basis_parser.add_argument("-o", default="basis", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
            basis_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
        if R2K:
            r2k_parser.add_argument("Mb" , type=int           , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)             , help="Maximum band to consider")
            r2k_parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            r2k_parser.add_argument("-mb", type=int, default=0, metavar=f"[0-{d.nbnd-1}]"      , choices=range(d.nbnd)             , help="Minimum band to consider (default: 0)")
            r2k_parser.add_argument("-o", default="r2k", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
            r2k_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
        if GEOMETRY:
            geometry_parser.add_argument("Mb"   , type=int                , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)                      , help="Maximum band to consider")
            geometry_parser.add_argument("-np"  , type=int, default=1     , metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1)         , help="Number of processes to use (default: 1)")
            geometry_parser.add_argument("-mb"  , type=int, default=0     , metavar=f"[0-{d.nbnd-1}]"      , choices=range(d.nbnd)                      , help="Minimum band to consider (default: 0)")
            geometry_parser.add_argument("-prop", type=str, default="both"                                 , choices=["both", "connection", "curvature"], help="Specify which proprety to calculate. (default: both)")
            geometry_parser.add_argument("-o", default="geometry", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
            geometry_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
        if CONDUCTIVITY:
            conductivity_parser.add_argument("cb" , type=int                        ,metavar=f"cb ({d.vb+1}-{d.nbnd-1})", choices=range(d.vb+1, d.nbnd)     , help="Index of the conduction band.")
            conductivity_parser.add_argument("-np"       , type=int  , default=1    , metavar=f"[1-{os.cpu_count()}]"   , choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            conductivity_parser.add_argument("-eM"  , type=float, default=2.5                                                                          , help="Maximum energy in Ry units (default: 2.5).")
            conductivity_parser.add_argument("-eS" , type=float, default=0.001                                                                        , help="Energy step in Ry units (default: 0.001).")
            conductivity_parser.add_argument("-brd", type=float, default=0.01j                                                                        , help="Energy broading in Ry units (default: 0.01).")
            conductivity_parser.add_argument("-o", default="conductivity", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
            conductivity_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
        if SHG:
            shg_parser.add_argument("cb" , type=int                        ,metavar=f"cb ({d.vb+1}-{d.nbnd-1})", choices=range(d.vb+1, d.nbnd)     , help="Index of the conduction band.")
            shg_parser.add_argument("-np"       , type=int  , default=1    , metavar=f"[1-{os.cpu_count()}]"   , choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
            shg_parser.add_argument("-eM"  , type=float, default=2.5                                                                          , help="Maximum energy in Ry units (default: 2.5).")
            shg_parser.add_argument("-eS" , type=float, default=0.001                                                                        , help="Energy step in Ry units (default: 0.001).")
            shg_parser.add_argument("-brd", type=float, default=0.01j                                                                        , help="Energy broading in Ry units (default: 0.01).")
            shg_parser.add_argument("-o", default="shg", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
            shg_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
    except NameError as err:
        parser = CustomParser(formatter_class=argparse.RawTextHelpFormatter, description=
        """Master CLI for the Berry Suite. Other programs will become available after running the 'preprocess' command.
For more information add the '-h' flag to the 'preprocess' subcommand.""")
        sub_parser = parser.add_subparsers(dest="program", help="Run the 'preprocess' program.")

        preprocess_parser = sub_parser.add_parser("preprocess", help="Extract DFT calculations from specific program.", description="Extract DFT calculations from specific program.")
        preprocess_parser.add_argument("input_file", type=str, help="Path to input file from where to extract the run parameters.")
        preprocess_parser.add_argument("-o", default="preprocess", type=str, metavar="file_path", help="Name of output log file. If extension is provided it will be ignored!")
        preprocess_parser.add_argument("-v"        , action="store_true", help="Increase output verbosity")
    finally:
        args = parser.parse_args()

    ###########################################################################
    # ASSERTIONS
    ###########################################################################
    if args.program is None:
        parser.print_help()
        sys.exit(0)

    ###########################################################################
    # PROCESSING ARGS
    ###########################################################################

    program_dict: Dict[str, Callable] = {
        "enable_autocomplete": enable_autocomplete,
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
def enable_autocomplete(args: argparse.Namespace):
    """
    Checks if ~/.bashrc exists and if not creates it. 
    Then checks if eval '\"$(register-python-argcomplete berry)\"' is in the .bashrc file and if not asks the user if he wants to enable it.
    """
    bashrc_path = os.path.expanduser("~/.bashrc")
    if not os.path.exists(bashrc_path):
        with open(bashrc_path, "w") as f:
            f.write("#!/bin/bash \n")
            with open(bashrc_path, "r") as f:
                if "eval \"$(register-python-argcomplete berry)\"" not in f.read():
                    while True:
                        user_input = input("Autocomplete is not enabled. Do you want to enable it? [y/n]: ")
                        if user_input.lower() == "y":
                            f.write("eval \"$(register-python-argcomplete berry)\"")
                            print("Autocomplete enabled. Please restart your terminal!")
                            break
                        elif user_input.lower() == "n":
                            print("Autocomplete not enabled.")
                            break
                        else:
                            print("Please enter 'y' or 'n'!")
                else:
                    print("Autocomplete already enabled!")
    else:
        with open(bashrc_path, "r") as f:
            if "eval \"$(register-python-argcomplete berry)\"" not in f.read():
                while True:
                    user_input = input("Autocomplete is not enabled. Do you want to enable it? [y/n]: ")
                    if user_input.lower() == "y":
                        with open(bashrc_path, "a") as f:
                            f.write("eval \"$(register-python-argcomplete berry)\"")
                        print("Autocomplete enabled. Please restart your terminal!")
                        break
                    elif user_input.lower() == "n":
                        print("Autocomplete not enabled.")
                        break
                    else:
                        print("Please enter 'y' or 'n'!")
            else:
                print("Autocomplete already enabled!")


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