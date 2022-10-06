from multiprocessing.util import sub_debug
from typing import Dict, Any

import os
import argparse

import python.loaddata as d


class CustomParser(argparse.ArgumentParser):
    def _check_value(self, action, value):
        if not isinstance(action.choices, range):
            super()._check_value(action, value)
        elif action.choices is not None and value not in action.choices:
            first, last = action.choices[0], action.choices[-1]
            msg = f"invalid choice: {value}. Choose from {first} up to (and including) {last}."
            raise argparse.ArgumentError(action, msg)


##################################################################################################
# MAIN PROGRAMS
##################################################################################################

def preprocessing_cli() -> Dict[str, Any]:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="")

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################

    return

def generatewfc_cli() -> Dict[str, Any]:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="""
    This program reads the wfc from DFT calculations. Makes them coherent and saves them in separate files.""")

    parser.add_argument("-nk"  , type=int, metavar=f"[0-{d.nr-1}]"  , default=None, choices=range(d.nr)  , help="K-point to generate the wavefunction for all bands (default: All).")
    parser.add_argument("-band", type=int, metavar=f"[0-{d.nbnd-1}]", default=None, choices=range(d.nbnd), help="Band to generate the wavefunction for a single k-point (default: All).")
    args = parser.parse_args()

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert not (args.band and not args.nk), "If you specify a band (-band), you must specify a k-point (-nk)."

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    if args.nk is None:
        args_dict["NK"] = range(d.nks)
        args_dict["BAND"] = None
    elif args.band is None:
        args_dict["NK"] = args.nk
        args_dict["BAND"] = range(d.nbnd)
    else:
        args_dict["NK"] = args.nk
        args_dict["BAND"] = args.band

    return args_dict

def dotproduct_cli() -> Dict[str, Any]:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="Calculates the dot product between the neighbouring points of the wavefunction.")
    parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
    args = parser.parse_args()

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = max(args.np, 1)

    return args_dict

def r2k_cli() -> Dict[str, Any]:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="Calculates the grid of points in the k-space")
    parser.add_argument("Mb" , type=int           , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)             , help="Maximum band to consider")
    parser.add_argument("-np", type=int, default=1, metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
    parser.add_argument("-mb", type=int, default=0, metavar=f"[0-{d.nbnd-1}]"      , choices=range(d.nbnd)             , help="Minimum band to consider (default: 0)")
    args = parser.parse_args()

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = args.np
    args_dict["MIN_BAND"] = args.mb
    args_dict["MAX_BAND"] = args.Mb

    return args_dict

def berry_props_cli() -> Dict[str, Any]:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="Calculates the Berry connection and curvature of all the combinations of bands.")
    parser.add_argument("Mb"   , type=int                , metavar=f"Mb (0-{d.nbnd-1})"   , choices=range(d.nbnd)                      , help="Maximum band to consider")
    parser.add_argument("-np"  , type=int, default=1     , metavar=f"[1-{os.cpu_count()}]", choices=range(1, os.cpu_count()+1)         , help="Number of processes to use (default: 1)")
    parser.add_argument("-mb"  , type=int, default=0     , metavar=f"[0-{d.nbnd-1}]"      , choices=range(d.nbnd)                      , help="Minimum band to consider (default: 0)")
    parser.add_argument("-prop", type=str, default="both"                                 , choices=["both", "connection", "curvature"], help="Specify which proprety to calculate. (default: both)")
    args = parser.parse_args()
    
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = min(args.np, args.Mb - args.mb + 1)
    args_dict["MIN_BAND"] = args.mb
    args_dict["MAX_BAND"] = args.Mb
    args_dict["PROP"] = args.prop
    
    return args_dict


def conductivity_cli() -> Dict[str, Any]:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="")
    parser.add_argument("cb" , type=int                        ,metavar=f"cb ({d.vb+1}-{d.nbnd-1})", choices=range(d.vb+1, d.nbnd)     , help="Index of the conduction band.")
    parser.add_argument("-np"       , type=int  , default=1    , metavar=f"[1-{os.cpu_count()}]"   , choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
    parser.add_argument("-enermax"  , type=float, default=2.5                                                                          , help="Maximum energy in Ry units (default: 2.5).")
    parser.add_argument("-enerstep" , type=float, default=0.001                                                                        , help="Energy step in Ry units (default: 0.001).")
    parser.add_argument("-broadning", type=float, default=0.01j                                                                        , help="Energy broading in Ry units (default: 0.01).")
    args = parser.parse_args()
    
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    
    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = args.np
    args_dict["ENERMAX"] = args.enermax
    args_dict["ENERSTEP"] = args.enerstep
    args_dict["BROADNING"]= args.broadning
    args_dict["BANDEMPTY"]= args.cb

    return args_dict


def shg_cli() -> Dict[str, Any]:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = CustomParser(description="")
    parser.add_argument("cb"        , type=int                 , metavar=f"cb ({d.vb+1}-{d.nbnd-1})", choices=range(d.vb+1, d.nbnd)     , help="Index of the conduction band.")
    parser.add_argument("-np"       , type=int  , default=1    , metavar=f"[1-{os.cpu_count()}]"    , choices=range(1, os.cpu_count()+1), help="Number of processes to use (default: 1)")
    parser.add_argument("-enermax"  , type=float, default=2.5                                                                           , help="Maximum energy in Ry units (default: 2.5).")
    parser.add_argument("-enerstep" , type=float, default=0.001                                                                         , help="Energy step in Ry units (default: 0.001).")
    parser.add_argument("-broadning", type=float, default=0.01j                                                                         , help="Energy broading in Ry units (default: 0.01).")
    args = parser.parse_args()

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = args.np
    args_dict["ENERMAX"] = args.enermax
    args_dict["ENERSTEP"] = args.enerstep
    args_dict["BROADNING"]= args.broadning
    args_dict["BANDEMPTY"]= args.cb

    ###########################################################################
    return args_dict

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