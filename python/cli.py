import argparse
import os

import loaddata as d

def preprocessing_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = argparse.ArgumentParser(description="")

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    # assert args.np <= os.cpu_count(), f"The number of processes ({args.np}) requested are not available."

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################

    return

def generatewfc_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = argparse.ArgumentParser(description="""
    This program reads the wfc from DFT calculations. Makes them coherent and saves them in separate files.""")

    parser.add_argument("-nk", type=int, default=None, help="K-point to generate the wavefunction for all bands (default: None).")
    parser.add_argument("-band", type=int, default=None, help="Band to generate the wavefunction for a single k-point (default: None).")
    args = parser.parse_args()

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert not (args.band and not args.nk), "If you specify a band (-band), you must specify a k-point (-nk)."

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    if args.nk and not args.band:
        args_dict["NK"] = args.nk
        args_dict["BAND"] = range(d.nbnd)
    elif args.nk and args.band:
        args_dict["NK"] = args.nk
        args_dict["BAND"] = [args.band]
    else:
        args_dict["NK"] = range(d.nks)
        args_dict["BAND"] = None

    return args_dict

def dotproduct_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = argparse.ArgumentParser(description="Calculates the dot product between the neighbouring points of the wavefunction.")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    args = parser.parse_args()

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert args.np <= os.cpu_count(), f"The number of processes ({args.np}) requested are not available."

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = max(args.np, 1)

    return args_dict

def r2k_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = argparse.ArgumentParser(description="Calculates the grid of points in the k-space")
    parser.add_argument("Mb", type=int, help="Maximum band to consider (>0)")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    parser.add_argument("-mb", type=int, default=0, help="Minimum band to consider (>0; default: 0)")
    args = parser.parse_args()

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert args.np <= os.cpu_count(), f"The number of processes ({args.np}) requested are not available."
    assert args.Mb < d.nbnd, f"The maximum band index ({args.Mb}) must be smaller than {d.nbnd}."


    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = max(args.np, 1)
    args_dict["MIN_BAND"] = max(args.mb, 0)
    args_dict["MAX_BAND"] = args.Mb

    return args_dict

def berry_props_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = argparse.ArgumentParser(description="Calculates the Berry connection and curvature of all the combinations of bands.")
    parser.add_argument("Mb", type=int, help="Maximum band to consider (>0)")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    parser.add_argument("-mb", type=int, default=0, help="Minimum band to consider (>0; default: 0)")
    parser.add_argument("-prop", type=str, default="both", choices=["both", "connection", "curvature"] ,help="Specify which proprety to calculate. (default: both)")
    args = parser.parse_args()
    
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert args.np <= os.cpu_count(), f"The number of processes ({args.np}) requested are not available."
    assert args.Mb < d.nbnd, f"The maximum band index ({args.Mb}) must be smaller than {d.nbnd}."

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = min(max(args.np, 1), args.Mb - args.mb + 1) 
    args_dict["MIN_BAND"] = max(args.mb, 0)
    args_dict["MAX_BAND"] = args.Mb
    args_dict["PROP"] = args.prop
    
    return args_dict


def conductivity_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("bandempty", type=int, help="Index of the conduction band.")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    parser.add_argument("-enermax",  type=float, default=2.5, help="Maximum energy in Ry units (default: 2.5).")
    parser.add_argument("-enerstep", type=float, default=0.001, help="Energy step in Ry units (default: 0.001).")
    parser.add_argument("-broadning",type=float, default=0.01j, help="Energy broading in Ry units (default: 0.01).")
    args = parser.parse_args()
    
    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert args.np <= os.cpu_count(), f"The number of processes ({args.np}) requested are not available."
    assert args.bandempty > d.nbnd, f"The conduction band index ({args.bandempty}) must be smaller than the number of bands ({d.nbnd})."
    assert args.bandempty > d.vb, f"The conduction band index ({args.bandempty}) must be greater than the valence band index ({d.vb})."
    
    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = max(args.np, 1)
    args_dict["ENERMAX"] = args.enermax
    args_dict["ENERSTEP"] = args.enerstep
    args_dict["BROADNING"]= args.broadning
    args_dict["BANDEMPTY"]= args.bandempty

    return args_dict


def shg_cli() -> argparse.Namespace:
    ###########################################################################
    # 1. DEFINING CLI ARGS
    ###########################################################################
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("bandempty", type=int, help="Index of the conduction band.")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    parser.add_argument("-enermax",  type=float, default=2.5, help="Maximum energy in Ry units (default: 2.5).")
    parser.add_argument("-enerstep", type=float, default=0.001, help="Energy step in Ry units (default: 0.001).")
    parser.add_argument("-broadning",type=float, default=0.01j, help="Energy broading in Ry units (default: 0.01).")
    args = parser.parse_args()

    ###########################################################################
    # 2. ASSERTIONS
    ###########################################################################
    assert args.np <= os.cpu_count(), f"The number of processes ({args.np}) requested are not available."
    assert args.bandempty > d.nbnd, f"The conduction band index ({args.bandempty}) must be smaller than the number of bands ({d.nbnd})."
    assert args.bandempty > d.vb, f"The conduction band index ({args.bandempty}) must be greater than the valence band index ({d.vb})."

    ###########################################################################
    # 3. PROCESSING ARGS
    ###########################################################################
    args_dict = {}
    args_dict["NPR"] = max(args.np, 1)
    args_dict["ENERMAX"] = args.enermax
    args_dict["ENERSTEP"] = args.enerstep
    args_dict["BROADNING"]= args.broadning
    args_dict["BANDEMPTY"]= args.bandempty

    ###########################################################################
    return args_dict