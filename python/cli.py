import argparse

def preprocessing_cli() -> argparse.Namespace:
    # Define all cli arguments
    parser = argparse.ArgumentParser(description="")
    
    # Process the arguments given

    # Save them in a dict and return
    
    return

def generatewfc_cli() -> argparse.Namespace:
    # Define all cli arguments
    parser = argparse.ArgumentParser(description="")
    
    # Process the arguments given

    # Save them in a dict and return
    
    return

def dotproduct_cli() -> argparse.Namespace:
    # Define all cli arguments
    parser = argparse.ArgumentParser(description="Calculates the dot product between the neighbouring points of the wavefunction.")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    args = parser.parse_args()
    
    # Process the arguments given
    args_dict = {}
    args_dict["NPR"] = max(args.np, 1)

    # Save them in a dict and return
    return args_dict

def r2k_cli() -> argparse.Namespace:
    # Define all cli arguments
    parser = argparse.ArgumentParser(description="Calculates the grid of points in the k-space")
    parser.add_argument("Mb", type=int, help="Maximum band to consider (>0)")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    parser.add_argument("-mb", type=int, default=0, help="Minimum band to consider (>0; default: 0)")
    args = parser.parse_args()
    
    # Process the arguments given
    args_dict = {}
    args_dict["NPR"] = max(args.np, 1)
    args_dict["MIN_BAND"] = max(args.mb, 0)
    args_dict["MAX_BAND"] = args.Mb
    
    # Save them in a dict and return
    return args_dict

def berry_props_cli() -> argparse.Namespace:
    # Define all cli arguments
    parser = argparse.ArgumentParser(description="Calculates the Berry connection and curvature of all the combinations of bands.")
    parser.add_argument("Mb", type=int, help="Maximum band to consider (>0)")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    parser.add_argument("-mb", type=int, default=0, help="Minimum band to consider (>0; default: 0)")
    parser.add_argument("-prop", type=str, default="both", choices=["both", "connection", "curvature"] ,help="Specify which proprety to calculate. (default: both)")
    args = parser.parse_args()
    
    # Process the arguments given
    args_dict = {}
    args_dict["NPR"] = min(max(args.np, 1), args.Mb - args.mb + 1) 
    args_dict["MIN_BAND"] = max(args.mb, 0)
    args_dict["MAX_BAND"] = args.Mb
    args_dict["PROP"] = args.prop
    
    # Save them in a dict and return
    return args_dict

def conductivity_cli() -> argparse.Namespace:
    # Define all cli arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("bandempty", type=int, help="Index of the conduction band.")
    parser.add_argument("-np", type=int, default=1, help="Number of processes to use (default: 1)")
    parser.add_argument("-enermax",  type=float, default=2.5, help="Maximum energy in Ry units (default: 2.5).")
    parser.add_argument("-enerstep", type=float, default=0.001, help="Energy step in Ry units (default: 0.001).")
    parser.add_argument("-broadning",type=float, default=0.01j, help="Energy broading in Ry units (default: 0.01).")
    args = parser.parse_args()

    # Process the arguments given
    args_dict = {}

    args_dict["NPR"] = max(args.np, 1)
    args_dict["ENERMAX"] = args.enermax
    args_dict["ENERSTEP"] = args.enerstep
    args_dict["BROADNING"]= args.broadning
    args_dict["BANDEMPTY"]= args.bandempty

    # Save them in a dict and return
    return args_dict

def shg_cli() -> argparse.Namespace:
    # Define all cli arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("bandempty", type=int, help="Index of the conduction band.")
    parser.add_argument("-enermax", type=float, default=2.5, help="Maximum energy in Ry units (default: 2.5).")
    parser.add_argument("-enerstep", type=float, default=0.001, help="Energy step in Ry units (default: 0.001).")
    parser.add_argument("-broadning", type=float, default=0.01j, help="Energy broading in Ry units (default: 0.01).")
    args = parser.parse_args()

    # Process the arguments given
    args_dict = {}
    args_dict["ENERMAX"] = args.enermax
    args_dict["ENERSTEP"] = args.enerstep
    args_dict["BROADNING"] = args.broadning
    args_dict["BANDEMPTY"] = args.bandempty

    # Save them in a dict and return
    return args_dict