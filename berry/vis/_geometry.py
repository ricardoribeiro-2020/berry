import numpy as np
import sys
import matplotlib.pyplot as plt
try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass

def bcc(band:int, grad:int, type: str):

    if band < m.initial_band or band > m.final_band or grad < m.initial_band or grad > m.final_band:
        print(f"\n\tBands have to be between {m.initial_band} and {m.final_band}.")
        print("\tExiting.")
        sys.exit(0)

    berry_conn = np.load(f"{m.geometry_dir}/berryConn{band}_{grad}.npy")

    if m.dimensions == 1:
        print(np.real(berry_conn[0]))
        print(np.imag(berry_conn[0]))
        plt.title(f"Berry connection   {band},{grad}")
        plt.plot(np.real(berry_conn[0]), label='Real')
        plt.plot(np.imag(berry_conn[0]), label='Imag')
        plt.legend()

    elif m.dimensions == 2:
        M = np.hypot(np.real(berry_conn[0]), np.real(berry_conn[1]))  # Colors for real part
        Q = np.hypot(np.imag(berry_conn[0]), np.imag(berry_conn[1]))  # Colors for imag part
     
        if type == "all":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.set_title("real space")
            ax1.quiver(np.real(berry_conn[0]), np.real(berry_conn[1]), M, units="x", width=0.042, scale=20 / 1,)
            
            ax2.set_title("imag space")
            ax2.quiver(np.imag(berry_conn[0]), np.imag(berry_conn[1]), Q, units="x", width=0.042, scale=20 / 1,)
        else:
            fig, ax1 = plt.subplots()
            if type == "real":
                ax1.set_title("real")
                ax1.quiver(np.real(berry_conn[0]), np.real(berry_conn[1]), M, units="x", width=0.042, scale=20 / 1,)
            elif type in "imag":
                ax1.set_title("imag")
                ax1.quiver(np.imag(berry_conn[0]), np.imag(berry_conn[1]), Q, units="x", width=0.042, scale=20 / 1,)

        # Add fig.suptitle() in bold
        fig.suptitle(f"Berry Connections {band}_{grad}\n", fontweight="bold")

    plt.show()


  

def bcr(band:int, grad:int, type: str):

    if m.dimensions == 1:
        print("\n\tBerry curvature is not defined for one dimensional systems.")
        print("\tExiting.")
        sys.exit(0)

    if band < m.initial_band or band > m.final_band or grad < m.initial_band or grad > m.final_band:
        print(f"\n\tBands have to be between {m.initial_band} and {m.final_band}.")
        print("\tExiting.")
        sys.exit(0)

    berry_curv = np.load(f"{m.geometry_dir}/berryCur{band}_{grad}.npy")

    if m.dimensions == 2:
        M = np.hypot(np.real(berry_curv[0]), np.real(berry_curv[1]))  # Colors for real part
        Q = np.hypot(np.imag(berry_curv[0]), np.imag(berry_curv[1]))  # Colors for imag part

        if type == "all":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.set_title("real space")
            ax1.quiver(np.real(berry_curv[0]), np.real(berry_curv[1]), M, units="x", width=0.042, scale=20 / 1,)
            
            ax2.set_title("imag space")
            ax2.quiver(np.imag(berry_curv[0]), np.imag(berry_curv[1]), Q, units="x", width=0.042, scale=20 / 1,)
            
        else:
            fig, ax1 = plt.subplots()
            if type == "real":
                ax1.set_title("real")
                ax1.quiver(np.real(berry_curv[0]), np.real(berry_curv[1]), M, units="x", width=0.042, scale=20 / 1,)
            elif type in "imag":
                ax1.set_title("imag")
                ax1.quiver(np.imag(berry_curv[0]), np.imag(berry_curv[1]), Q, units="x", width=0.042, scale=20 / 1,)

        # Add fig.suptitle() in bold
        fig.suptitle(f"Berry Curvatures {band}_{grad}\n", fontweight="bold")

    plt.show()


def geometry(args):
    if args.geometry_vis == "bcc":
        bcc(args.band, args.grad, args.space)
    elif args.geometry_vis == "bcr":
        bcr(args.band, args.grad, args.space)