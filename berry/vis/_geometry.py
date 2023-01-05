import numpy as np
import matplotlib.pyplot as plt

def bcc(band:int, grad:int, type: str):
    berry_conn = np.load(f"berryConn{band}_{grad}.npy")

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
    berry_curv = np.load(f"berryCur{band}_{grad}.npy")

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