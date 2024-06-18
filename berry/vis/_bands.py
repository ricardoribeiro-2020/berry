import time

import numpy as np
import matplotlib.pyplot as plt

from berry._subroutines.contatempo import tempo
from berry._subroutines.headerfooter import header, footer
import berry._subroutines.loadmeta as m
import berry._subroutines.loaddata as d

def corrected(args):
    header("DRAWBANDS", m.version, time.asctime())

    starttime = time.time()  # Starts counting time

    startband = args.mb - m.initial_band  # Number of the first band
    endband = args.Mb - m.initial_band # Number of the last band

    fig = plt.figure(figsize=(6, 6))

    cores = [
        "black",
        "blue",
        "green",
        "red",
        "grey",
        "brown",
        "violet",
        "seagreen",
        "dimgray",
        "darkorange",
        "royalblue",
        "darkviolet",
        "maroon",
        "yellowgreen",
        "peru",
        "steelblue",
        "crimson",
        "silver",
        "magenta",
        "yellow",
    ]

    # Reading data needed for the run

    wfcdirectory = str(m.wfcdirectory)
    print(" Directory where the wfc are:", wfcdirectory)
    nkx = m.nkx
    nky = m.nky
    nkz = m.nkz
    print(" Number of k-points in each direction:", nkx, nky, nkz)
    nks = m.nks
    print(" Total number of k-points:", nks)
    nbnd = m.nbnd
    print(" Number of bands:", nbnd)
    print()
    eigenvalues = d.eigenvalues[:, m.initial_band:]
    print(" Eigenvalues loaded")
    kpoints = d.kpoints
    print(" K-points loaded")

    with open(m.data_dir+"/bandsfinal.npy", "rb") as f:
        bandsfinal = np.load(f)
    f.close()
    print(" bandsfinal loaded")

    xarray = np.zeros((nkx, nky))
    yarray = np.zeros((nkx, nky))
    zarray = np.zeros((nkx, nky))
    count = -1
    for j in range(nky):
        for i in range(nkx):
            count = count + 1
            xarray[i, j] = kpoints[count, 0]
            yarray[i, j] = kpoints[count, 1]

    ax = fig.add_subplot(projection='3d')
    for banda in range(startband, endband + 1):
        count = -1
        for j in range(nky):
            for i in range(nkx):
                count = count + 1
                zarray[i, j] = eigenvalues[count, bandsfinal[count, banda]]

        ax.plot_wireframe(xarray, yarray, zarray, color=cores[banda])

    # Para desenhar no mathematica!
    #
    # print('b'+str(banda)+'={', end = '')
    # for beta in range(nky):
    #   print('{', end = '')
    #   for alfa in range(nkx):
    #     if alfa != nkx-1:
    #       print(str(zarray[alfa][beta])+str(','), end = '')
    #     else:
    #       print(str(zarray[alfa][beta]), end = '')
    #   if beta != nky-1:
    #     print('},')
    #   else:
    #     print('}', end = '')
    # print('};\n')


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot_trisurf(xarray, yarray, zarray, linewidth=0.2, antialiased=True)

    plt.show()


    #    sys.exit("Stop")

    # Finished
    endtime = time.time()

    footer(tempo(starttime, endtime))

def machine(args):
    header("DRAWBANDS", m.version, time.asctime())

    starttime = time.time()  # Starts counting time

    startband = args.mb - m.initial_band # Number of the first band
    endband = args.Mb - m.initial_band # Number of the last band

    fig = plt.figure(figsize=(6, 6))

    cores = [
        "black",
        "blue",
        "green",
        "red",
        "grey",
        "brown",
        "violet",
        "seagreen",
        "dimgray",
        "darkorange",
        "royalblue",
        "darkviolet",
        "maroon",
        "yellowgreen",
        "peru",
        "steelblue",
        "crimson",
        "silver",
        "magenta",
        "yellow",
    ]

    # Reading data needed for the run

    wfcdirectory = str(m.wfcdirectory)
    print(" Directory where the wfc are:", wfcdirectory)
    nkx = m.nkx
    nky = m.nky
    nkz = m.nkz
    print(" Number of k-points in each direction:", nkx, nky, nkz)
    nks = m.nks
    print(" Total number of k-points:", nks)
    nbnd = m.nbnd
    print(" Number of bands:", nbnd)
    print()
    eigenvalues = d.eigenvalues[:, m.initial_band:]
    print(" Eigenvalues loaded")
    kpoints = d.kpoints
    print(" K-points loaded")


    xarray = np.zeros((nkx, nky))
    yarray = np.zeros((nkx, nky))
    zarray = np.zeros((nkx, nky))
    count = -1
    for j in range(nky):
        for i in range(nkx):
            count = count + 1
            xarray[i, j] = kpoints[count, 0]
            yarray[i, j] = kpoints[count, 1]

    ax = fig.add_subplot(projection='3d')
    for banda in range(startband, endband + 1):
        count = -1
        for j in range(nky):
            for i in range(nkx):
                count = count + 1
                zarray[i, j] = eigenvalues[count, banda]

        ax.plot_wireframe(xarray, yarray, zarray, color=cores[banda])

    # Para desenhar no mathematica!
    #
    # print('b'+str(banda)+'={', end = '')
    # for beta in range(nky):
    #   print('{', end = '')
    #   for alfa in range(nkx):
    #     if alfa != nkx-1:
    #       print(str(zarray[alfa][beta])+str(','), end = '')
    #     else:
    #       print(str(zarray[alfa][beta]), end = '')
    #   if beta != nky-1:
    #     print('},')
    #   else:
    #     print('}', end = '')
    # print('};\n')


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot_trisurf(xarray, yarray, zarray, linewidth=0.2, antialiased=True)

    plt.show()


    #    sys.exit("Stop")

    # Finished
    endtime = time.time()

    footer(tempo(starttime, endtime))

def bands(args):
    if args.bands_vis == "corrected":
        corrected(args)
    elif args.bands_vis == "machine":
        machine(args)
