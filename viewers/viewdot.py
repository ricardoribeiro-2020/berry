"""
 This program reads the dot products and prints them
"""
import numpy as np
import loaddata as d

###################################################################################
if __name__ == "__main__":
    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Total number of k-points:", d.nks)
    print("     Number of bands:", d.nbnd)

    connections = np.load("dp.npy")

    print(connections.shape)

    for i in range(d.nks):
        for j in range(4):
            nk = d.neighbors[i, j]
            print("nk = ", i, "neig = ", nk)
            for band in range(d.nbnd):
                line = "  "
                for band1 in range(d.nbnd):
                    if connections[i, j, band, band1] > 0.1:
                        line = (
                            line
                            + "{:0.1f}".format(connections[i, j, band, band1])
                            + " "
                        )
                    else:
                        line = line + "    "
                print(line)
