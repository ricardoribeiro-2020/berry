"""
    This moldule writes in stdout the k-points in a useful layout
"""

def list_kpoints(nkx, nky):
    """Output the list of k-points in a convenient way."""
    print()
    nk = -1
    SEP = " "
    print("         | y  x ->")
    for j in range(nky):
        lin = ""
        print()
        for i in range(nkx):
            nk = nk + 1
            if nk < 10:
                lin += SEP + SEP + SEP + SEP + str(nk)
            elif 9 < nk < 100:
                lin += SEP + SEP + SEP + str(nk)
            elif 99 < nk < 1000:
                lin += SEP + SEP + str(nk)
            elif 999 < nk < 10000:
                lin += SEP + str(nk)
        print(lin)


def bands_numbers(nkx, nky, valuesarray):
    """Output the valuesarray numbers."""
    nk = -1
    SEP = " "
    print("         | y  x ->")
    for j in range(nky):
        lin = ""
        print()
        for i in range(nkx):
            nk = nk + 1
            if valuesarray[nk] < 0:
                lin += SEP + str(valuesarray[nk])
            elif 0 <= valuesarray[nk] < 10:
                lin += SEP + SEP + str(valuesarray[nk])
            elif 9 < valuesarray[nk] < 100:
                lin += SEP + str(valuesarray[nk])
        print(lin)
