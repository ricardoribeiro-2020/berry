"""
    This moldule writes in stdout the k-points in a useful layout
"""

# pylint: disable=C0103
###################################################################################


def _list_kpoints(nkx, nky):
    """Output the list of k-points in a convenient way."""
    print()
    nk = -1
    SEP = " "
    print("         | y  x ->")
    for _ in range(nky):
        lin = ""
        print()
        for _ in range(nkx):
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


def _bands_numbers(nkx, nky, valuesarray):
    """Output the valuesarray numbers."""
    nk = -1
    SEP = " "
    print("         | y  x ->")
    for _ in range(nky):
        lin = ""
        print()
        for _ in range(nkx):
            nk = nk + 1
            if valuesarray[nk] < 0:
                lin += SEP + str(valuesarray[nk])
            elif 0 <= valuesarray[nk] < 10:
                lin += SEP + SEP + str(valuesarray[nk])
            elif 9 < valuesarray[nk] < 100:
                lin += SEP + str(valuesarray[nk])
        print(lin)


def _float_numbers(nkx, nky, valuesarray, precision):
    """Outputs the valuesarray float numbers with the precision number of decimal places."""
    nk = -1
    SEP = " "
    print("         | y  x ->")
    for _ in range(nky):
        lin = ""
        print()
        for _ in range(nkx):
            val = "{0:.{1}f}".format(valuesarray[nk], precision)
            nk = nk + 1
            if valuesarray[nk] < 0:
                lin += SEP + str(val)
            elif 0 <= valuesarray[nk] < 10:
                lin += SEP + SEP + str(val)
            elif 9 < valuesarray[nk] < 100:
                lin += SEP + str(val)
        print(lin)
