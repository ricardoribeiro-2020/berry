""" Module to include some commutators for condutivity calculations """

from findiff import Gradient
# pylint: disable=C0103


def comute(berryConnection, sprime, s, beta, alpha):
    """ Commute two Berry connections."""
    s_sprime = str(s) + " " + str(sprime)
    sprime_s = str(sprime) + " " + str(s)
    e = (
        berryConnection[sprime_s][beta] * berryConnection[s_sprime][alpha]
        - berryConnection[sprime_s][alpha] * berryConnection[s_sprime][beta]
    )

    return e


def comute3(berryConnection, sprime, s, r, beta, alpha2, alpha1):
    """ Commute three Berry connections."""
    s_sprime = str(s) + " " + str(sprime)
    sprime_s = str(sprime) + " " + str(s)
    r_s = str(r) + " " + str(s)
    s_r = str(s) + " " + str(r)
    r_sprime = str(r) + " " + str(sprime)
    sprime_r = str(sprime) + " " + str(r)

    e = (
        berryConnection[sprime_s][beta]
        * berryConnection[s_r][alpha2]
        * berryConnection[r_sprime][alpha1]
        + berryConnection[sprime_r][alpha1]
        * berryConnection[r_s][alpha2]
        * berryConnection[s_sprime][beta]
    )

    return e


def deriv(berryConnection, s, sprime, alpha1, alpha2, dk):
    """ Derivative of the Berry connection."""
    grad = Gradient(h=[dk, dk], acc=3)  # Defines gradient function in 2D
    s_sprime = str(s) + " " + str(sprime)
    #    sprime_s = str(sprime) + ' ' + str(s)
    s_s = str(s) + " " + str(s)
    sprime_sprime = str(sprime) + " " + str(sprime)

    a = grad(berryConnection[s_sprime][alpha1])

    e = (
        a[alpha2]
        - 1j
        * (berryConnection[s_s][alpha2] - berryConnection[sprime_sprime][alpha2])
        * berryConnection[s_sprime][alpha1]
    )

    return e


def comutederiv(berryConnection, s, sprime, beta, alpha1, alpha2, dk):
    """ Commute Berry connection and a derivative."""
    s_sprime = str(s) + " " + str(sprime)
    sprime_s = str(sprime) + " " + str(s)

    e = (
        berryConnection[sprime_s][beta]
        * deriv(berryConnection, s, sprime, alpha1, alpha2, dk)
        - deriv(berryConnection, sprime, s, alpha1, alpha2, dk)
        * berryConnection[s_sprime][beta]
    )

    return e
