""" Module to include some commutators for condutivity calculations """

from findiff import Gradient


def comute(berryConnection, sprime, s, beta, alpha):
    """ Commute two Berry connections."""
    e = (
        berryConnection[sprime][s][beta] * berryConnection[s][sprime][alpha]
        - berryConnection[sprime][s][alpha] * berryConnection[s][sprime][beta]
    )

    return e


def comute3(berryConnection, sprime, s, r, beta, alpha2, alpha1):
    """ Commute three Berry connections."""

    e = (
        berryConnection[sprime][s][beta]
        * berryConnection[s][r][alpha2]
        * berryConnection[r][sprime][alpha1]
        + berryConnection[sprime][r][alpha1]
        * berryConnection[r][s][alpha2]
        * berryConnection[s][sprime][beta]
    )

    return e


def deriv(berryConnection, s, sprime, alpha1, alpha2, dk):
    """ Derivative of the Berry connection."""
    grad = Gradient(h=[dk, dk], acc=2)  # Defines gradient function in 2D

    a = grad(berryConnection[s][sprime][alpha1])

    e = (
        a[alpha2]
        - 1j
        * (berryConnection[s][s][alpha2] - berryConnection[sprime][sprime][alpha2])
        * berryConnection[s][sprime][alpha1]
    )

    return e


def comutederiv(berryConnection, s, sprime, beta, alpha1, alpha2, dk):
    """ Commute Berry connection and a derivative."""

    e = (
        berryConnection[sprime][s][beta]
        * deriv(berryConnection, s, sprime, alpha1, alpha2, dk)
        - deriv(berryConnection, sprime, s, alpha1, alpha2, dk)
        * berryConnection[s][sprime][beta]
    )

    return e
