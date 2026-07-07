""" Module to include some auxiliary calculations for conductivity and shg. """

from findiff import Gradient

import berry._subroutines.loadmeta as m

# findiff operators are relatively expensive to build; cache one per (dims, dk, acc)
_gradient_cache = {}


def _gradient(dk, acc):
    key = (m.dimensions, dk, acc)
    if key not in _gradient_cache:
        _gradient_cache[key] = Gradient(h=[dk] * m.dimensions, acc=acc)
    return _gradient_cache[key]


def comute(berryConnection, sprime, s, beta, alpha):
    """ Commutator of two Berry connections: [xi^beta_{s's}, xi^alpha_{ss'}]."""
    e = (
        berryConnection[sprime][s][beta] * berryConnection[s][sprime][alpha]
        - berryConnection[sprime][s][alpha] * berryConnection[s][sprime][beta]
    )

    return e


def comute3(berryConnection, sprime, s, r, beta, alpha2, alpha1):
    """ Three-band product of Berry connections:
    xi^beta_{s's} xi^alpha2_{sr} xi^alpha1_{rs'} + xi^alpha1_{s'r} xi^alpha2_{rs} xi^beta_{ss'}."""

    e = (
        berryConnection[sprime][s][beta]
        * berryConnection[s][r][alpha2]
        * berryConnection[r][sprime][alpha1]
        + berryConnection[sprime][r][alpha1]
        * berryConnection[r][s][alpha2]
        * berryConnection[s][sprime][beta]
    )

    return e


def deriv(berryConnection, s, sprime, alpha1, alpha2, dk, acc=2):
    """ Generalized (covariant) derivative of the Berry connection,
    (xi^alpha1_{ss'})_{;alpha2} = grad^alpha2 xi^alpha1_{ss'}
                                  - i (xi^alpha2_{ss} - xi^alpha2_{s's'}) xi^alpha1_{ss'}.

    acc=2 matches the finite-difference accuracy used to build the Berry connections
    and grad_dea.  Not to be confused with berry_geometry.deriv, which is the plain
    gradient (no covariant term) used for the curvature curl at acc=4."""
    a = _gradient(dk, acc)(berryConnection[s][sprime][alpha1])

    e = (
        a[alpha2]
        - 1j
        * (berryConnection[s][s][alpha2] - berryConnection[sprime][sprime][alpha2])
        * berryConnection[s][sprime][alpha1]
    )

    return e


def comutederiv(berryConnection, s, sprime, beta, alpha1, alpha2, dk, acc=2):
    """ Commutator of a Berry connection with a generalized derivative:
    [xi^beta_{s's}, (xi^alpha1_{ss'})_{;alpha2}]."""

    e = (
        berryConnection[sprime][s][beta]
        * deriv(berryConnection, s, sprime, alpha1, alpha2, dk, acc)
        - deriv(berryConnection, sprime, s, alpha1, alpha2, dk, acc)
        * berryConnection[s][sprime][beta]
    )

    return e
