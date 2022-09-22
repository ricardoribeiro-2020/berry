def numba_njit(func):
    """
    Decorator that tests if numba is installed and if so, uses it.
    """
    try:
        import numba
        return numba.njit(func)
    except ImportError:
        return func