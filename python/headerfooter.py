""" Module with the header and footer that appears in the output of the programas."""


def header(title, time):
    """ Header that appears in the output of the programas."""

    print()
    print("     Program " + title + " v.0.2 starts on " + time)
    print()
    print("     This program is part of the open-source BERRY suite.")
    print("         https://ricardoribeiro-2020.github.io/berry/")
    print()


def footer(time):
    """ Footer that appears in the output of the programas."""

    print()
    print()
    print("     " + time)

    print()
    print(
        "=------------------------------------------------------------------------------="
    )
    print("      Program finished.")
    print(
        "=------------------------------------------------------------------------------="
    )
