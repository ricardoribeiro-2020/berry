""" Module with the header and footer that appears in the output of the programas."""

# pylint: disable=C0103

BERRY_LOGO = '''
                     .     .=*=             .     .
                     .      #@%:        ...       .
                     .           .    ..          .
                      .           +++:           . 
                       .        ==+=+++=        .  
                        ..   .-++======++=.    .   
                          . .*=++=====++=++- .     
                           *=+*@#.===+@%:==++      
                          .-=++++=====+*=++: .     
                        ..   -+++======*==     .   
                       .       :+++==*==        .  
                      .           =+++           . 
                     .           .    .           .
                     .        ..        .#@#.     .

'''

def header(title, version, time):
    """Header that appears in the output of the programas."""

    print()
    print(BERRY_LOGO)
    print(f"\tProgram {title} {version} stars on {time}")
    print()
    print("\tThis program is part of the open-source BERRY suite.")
    print("\t\thttps://ricardoribeiro-2020.github.io/berry/")
    print()


def footer(time):
    """Footer that appears in the output of the programas."""

    print()
    print()
    print(f"\t{time}")

    print()
    print(
        "=------------------------------------------------------------------------------="
    )
    print("\tProgram finished.")
    print(
        "=------------------------------------------------------------------------------="
    )
