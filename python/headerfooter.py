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
    """ Header that appears in the output of the programas."""

    print()
    print(BERRY_LOGO)
    print("     Program " + title + " " + version + " starts on " + time)
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
