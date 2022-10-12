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

    H = BERRY_LOGO \
        + f'''
         Program {title} {version} starts on {time}

         This program is part of the open-source BERRY suite.
             https://ricardoribeiro-2020.github.io/berry/

    '''

    return H


def footer(time):
    """Footer that appears in the output of the programas."""

    F = f'''
    
        {time}
    
    =------------------------------------------------------------------------------=
        Program finished.
    =------------------------------------------------------------------------------=
    '''
    return F
