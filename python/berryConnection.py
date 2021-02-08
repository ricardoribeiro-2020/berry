###################################################################################
# This program calculates the Berry connections
###################################################################################

# This is to include maths
import numpy as np

# This is to make operations in the shell
import sys
import time

# This are the subroutines and functions
import contatempo
from headerfooter import header,footer
import loaddata as d

# This to make parallel processing
import joblib

###################################################################################
def berryConnect(bandwfc,gradwfc):

# Reading data needed for the run
  nr = d.nr
  print()
  print(' Reading dump files')
  wfcpos = joblib.load('./wfcpos'+str(bandwfc)+'.gz')
  wfcgra = joblib.load('./wfcgra'+str(gradwfc)+'.gz')

  print(' Finished reading data')
  sys.stdout.flush()
#  sys.exit("Stop")

### Finished reading data
# Calculation of the Berry connection
  berryConnection = np.zeros(wfcgra[0].shape,dtype=complex)

  for posi in range(nr):
    berryConnection += 1j*wfcpos[posi].conj()*wfcgra[posi] 
# we are assuming that normalization is \sum |\psi|^2 = 1
# if not, needs division by nr
  berryConnection /= nr

  print(' Finished calculating Berry connection for index '+str(bandwfc)+'  '+str(gradwfc)+'.\
                                  \n Saving results to file')
  sys.stdout.flush()

  filename = './berryCon'+str(bandwfc)+'-'+str(gradwfc)
# output units of Berry connection are bohr

  joblib.dump(berryConnection,filename+'.gz', compress=3)

  return

###################################################################################
if __name__ == '__main__':
  header('BERRY CONNECTION',time.asctime())

  starttime = time.time()                         # Starts counting time

  if len(sys.argv) == 2:
    print(' Will calculate all combinations of bands from 0 up to '+str(sys.argv[1]))
    for bandwfc in range(int(sys.argv[1])+1):
      for gradwfc in range(int(sys.argv[1])+1):
        berryConnect(bandwfc,gradwfc)

  elif len(sys.argv) == 3:
    print(' Will calculate just for band '+str(sys.argv[1])+' and '+str(sys.argv[2]))
    bandwfc = int(sys.argv[1])
    gradwfc = int(sys.argv[2])
    berryConnect(bandwfc,gradwfc)

  else:
    print(' ERROR in number of arguments. Has to have one or two integers.')
    print(' Stoping.')
    print()

##################################################################################r
# Finished
  endtime = time.time()

  footer(contatempo.tempo(starttime,endtime))


