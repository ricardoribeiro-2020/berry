###################################################################################
# This program  caluculates the Berry connections
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
if __name__ == '__main__':
  header('BERRY',time.asctime())

  starttime = time.time()                         # Starts counting time

  if len(sys.argv)!=3:
    print(' ERROR in number of arguments. Has to have two integers.')
    sys.exit("Stop")

  bandwfc = int(sys.argv[1])
  gradwfc = int(sys.argv[2])
#print(str(bandwfc),str(gradwfc))

# Reading data needed for the run
  nr = d.nr
  print(' Total number of points in real space:',nr)
  print()
  print(' Start reading dump files')
  wfcpos = joblib.load('./wfcpos'+str(bandwfc)+'.gz')
  wfcgra = joblib.load('./wfcgra'+str(gradwfc)+'.gz')

  print(' Finished reading data')
  sys.stdout.flush()
#sys.exit("Stop")

################################################## Finished reading data
################################################## Calculation of the Berry connection

  berryConnection = 1j*wfcpos[0].conj()*wfcgra[0]

  for posi in range(1,nr):
    berryConnection += 1j*wfcpos[posi].conj()*wfcgra[posi] 
# we are assuming that normalization is \sum |\psi|^2 = 1
# if not, needs division by nr
  berryConnection /= nr

  print(' Finished calculating Berry connection for index '+str(bandwfc)+'  '+str(gradwfc)+'.\
                                  \n Saving results to file')
  sys.stdout.flush()

  filename = './berryCon'+str(bandwfc)+'-'+str(gradwfc)
#print(filename)
# output units of Berry connection are bohr

  joblib.dump(berryConnection,filename+'.gz', compress=3)



#sys.exit("Stop")


##################################################################################r
# Finished
  endtime = time.time()

  footer(contatempo.tempo(starttime,endtime))










