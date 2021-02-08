###################################################################################
# This program calculates the Berry curvature
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

  gradwfc0 = int(sys.argv[1])
  gradwfc1 = int(sys.argv[2])

# Reading data needed for the run
  nr = d.nr
  print(' Total number of points in real space:',nr)
  print()
  print(' Start reading dump files')
  wfcgra0 = joblib.load('./wfcgra'+str(gradwfc0)+'.gz')
  wfcgra1 = joblib.load('./wfcgra'+str(gradwfc1)+'.gz')

  print(' Finished reading data')
  sys.stdout.flush()
#  sys.exit("Stop")
################################################## Finished reading data
# Calculation of the Berry curvature
  berryCurvature = np.zeros(wfcgra0[0].shape,dtype=complex)

  for posi in range(nr):
    berryCurvature += 1j*wfcgra0[posi][0]*wfcgra1[posi][1].conj() - 1j*wfcgra0[posi][1]*wfcgra1[posi][0].conj()

  berryCurvature /= nr

  print(' Finished calculating Berry curvature for index '+str(gradwfc0)+'  '+str(gradwfc1)+'.\
                                  \n Saving results to file')
  sys.stdout.flush()

  filename = './berryCurvature'+str(gradwfc0)+'-'+str(gradwfc1)
# output units of Berry connection are bohr

  joblib.dump(berryCurvature,filename+'.gz', compress=3)


#sys.exit("Stop")

##################################################################################r
# Finished
  endtime = time.time()

  footer(contatempo.tempo(starttime,endtime))



