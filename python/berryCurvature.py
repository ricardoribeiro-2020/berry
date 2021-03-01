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
def berryCurv(gradwfc0,gradwfc1):

# Reading data needed for the run
  nr = d.nr
  print()
  print('     Reading files ./wfcpos'+str(gradwfc0)+'.gz and ./wfcgra'+str(gradwfc1)+'.gz')
  wfcgra0 = joblib.load('./wfcgra'+str(gradwfc0)+'.gz')
  wfcgra1 = joblib.load('./wfcgra'+str(gradwfc1)+'.gz')

  print('     Finished reading data ',str(gradwfc0),' and ',str(gradwfc1),'   {:5.2f}'.format((time.time()-starttime)/60.),' min')
  sys.stdout.flush()
#  sys.exit("Stop")

### Finished reading data
# Calculation of the Berry curvature
  berryCurvature = np.zeros(wfcgra0[0].shape,dtype=complex)

  for posi in range(nr):
    berryCurvature += 1j*wfcgra0[posi][0]*wfcgra1[posi][1].conj() - 1j*wfcgra0[posi][1]*wfcgra1[posi][0].conj()
# we are assuming that normalization is \sum |\psi|^2 = 1
# if not, needs division by nr
  berryCurvature /= nr

  print('     Finished calculating Berry curvature for index '+str(gradwfc0)+'  '+str(gradwfc1)+'.\
                                  \n     Saving results to file   {:5.2f}'.format((time.time()-starttime)/60.),' min')
  sys.stdout.flush()

  filename = './berryCurvature'+str(gradwfc0)+'-'+str(gradwfc1)
# output units of Berry curvature is none

  joblib.dump(berryCurvature,filename+'.gz', compress=3)

  return

###################################################################################
if __name__ == '__main__':
  header('BERRY CURVATURE',time.asctime())

  starttime = time.time()                         # Starts counting time

  if len(sys.argv) == 2:
    print('     Will calculate all combinations of bands from 0 up to '+str(sys.argv[1]))
    for gradwfc0 in range(int(sys.argv[1])+1):
      for gradwfc1 in range(int(sys.argv[1])+1):
        berryCurv(gradwfc0,gradwfc1)

  elif len(sys.argv) == 3:
    print('     Will calculate just for band '+str(sys.argv[1])+' and '+str(sys.argv[2]))
    gradwfc0 = int(sys.argv[1])
    gradwfc1 = int(sys.argv[2])
    berryCurv(gradwfc0,gradwfc1)

  else:
    print('     ERROR in number of arguments. Has to have one or two integers.')
    print('     Stoping.')
    print()

##################################################################################r
# Finished
  endtime = time.time()

  footer(contatempo.tempo(starttime,endtime))


