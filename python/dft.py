###################################################################################
# These routines run the DFT calculations
###################################################################################

import numpy as np
# This is to make operations in the shell
import os
import sys
import subprocess

# SCF calculation
def scf(mpi,directory,name_scf):

  scffile = directory + name_scf
  verifica = os.popen("test -f " + scffile + ".out;echo $?").read()   #if 1\n does not exist if 0\n exists
  
  if verifica == '1\n':  # If output file does not exist, run the scf calculation
    print(' Running scf calculation')
    comando = mpi + "pw.x -i " + scffile +".in" + " > " + scffile + ".out"
    print(' Command: '+comando+'\n')
    os.system(comando)
  return




# Creates template for nscf calculation
def template(directory,name_scf):

  # opens the template file from which it will do nscf calculations
  scffile = directory + name_scf +".in"
  with open(scffile,'r') as templa:
    original = templa.read()
  templa.closed
  nscf1 = original.replace('automatic','tpiba').replace('scf','nscf')

  nscf2 = nscf1.split('\n')
  for i in range(1,5):                   # Clean empty lines in the end of the file
    if nscf2[-1] == '':
      del nscf2[-1]

  del nscf2[-1]                          # Delete SCF k points

  nscf = '\n'.join(nscf2)                # Build the template nscf file

  return nscf




# NSCF calculation ****************************************************
def nscf(mpi,directory,name_nscf,nscf,nkps,kpoints,nbands):

  nscffile = directory + name_nscf
  verifica = os.popen("test -f " + nscffile + ".out;echo $?").read()   #if 1\n does not exist if 0\n exists
  
  if verifica == '1\n':  # If output file does not exist, run the nscf calculation

    lines = nscf.split('\n')
    subst = ''
    for line in lines:
      if line.find('nbnd') >= 0:
        rep = line.split()
        subst = subst + line.find('nbnd')*' '+'nbnd = ' + str(nbands) + ' ,\n'
      else:
        subst = subst + line.rstrip() + '\n'

    nscf = subst

    with open(nscffile+'.in','w') as output:
      output.write(nscf + str(nkps) + '\n' + kpoints)
    output.closed

#    sys.exit("Stop")

    print(' Running nscf calculation')
    comando = mpi + "pw.x -i " + nscffile +".in" + " > " + nscffile + ".out"
    print(' Command: '+comando+'\n')
    os.system(comando)
  return





# wfck2r calculation and wavefunctions extraction *********************
def wfck2r(nk1,nb1,total_bands=0):
# These are the subroutines and functions
  import loaddata as d 

  if int(d.npr) == 1:
    mpi = ''
  else:
    mpi = 'mpirun -np ' + str(d.npr) + ' '

  comando = "&inputpp  prefix = 'bn' ,\
                       outdir = '"+str(d.dftdirectory)+"out/' ,\
                      first_k = "+str(nk1+1)+" ,\
                       last_k = "+str(nk1+1)+" ,\
                   first_band = "+str(nb1+1)+" ,\
                    last_band = "+str(total_bands+1)+" , \
                      loctave = .true., /"
  sys.stdout.flush()
  # Name of temporary file
  filename = 'wfc'+str(nk1)+'-'+str(nb1)

  cmd = 'echo "'+comando+'"|'+mpi+'wfck2r.x > tmp;tail -'+str(d.nr*((total_bands-nb1)+1))+' wfck2r.mat'

  output = subprocess.check_output(cmd,shell=True)

  out1 = output.decode('utf-8').replace(')','j')
  out2 = out1.replace(', -','-')
  out3 = out2.replace(',  ','+').replace('(','')

  psi  = np.fromstring(out3,dtype=complex,sep='\n')


  psi_rpoint = np.array([psi[int(d.rpoint)+d.nr*i]for i in range((total_bands-nb1)+1)])
  deltaphase = np.arctan2(psi_rpoint.imag,psi_rpoint.real)
  mod_rpoint = np.absolute(psi_rpoint) 
  psifinal = []
  for i in range((total_bands-nb1)+1):
    print(nk1,i+nb1,mod_rpoint[i],deltaphase[i],not mod_rpoint[i] < 1E-5)
    psifinal += list(psi[i*d.nr:(i+1)*d.nr]*np.exp(-1j*deltaphase[i]))
  psifinal = np.array(psifinal)

  # Name of the final wfc file
  outfiles = [str(d.wfcdirectory)+'k0'+str(nk1)+'b0'+str(band)+'.wfc' for band in range(nb1,total_bands+1)] 
  # Save wavefunction to file
  for i,outfile in enumerate(outfiles):
    with open(outfile, 'wb') as f:
      np.save(f,psifinal[i*d.nr:(i+1)*d.nr])
    f.close()

#  os.system('rm -rf '+directories)



