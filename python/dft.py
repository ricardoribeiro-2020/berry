###################################################################################
# These routines run the DFT calculations
###################################################################################

import numpy as np
# This is to make operations in the shell
import os
import sys
import subprocess

# SCF calculation
def scf(mpi,directory,name_scf,outdir,pseudodir):

  scffile = directory + name_scf
  if scffile[-3:] != '.in':
    verifica = os.popen("test -f " + scffile + ".out;echo $?").read()   #if 1\n does not exist if 0\n exists
  else:
    verifica = os.popen("test -f " + scffile[:-3] + ".out;echo $?").read()   #if 1\n does not exist if 0\n exists
  
  if verifica == '1\n':  # If output file does not exist, run the scf calculation
    with open(scffile,'r') as templa:
      original = templa.read()
    templa.closed

    # Changes file names to absolute paths: its safer
    lines = original.split('\n')
    subst = ''
    for line in lines:
      if line.find('outdir') >= 0:
        subst = subst + line.find('outdir')*" "+"outdir = '" + outdir + "' ,\n"
      elif line.find('pseudo_dir') >= 0:
        subst = subst + line.find('pseudo_dir')*" "+"pseudo_dir = '" + pseudodir + "' ,\n"
      else:
        subst = subst + line.rstrip() + '\n'

    # Saves file with absolute paths: overides original!
    with open(scffile,'w') as output:
      output.write(subst)
    output.closed


    print('     Running scf calculation')
    if scffile[-3:] != '.in':
      comando = mpi + "pw.x -i " + scffile + " > " + scffile + ".out"
    else:
      comando = mpi + "pw.x -i " + scffile + " > " + scffile[:-3] + ".out"
    print('     Command: '+comando+'\n')
    os.system(comando)
  return
#    sys.exit("Stop")



# Creates template for nscf calculation
def template(directory,name_scf):

  # opens the template file from which it will do nscf calculations
  scffile = directory + name_scf 
  with open(scffile,'r') as templa:
    original = templa.read()
  templa.closed
  nscf1 = original.replace('automatic','tpiba').replace('scf','nscf')
  nscf2 = nscf1.split('\n')
  nscf2 = list(filter(None,nscf2))

  del nscf2[-1]                          # Delete SCF k points

  nscf = '\n'.join(nscf2)                # Build the template nscf file

#  sys.exit("Stop")
  return nscf




# NSCF calculation ****************************************************
def nscf(mpi,directory,name_nscf,nscf,nkps,kpoints,nbands):

  nscffile = directory + name_nscf
  verifica = os.popen("test -f " + nscffile[:-3] + ".out;echo $?").read()   #if 1\n does not exist if 0\n exists
  
  if verifica == '1\n':  # If output file does not exist, run the nscf calculation

    lines = nscf.split('\n')
    subst = ''
    flag = 0
    for line in lines:
      if line.find('SYSTEM') >= 0:
        flag = 1
      if line.find('nbnd') >= 0:
        rep = line.split()
        subst = subst + line.find('nbnd')*' '+'nbnd = ' + str(nbands) + ' ,\n'
        flag = 2
      elif flag == 1 and line.find('/') >= 0:
        subst = subst + '                        '+'nbnd = ' + str(nbands) + ' ,\n'
        subst = subst + line.rstrip() + '\n'
        flag = 2
      else:
        subst = subst + line.rstrip() + '\n'

    nscf = subst

    with open(nscffile,'w') as output:
      output.write(nscf + str(nkps) + '\n' + kpoints)
    output.closed

#    sys.exit("Stop")

    print('     Running nscf calculation')
    comando = mpi + "pw.x -i " + nscffile + " > " + nscffile[:-3] + ".out"
    print('     Command: '+comando+'\n')
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

  comando = "&inputpp  prefix = '"+str(d.prefix)+"' ,\
                       outdir = '"+str(d.outdir)+"' ,\
                      first_k = "+str(nk1+1)+" ,\
                       last_k = "+str(nk1+1)+" ,\
                   first_band = "+str(nb1+1)+" ,\
                    last_band = "+str(total_bands+1)+" , \
                      loctave = .true., /"
  sys.stdout.flush()
  # Name of temporary file
  filename = 'wfc'+str(nk1)+'-'+str(nb1)
  # comand to send to the shell
  cmd = 'echo "'+comando+'"|'+mpi+'wfck2r.x > tmp;tail -'+str(d.nr*((total_bands-nb1)+1))+' wfck2r.oct'
  # Captures the output of the shell command
  output = subprocess.check_output(cmd,shell=True)
  # Strips the output of fortran formating and converts to numpy formating
  out1 = output.decode('utf-8').replace(')','j')
  out2 = out1.replace(', -','-')
  out3 = out2.replace(',  ','+').replace('(','')
  # puts the wavefunctions into a numpy array
  psi  = np.fromstring(out3,dtype=complex,sep='\n')

  # For each band, find the value of teh wfc at the specific point rpoint (in real space)
  psi_rpoint = np.array([psi[int(d.rpoint)+d.nr*i]for i in range((total_bands-nb1)+1)])
  # Calculate the phase at rpoint for all the bands
  deltaphase = np.arctan2(psi_rpoint.imag,psi_rpoint.real)
  # and the modulus
  mod_rpoint = np.absolute(psi_rpoint) 
  psifinal = []
  # For each band beeing considered
  for i in range((total_bands-nb1)+1):
    # Print to the output file for verification
    print('    %6d  %4d  %12.8f  %12.8f   %r' % (nk1,i+nb1,mod_rpoint[i],deltaphase[i],not mod_rpoint[i] < 1E-5))
    # Subtract the reference phase for each point
    psifinal += list(psi[i*d.nr:(i+1)*d.nr]*np.exp(-1j*deltaphase[i]))
  psifinal = np.array(psifinal)

  # Name of the final wfc file
  outfiles = [str(d.wfcdirectory)+'k0'+str(nk1)+'b0'+str(band)+'.wfc' for band in range(nb1,total_bands+1)] 
  # Save wavefunction to file
  for i,outfile in enumerate(outfiles):
    with open(outfile, 'wb') as f:
      np.save(f,psifinal[i*d.nr:(i+1)*d.nr])
    f.close()



