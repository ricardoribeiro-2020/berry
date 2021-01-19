###################################################################################
# These routines run the DFT calculations
###################################################################################

import numpy as np
# This is to make operations in the shell
import os
import sys
import subprocess
# This are the subroutines and functions
import loaddata as d

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
def wfck2r(nk1,nb1):
#  print(' Extracting wavefunctions; nk = ',nk1,'; nb = ',nb1)

  directories = str(d.dftdirectory)+'run'+str(nk1)+'-'+str(nb1)
  os.system('mkdir -p '+directories)

  mpi = ''
  comando = "&inputpp  prefix = 'bn' ,\
                       outdir = '../out/' ,\
                      first_k = "+str(nk1+1)+" ,\
                       last_k = "+str(nk1+1)+" ,\
                   first_band = "+str(nb1)+" ,\
                    last_band = "+str(nb1)+" , \
                      loctave = .true., /"
  #print(comando)
  sys.stdout.flush()
  # Name of temporary file
  filename = 'wfc'+str(nk1)+'-'+str(nb1)
  cmd0 = 'cd '+directories+';'
  cmd = cmd0+'echo "'+comando+'"|wfck2r.x > tmp;tail -'+str(d.nr)+' wfck2r.mat'

  output = subprocess.check_output(cmd,shell=True)

#  os.system('echo "'+comando+'"|wfck2r.x > '+str(d.dftdirectory)+'tmp;tail -'+str(d.nr)+' wfck2r.mat>'+filename)
  out1 = output.decode('utf-8').replace(')','j')
  out2 = out1.replace(', -','-')
  out3 = out2.replace(',  ','+').replace('(','')

  psi  = np.fromstring(out3,dtype=complex,sep='\n')

  modulus = np.abs(psi)
  phase = np.angle(psi)

  # Phase that has to be subtracted to all points of the wavefunction
  deltaphase = phase[int(d.rpoint)]

  if modulus[int(d.rpoint)] < 1E-5:
    flag = False
  else:
    flag = True
  # Log to output
  print(nk1,nb1,modulus[int(d.rpoint)],deltaphase,flag)

  phase = phase - deltaphase

  psifinal = modulus*np.vectorize(complex)(np.cos(phase),np.sin(phase))

  # Name of the final wfc file
  outfile = str(d.wfcdirectory)+'k0'+str(nk1)+'b0'+str(nb1)+'.wfc'

  # Save wavefunction to file
  with open(outfile, 'wb') as f:
    np.save(f,psifinal)
  f.close()
#  print(' Wavefunction saved to file '+outfile)

#  for i in range(10):
#    print(psifinal[i])
  os.system('rm -rf '+directories)


  return






