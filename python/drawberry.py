# This program is to read and draw wfc and grad wfc

# This is to include maths
import numpy as np

# This is to draw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D,axes3d

# This is to make operations in the shell
import sys

# This to make parallel processing
import joblib

bandwfc = int(sys.argv[1])
gradwfc = int(sys.argv[2])
tipo = sys.argv[3]

filename = 'berryCon'+str(bandwfc)+'-'+str(gradwfc)

berryConnection = joblib.load(filename+'.gz')


X, Y = np.meshgrid(np.arange(0, 8), np.arange(0, 8))

fig1, ax1 = plt.subplots()
#fig = plt.figure(figsize=(6,6))
#ax = fig.gca(projection='3d')

M = np.hypot(np.real(berryConnection[0]),np.real(berryConnection[1]))   # Colors for real part
Q = np.hypot(np.imag(berryConnection[0]),np.imag(berryConnection[1]))   # Colors for imag part

if tipo == 'a' or tipo == 'r':
  ax1.quiver(np.real(berryConnection[0]),np.real(berryConnection[1]),M, units='x',width=0.042,scale=20 / 1)
if tipo == 'a' or tipo == 'i':
  ax1.quiver(np.imag(berryConnection[0]),np.imag(berryConnection[1]),Q, units='x',width=0.042,scale=20 / 1)
if tipo == 'a' or tipo == 'c':
  ax1.quiver(berryConnection[0],berryConnection[1],color='b')
#ax.plot_wireframe(X,Y,np.real(wfcpos[ponto]))

plt.show()

#sys.exit("Stop")



