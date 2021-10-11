import numpy as np

import sys
import time
import os

from sklearn import neighbors

import contatempo
from headerfooter import header,footer
import loaddata as d

from functions import evaluate_ReciprocalSpace, get_kstarts, correct_dotProduct, correct_bands, correct_dPNeighs, Energy_criteria, count_points, make_adjacentList

#DEFINE VARIABLES
TOL = 0.95
NUM_KSTART = 1

if __name__ == "__main__":
    header("CORRECT", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) > 1:  # To enter an initial value for k-point (optional)
        firskpoint = int(sys.argv[1])
    else:
        firskpoint = -1
    
    if not os.path.exists(d.path+'output/'):
        print('Make Output directory')
        os.mkdir(d.path+'output/')
    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Total number of k-points:", d.nks)
    print("     Number of bands:", d.nbnd)
    print()
    print("     Neighbors loaded")
    print("     Eigenvalues loaded")
    
    Matrix = np.empty((d.nkx,d.nky),int)
    kpoints_index = np.empty((d.nks,2),int)
    count = 0
    for j in range(d.nky):
        for i in range(d.nkx):
          kpoints_index[count]=[i,j]
          Matrix[i,j] = count
          count += 1

    connections = np.load(d.path+"dp.npy")
    print("     Modulus of direct product loaded")

    print()
    print("     Finished reading data\n")
    print()

    errors = evaluate_ReciprocalSpace(connections, d.neighbors, TOL)
    print('\nEvaluate the initial state with a tolerance of ',TOL)
    print('     The score of the initial state is: ', np.sum(errors))
    print('         score 2: ',np.sum(errors == 2))
    print('         score 1: ',np.sum(errors == 1))
    print('         score 0: ',np.sum(errors == 0))

    k_starts = get_kstarts(errors)
    print('     The number of points with')
    print('        All the bands correct: ',len(k_starts[0]))
    print('        10 bands correct:      ',len(k_starts[1]))
    print('        8 bands correct:       ',len(k_starts[2]))
    print('        5 bands correct:       ',len(k_starts[3]))
    print('        3 bands correct:       ',len(k_starts[4]))

    scores = count_points(errors)
    for bn,score in enumerate(scores):
        print(f'Band {bn} Score = {score}')

    
    
    aux = k_starts[0] + k_starts[1] + k_starts[2] + k_starts[3]
    Ti = int((d.nkx-1)/3)
    Tj = int((d.nky-1)/3)
    ni = Ti
    Ni = int(2*Ti)
    nj = Tj
    Nj = int(2*Tj)
    for k in aux:
        ik,jk = kpoints_index[k]
        if not (ik > ni and ik < Ni and jk > nj and jk < Nj):
            for i in range(4):
                if k in k_starts[i]:
                    k_starts[i].remove(k)

    for kstarts in k_starts:
        if len(kstarts) > 10:
           list_kstart = np.random.choice(kstarts,NUM_KSTART) 
           break

    bands_final,_ = np.meshgrid(np.arange(0,d.nbnd),np.arange(0,d.nks))
    list_bandsDone = []
    kErrors = [[] for i in range(d.nkx*d.nky)]
    BandsEnergy = np.empty((d.nbnd,d.nkx,d.nky),float)

    for bn in range(d.nbnd):
      count = -1
      zarray = np.empty((d.nkx,d.nky),float)
      for j in range(d.nky):
        for i in range(d.nkx):
          count += 1
          zarray[i,j] = d.eigenvalues[count, bands_final[count,bn]]
      BandsEnergy[bn] = zarray
    print('\nEnergy Score')
    DScore = []
    for bn,band in enumerate(BandsEnergy):
        dx,dy = np.gradient(band)
        fourier_dx = np.sum(np.abs((np.fft.rfft2(dx))))
        fourier_dy = np.sum(np.abs((np.fft.rfft2(dy))))
        print(f'bn: {bn} F(Grad)= {fourier_dx}, {fourier_dy}\t SCORE= {fourier_dx+fourier_dy}')
        DScore.append(fourier_dx+fourier_dy)

    print(f'TOTAL SCORE: {np.sum(scores)} {np.sum(DScore)}')
    print('\nThe kstarts are ',list_kstart)
    X = make_adjacentList(kpoints_index,BandsEnergy,12)

    with open(d.path+"adjacentList.npy", "wb") as f:
            np.save(f, X)
    f.close()
    with open(d.path+"bandsfinal.npy", "wb") as f:
            np.save(f, bands_final)
    f.close()

    with open(d.path+"output/initial_scores.npy", "wb") as f:
            scores = np.array(scores)
            np.save(f, scores)
    f.close()
    with open(d.path+"output/initial_bandScores.npy", "wb") as f:
            DScore = np.array(DScore)
            np.save(f, DScore)
    f.close()
    ans = input()
    if ans == '-1':
        exit()
    
        
    for k_start in list_kstart:
        print('K_start ',k_start)
        list_k = [k_start]
        list_kdone = []
        for k in list_k:
            for i_neig,dp in enumerate(connections[k]):
                neig = d.neighbors[k,i_neig]
                if neig != -1:
                    for bn, row in enumerate(dp):
                        ik_neig, jk_neig = kpoints_index[neig]
                        ik, jk = kpoints_index[k]
                        bm = np.argmax(row)
                        flag = not (np.all(np.any(np.array([bm,bn]) == np.array(kErrors[neig]),axis=0)) or \
                                    np.all(np.any(np.array([bn,bm]) == np.array(kErrors[neig]),axis=0)))
                        if flag and bm > bn and bm == Energy_criteria(BandsEnergy[bn],BandsEnergy[:,ik_neig,jk_neig],i_neig,(ik,jk),(ik_neig,jk_neig)):
                            for j_neig in range(4):
                                k_neig_neig = d.neighbors[neig,j_neig]
                                aux_i = j_neig+2-4*(j_neig+2>3)
                                connections[k_neig_neig,aux_i] = np.copy(correct_dotProduct(connections[k_neig_neig,aux_i],bn,bm))
                            connections[neig] = np.copy(correct_dPNeighs(connections[neig],bn,bm))
                            bands_final[neig] = np.copy(correct_bands(bands_final[neig],bn,bm))
                            BandsEnergy[bn,ik_neig, jk_neig], BandsEnergy[bm,ik_neig, jk_neig] = BandsEnergy[bm,ik_neig, jk_neig], BandsEnergy[bn,ik_neig, jk_neig]
                            kErrors[neig].append([bn,bm])
                            print(neig,bn,bm)
                    if neig not in list_k and neig not in list_kdone:
                        list_k.append(neig)
            list_kdone.append(k)
    
    for kstarts in k_starts:
        if len(kstarts) > 10:
           list_kstart = np.random.choice(kstarts,NUM_KSTART) 
           break

    for k_start in list_kstart:
        print('K_start ',k_start)
        list_k = [k_start]
        list_kdone = []
        for k in list_k:
            for i_neig,dp in enumerate(connections[k]):
                neig = d.neighbors[k,i_neig]
                if neig != -1:
                    for bn, row in enumerate(dp):
                        ik_neig, jk_neig = kpoints_index[neig]
                        ik, jk = kpoints_index[k]
                        bm = Energy_criteria(BandsEnergy[bn],BandsEnergy[:,ik_neig,jk_neig],i_neig,(ik,jk),(ik_neig,jk_neig))
                        flag = not (np.all(np.any(np.array([bm,bn]) == np.array(kErrors[neig]),axis=0)) or \
                                    np.all(np.any(np.array([bn,bm]) == np.array(kErrors[neig]),axis=0)))
                        if flag and bm > bn and row[bm] > TOL:
                            for j_neig in range(4):
                                k_neig_neig = d.neighbors[neig,j_neig]
                                aux_i = j_neig+2-4*(j_neig+2>3)
                                connections[k_neig_neig,aux_i] = np.copy(correct_dotProduct(connections[k_neig_neig,aux_i],bn,bm))
                            connections[neig] = np.copy(correct_dPNeighs(connections[neig],bn,bm))
                            bands_final[neig] = np.copy(correct_bands(bands_final[neig],bn,bm))
                            BandsEnergy[bn,ik_neig, jk_neig], BandsEnergy[bm,ik_neig, jk_neig] = BandsEnergy[bm,ik_neig, jk_neig], BandsEnergy[bn,ik_neig, jk_neig]
                            kErrors[neig].append([bn,bm])
                            print(neig,bn,bm)
                    if neig not in list_k and neig not in list_kdone:
                        list_k.append(neig)
            list_kdone.append(k)

    
    X = make_adjacentList(kpoints_index,BandsEnergy,12)
    with open(d.path+"adjacentList.npy", "wb") as f:
            np.save(f, X)
    with open(d.path+"bandsfinal.npy", "wb") as f:
            np.save(f, bands_final)
    f.close()

    errors = evaluate_ReciprocalSpace(connections, d.neighbors, TOL)
    print('\nEvaluate the final state with a tolerance of ',TOL)
    print('     The score of the initial state is: ', np.sum(errors))
    print('         score 2: ',np.sum(errors == 2))
    print('         score 1: ',np.sum(errors == 1))
    print('         score 0: ',np.sum(errors == 0))
    scores = count_points(errors)
    for bn,score in enumerate(scores):
        print(f'Band {bn} Score = {score}')
    DScore = []
    print('\nEnergy Score')
    for bn,band in enumerate(BandsEnergy):
        dx,dy = np.gradient(band)
        fourier_dx = np.sum(np.abs((np.fft.rfft2(dx))))
        fourier_dy = np.sum(np.abs((np.fft.rfft2(dy))))
        print(f'bn: {bn} F(Grad)= {fourier_dx}, {fourier_dy}\t SCORE= {fourier_dx+fourier_dy}')
        DScore.append(fourier_dx+fourier_dy)
    print(f'TOTAL SCORE: {np.sum(scores)} {np.sum(DScore)}')

    with open(d.path+"output/final_scores.npy", "wb") as f:
            scores = np.array(scores)
            np.save(f, scores)
    f.close()
    with open(d.path+"output/final_bandScores.npy", "wb") as f:
            DScore = np.array(DScore)
            np.save(f, DScore)
    f.close()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    cores = [
                "black",
                "blue",
                "green",
                "red",
                "grey",
                "brown",
                "violet",
                "seagreen",
                "dimgray",
                "darkorange",
                "royalblue",
                "darkviolet",
                "maroon",
                "yellowgreen",
                "peru",
                "steelblue",
                "crimson",
                "silver",
                "magenta",
                "yellow",
            ]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection="3d")
    xarray = np.zeros((d.nkx, d.nky))
    yarray = np.zeros((d.nkx, d.nky))
    zarray = np.zeros((d.nkx, d.nky))
    count = -1
    for j in range(d.nky):
        for i in range(d.nkx):
            count = count + 1
            xarray[i, j] = d.kpoints[count, 0]
            yarray[i, j] = d.kpoints[count, 1]
    for banda in range(0, 3 + 1):
        zarray = BandsEnergy[banda]
        ax.plot_wireframe(xarray, yarray, zarray, color=cores[banda])

    plt.show()






