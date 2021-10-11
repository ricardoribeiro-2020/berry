import numpy as np


def evaluate_ReciprocalSpace(connections,neigs,TOL):
    nks, n_neigs, bnds, _ = connections.shape
    print('Start evaluationg the bands in reciprocal space with the following values:\n')
    print(f'\tNumber of K points = {nks}')
    print(f'\tNumber of neighbors points = {n_neigs}')
    print(f'\tNumber of bands = {bnds}')
    errors = np.zeros((nks,n_neigs,bnds),dtype=float)
    for k, connection in enumerate(connections):
        for i_neig, dp in enumerate(connection):
            neig = neigs[k,i_neig]
            if neig == -1:
                errors[k,i_neig,:] = 1
            else:
                for bn,row in enumerate(dp):
                    if bn == np.argmax(row):
                        errors[k,i_neig,bn] += row[bn]
                        if row[bn] > TOL:
                            errors[k,i_neig,bn] = 1
    
    return errors

def get_kstarts(errors):
    nks, n_neigs, bnds = errors.shape
    kstarts = [[],[],[],[],[]]                     #All bands are correct, >=10 bands are correct, >=8 bands are correct, >=5 bands are correct, >=3 bands are correct
    for k,values in enumerate(errors):
        score = np.zeros(bnds,dtype=float)
        for i_neig,bn_value in enumerate(values):
            for bn,value in enumerate(bn_value):
                score[bn] += value
        count = 0
        for bn,bn_score in enumerate(score):
            count += bn_score
            if count != (bn+1)*n_neigs:
                break
        if count == bnds*n_neigs:
            kstarts[0].append(k)
        if count >= 10*n_neigs:
            kstarts[1].append(k)
        if count >= 8*n_neigs:
            kstarts[2].append(k)
        if count >= 5*n_neigs:
            kstarts[3].append(k)
        if count >= 3*n_neigs:
            kstarts[4].append(k)
    return kstarts

def count_points(errors):
    nks, n_neigs, bnds = errors.shape
    bands = np.zeros(bnds,float)
    for k,values in enumerate(errors):
        score = np.zeros(bnds,dtype=float)
        for i_neig,bn_value in enumerate(values):
            for bn,value in enumerate(bn_value):
                score[bn] += value
        bands += score/4
    return bands
        

def correct_dotProduct(dp,bn,bm):
    correct = np.copy(dp)
    aux = np.copy(correct[:,bn])
    correct[:,bn] = correct[:,bm]
    correct[:,bm] = aux
    return correct

def correct_dPNeighs(connections,bn,bm):
    res = np.empty(connections.shape,float)
    for i,dp in enumerate(connections):
        res[i] = np.copy(dp)
        aux = np.copy(res[i][bn])
        res[i][bn] = np.copy(res[i][bm])
        res[i][bm] = aux
    return res

def correct_bands(band_neig,bn,bm):
    aux = band_neig[bn]
    band_neig[bn] = band_neig[bm]
    band_neig[bm] = aux
    return band_neig

def Diference_criteria(Ebn,E_neig,i_neig,indx_k,indx_kN): 
    Ni, Nj = Ebn.shape 
    ik, jk = indx_k
    ik_neig, jk_neig = indx_kN
    values = []
    for bn in range(len(E_neig)):
        d1 = np.abs(E_neig[bn] - Ebn[ik,jk])
        if i_neig == 0:
            if ik < Ni-1:
                d2 = np.abs((E_neig[bn] - Ebn[ik+1,jk])/2)
                val = np.mean([d1,d2]) 
            else:
                val = d1
        elif i_neig == 1:
            if jk < Nj-1:
                d2 = np.abs((E_neig[bn] - Ebn[ik,jk+1])/2)
                val = np.mean([d1,d2])   
            else:
                val = d1
        elif i_neig == 2:
            if ik > 0:
                d2 = np.abs((E_neig[bn] - Ebn[ik-1,jk])/2)
                val = np.mean([d1,d2])  
            else:
                val = d1
        elif i_neig == 3:
            if jk > 0:
                d2 = np.abs((E_neig[bn] - Ebn[ik,jk-1])/2)
                val = np.mean([d1,d2])   
            else:
                val = d1
        values.append(val)
    return np.argmin(values)

def Df_Dx_y(D,i,j,h):
  return (D[i+1,j]-D[i-1,j])/(2*h)

def Df_Dy_x(D,i,j,h):
  return (D[i,j+1]-D[i,j-1])/(2*h)

def Df_Dx_2Pontos(D,i,j,h):
  return (D[i+1,j]-D[i,j])/h

def Df_Dy_2Pontos(D,i,j,h):
  return (D[i,j+1]-D[i,j])/h

def calc_gradientlocal(A,i,j,h,Ni,Nj):
  if i == 0 or i == Ni-1:
    if i == 0:
      dx = Df_Dx_2Pontos(A,i,j,h)
    else:
      dx = Df_Dx_2Pontos(A,i-1,j,h)
  else:
    dx = Df_Dx_y(A,i,j,h)
  if j == 0 or j == Nj-1:
    if j == 0:
      dy = Df_Dy_2Pontos(A,i,j,h)
    else:
      dy = Df_Dy_2Pontos(A,i,j-1,h)
  else:
    dy = Df_Dy_x(A,i,j,h)
  return np.array([dx,dy])

def calc_Energies(A,ind1,ind2,quadrante):
  Ni = len(A)
  Nj = len(A[0])
  energies = []

  if quadrante == 1:
    a1 = 3
    a2 = 1
    b1 = 0
    b2 = -2
    aux1 = -1
    aux2 = 1
  if quadrante == 2:
    a1 = a2 = 3
    b1 = b2 = 0
    aux1 = aux2 = 1
  if quadrante == 3:
    a1 = 1
    a2 = 3
    b1 = -2
    b2 = 0
    aux1 = 1
    aux2 = -1
  if quadrante == 4:
    a1 = a2 = 1
    b1 = b2 = -2
    aux1 = aux2 = -1

  for i1 in range(ind1+b1,ind1+a1):
    for i2 in range(ind2+b2,ind2+a2):
      if (i1 < Ni and i2 < Nj) and ((i1 != ind1 and (i2 != ind2 or i2 != ind2+aux1)) or (i1 != ind1+aux2 and i2 != ind2)):
        new_energy = A[i1,i2]
        dx,dy = calc_gradientlocal(A, i1, i2, 1, Ni, Nj)
        auxDy = (i2-ind2)*dy
        auxDx = (i1-ind1)*dx
        new_energy -= auxDx+auxDy
        energies.append(new_energy)
  return energies

def Energy_criteria(Ebn,E_neig,i_neig,indx_k,indx_kN):
    Ni, Nj = Ebn.shape 
    ik, jk = indx_k
    ik_neig, jk_neig = indx_kN
    quandrantes = np.array([1,2]) + i_neig - np.array([0,4])*(i_neig == 3)
    energies = []
    for  quadrante in quandrantes:
        energies += list(calc_Energies(Ebn,ik_neig,jk_neig,quadrante))
    energies = np.array(energies)
    values = []
    for bn in range(len(E_neig)):
        DeltaE = np.mean(np.abs(energies - E_neig[bn]))
        values.append(DeltaE)
    return np.argmin(values)

def make_adjacentList(kpoints_index,bands_energies,bM):
    nks, _ = kpoints_index.shape
    X = np.empty((nks*(bM+1),3),float)
    for bn in range(0,bM+1):
      for k in range(nks):
        ik,jk = kpoints_index[k]
        X[bn*nks+k] = [ik*1e-4,jk*1e-4,bands_energies[bn,ik,jk]]
    return X
