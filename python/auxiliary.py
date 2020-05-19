# This file has some usefull routines

# This is to include maths
import math
import re
import numpy as np

def indices3(i,j,l):
  index = str(i)+' '+str(j)+' '+str(l)
  return index

def indices2(i,j):
  index = str(i)+' '+str(j)
  return index

def firstminimum(lista):           # returns the first minimum and its index in the list
  firstm = min(lista)
  indexmin = lista.index(firstm)

  return firstm, indexmin

def minimum_nr(lista,orde=1):       # returns the minimum nr. ord and its index in the list
  lista2 = lista.copy()
  lista2.sort()
  minim = lista2[orde-1]
  indexmin = lista.index(minim)

  return minim,indexmin

def firstmaximum(lista):           # returns the first maximum and its index in the list
  firstm = max(lista)
  indexmax = lista.index(firstm)

  return firstm, indexmax

def maximum_nr(lista,orde=1):       # returns the maximum nr. ord and its index in the list
  lista2 = lista.copy()
  lista2.sort(reverse=True)
  maxim = lista2[orde-1]
  indexmax = lista.index(maxim)

  return maxim,indexmax


def misses(lista,numero):           # returns the number of occurences of numero and their place in the list
  miss = []
  for i in range(len(lista)):
    if lista[i] == numero:
      miss.append(i)

  return len(miss),miss         # returns (2, [0, 9])


def duplicates(lista):             # returns a list of pairs in which the second element is repeated
  dupl = []                        #   and the first is the position in the list
  for i in range(len(lista)):
    if lista.count(lista[i]) > 1:
      dupl.append([i,lista[i]])

  return dupl

def findmissing(lista):            # returns a list of missing integers in the lista (starting in 1)
  falta = []
  for i in range(1,len(lista)):
    if not i in lista:
      falta.append(i)

  return falta

def divideJobs(nt,n):             # returns a list of (lists of jobs for each process)
  # nt - total number of jobs
  # n - number of processes to divide the jobs for

  jobsperprocess = nt//n
  remain = nt - jobsperprocess*n
  if remain == 0:
    nn = jobsperprocess
  else:
     nn = jobsperprocess + 1
  jobs = []
  listofjobs = []

  for j in range(1,nt+1,nn):
    for i in range(nn):
      if j+i <= nt:
        jobs.append(j+i)

    listofjobs.append(jobs)
    jobs = []

  return listofjobs







