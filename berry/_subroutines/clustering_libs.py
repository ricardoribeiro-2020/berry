"""This module contains the algorithm behind band classification.
It determines which points belong to each band used for posterior calculations.
The algorithm uses machine learning techniques to cluster the data.
"""

from __future__ import annotations
from multiprocessing import Process, Manager
from typing import Tuple, Union, Callable
from functools import partial

import logging

from scipy.ndimage import correlate
from scipy.optimize import curve_fit

import numpy as np
import networkx as nx

from berry import log
from .write_k_points import _bands_numbers


###########################################################################
# Type Definition
###########################################################################
Kpoint = int
Connection = float
Band = int
###########################################################################
# Constant Definition
###########################################################################
CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

N_NEIGS = 4

EVALUATE_RESULT_HELP = '''
    -------------- Report considering dot-product information -------------
            C -> Mean dot-product |<i|j>| of each k-point
            ---------------------------------------------------------
            Value |                              Description
            ------------------------------------------------------
            0     :                        The point is not solved
            1     :  MISTAKE               C <= 0.2
            2     :  DEGENERATE            It is a degenerate point.
            3     :  POTENTIAL_MISTAKE     C <= 0.8
            4     :  POTENTIAL_CORRECT     0.8 < C < 0.9
            5     :  CORRECT               C > 0.9
    -----------------------------------------------------------------------
'''

VALIDATE_RESULT_HELP = '''
    ------------- Report considering energy continuity criteria -----------
            N -> Number of directions that preserves energy continuity.
            ---------------------------------------------------------
            Value |                              Description
            ---------------------------------------------------------
            0     :                        The point is not solved
            1     :  MISTAKE               N = 0
            2     :  DEGENERATE            It is a degenerate point.
            3     :  OTHER                 0 < N < 4
            4     :  CORRECT               N = 4
    -----------------------------------------------------------------------
'''


def evaluate_result(values: Union[list[Connection], np.ndarray]) -> int:
    f'''
    This function attributes the correspondent signal using
    the dot product between each neighbor.

    Parameters
        values: array_like
            It is an array that contains the dot product
            between the k point and all neighbors.

    Returns
        signal: int
            C -> Mean connection of each k point
            Value :                              Description
            0     :                        The point is not solved
            1     :  MISTAKE               C <= 0.2
            2     :  DEGENERATE            It is a degenerate point.
            3     :  POTENTIAL_MISTAKE     C <= 0.8
            4     :  POTENTIAL_CORRECT     0.8 < C < 0.9
            5     :  CORRECT               C > 0.9
    '''

    TOL = 0.9       # Tolerance for CORRECT output
    TOL_DEG = 0.8   # Tolerance for POTENTIAL_CORRECT output
    TOL_MIN = 0.2   # Tolerance for POTENTIAL_MISTAKE output

    value = np.mean(values) # Mean conection of each k point

    if value > TOL:
        return CORRECT

    if value > TOL_DEG:
        return POTENTIAL_CORRECT

    if value > TOL_MIN and value < TOL_DEG:
        return POTENTIAL_MISTAKE

    return MISTAKE

def evaluate_point(k: Kpoint, bn: Band, k_index: np.ndarray, k_matrix: np.ndarray,
                   signal: np.ndarray, bands: np.ndarray, energies: np.ndarray) -> Tuple[int, list[int]]:
    '''
    Assign a signal value depending on energy continuity.

    Parameters
        k: Kpoint
            Integer that index the k point on analysis.
        bn: Band
            Integer that index the band number on analysis.
        k_index: array_like
            An array that contains the indices of each k point on the k-space matrix.
        k_matrix: array_like
            An array with the shape of the k-space. It contains the value of each k point in their corresponding position.
        signal: array_like
            An array with the current signal value for each k point.
        bands: array_like
            An array with the information of current solution of band clustering.
        energies: array_like
            It contais the energy value for each k point.
    
    Returns
        (signal, scores): Tuple[int, list[int]]
            scores: list[int]
                Sinalize if exist continuity on each direction [Down, Right, up, Left].
                    1 --- This direction preserves energy continuity.
                    0 --- This direction does not preserves energy continuity.
            N -> Number of directions with energy continuity.
            signal: int
                Value :                              Description
                0     :                        The point is not solved
                1     :  MISTAKE               N = 0
                2     :  DEGENERATE            It is a degenerate point.
                3     :  OTHER                 0 < N < 4
                4     :  CORRECT               N = 4
    '''
    
    CORRECT = 4
    MISTAKE = 1
    OTHER = 3

    TOL = 0.9         # Tolerance to consider that exist energy continuity
    N = 4             # Number of points to fit the curve

    mach_bn = bands[k, bn]          # original band
    sig = signal[k, bn]             # signal
    ik, jk = k_index[k]             # k point indices on k-space
    Ek = energies[k, mach_bn]       # k point's Energy value

    def difference_energy(Ek: float, Enew: float) -> float:
        '''
        Attributes a value that score how close is Ek to Enew.

        Parameters
            Ek: float
                K point's energy value.
            Enew: float
                Energy value to compare.
        Returns
            score: float [0, 1]
                Value that measures the closeness between Ek and Enew consider the other possible values.
        '''
        min_energy = np.min(np.abs(Enew-energies[k]))           # Computes all possible energy values for this k point
        delta_energy = np.abs(Enew-Ek)                          # Actual difference between Ek and Enew
        return min_energy/delta_energy if delta_energy else 1   # Score

    directions = np.array([[1,0], [0,1], [-1,0], [0,-1]])       # Down, Right, Up, Left
    energy_vals = []

    ###########################################################################
    # Calculate the score for each direction
    ###########################################################################

    for direction in directions:
        # Iterates each direction and obtain N points to be used for fit the curve
        n = np.repeat(np.arange(1,N+1),2).reshape(N,2)
        kn_index = n*direction + np.array([ik, jk])
        i, j = kn_index[:, 0], kn_index[:, 1]   # Selects the indices of these N points
        flag = len(np.unique(i)) > 1            # Necessary to identify which will be the direction of the fit
        if flag:
            i = i[i >= 0]
            i = i[i < k_matrix.shape[0]]
            j = np.full(len(i), j[0])
        else:
            j = j[j >= 0]
            j = j[j < k_matrix.shape[1]]
            i = np.full(len(j), i[0])
        
        ks = k_matrix[i, j] if len(i) > 0 else []   # Identify the N k points
        if len(ks) == 0:    
            # The direction in analysis does not have points
            energy_vals.append(1)
            continue
        if len(ks) <= 3:    
            # If there are not enough points to fit the curve it is used the Energy of the nearest neighbor
            Eneig = energies[ks[0], bands[ks[0], bn]]
            energy_vals.append(difference_energy(Ek, Eneig))
            continue
        
        k_bands = bands[ks, bn]
        Es = energies[ks, k_bands]
        X = i if flag else j
        new_x = ik if flag else jk
        pol = lambda x, a, b, c: a*x**2 + b*x + c           # Second order polynomial
        popt, pcov = curve_fit(pol, X, Es)                  # Curve fitting
        Enew = pol(new_x, *popt)                            # Obtain Energy value
        energy_vals.append(difference_energy(Ek, Enew))     # Calculate score
    
    energy_vals = np.array(energy_vals)
    scores = (energy_vals > TOL)*1  # Verification energy continuity on each direction
    score = np.sum(scores)          # Counting how many directions preserves energy continuity
    
    if score == N_NEIGS:
        return CORRECT, scores
    if score == 0:
        return MISTAKE, scores
    return OTHER, scores


class MATERIAL:
    '''
    This object contains all information about the material that
    will be used to solve their bands' problem.

    Atributes
        nkx : int
            The number ok k points on x direction.
        nky : int
            The number ok k points on y direction.
        nbnb : int
            Total number of bands.
        total_bands : int
            Total number of bands.
        nks : int
            Total number of k points.
        eigenvalues : array_like
            It contains the energy value for each k point.
        connections : array_like
            The dot product information between k points.
        neighbors : array_like
            An array with the information about which are the neighbors of each k point.
        vectors : array_like
            Each k point in the vector representation on k-space.
        n_process : int
            Number of processes to use.
        bands_final : array_like
            An array with final result of bands attribution.
        signal_final : array_like
            Contains the resulting signal for each k point.
        final_score : aray_like
            It contains the result score for each band.

    Methods
        solve() : None
            This method is the main algorithm which iterates between solutions
                trying to find the best result for the material.
        make_vectors() : None
            It transforms the information into more convenient data structures.
        make_BandsEnergy() : array_like
            It sets the energy information in more convinient data structure
        make_kpointsIndex() : None
            It computes the indices of each k point in their correspondence in k-space.
        make_connections() : None
            This function evaluates the connection between each k point, and adds an edge
                to the graph if its connection is greater than a tolerance value (tol).
        get_neigs() : list[Kpoint]
            Obtain the i's neighbors.
        find_path() : bool
            Verify if exist a path between two k points inside the graph.
        parallelize() : array_like
            Create processes for some function f over an iterator.
        get_components() : None
            Tt detects components well constructed.
        obtain_output() : None
            This function prepares the final data structures
                that are essential to other programs.
        print_report() : None
            Shows on screen the report for each band.
        correct_signal() : None
            This function evaluates the k-point signal calculated on previous analysis and attributes
                a new signal value depending only on energy continuity.
    '''     
    def __init__(self, nkx: int, nky: int, nbnd: int, nks: int, eigenvalues: np.ndarray,
                 connections: np.ndarray, neighbors: np.ndarray, logger: log, n_process: int=1) -> None:
        '''
        Initialize the object.

        Parameters
            nkx : int
                The number ok k points on x direction.
            nky : int
                The number ok k points on y direction.
            nbnb : int
                Total number of bands.
            total_bands : int
                Total number of bands.
            nks : int
                Total number of k points.
            eigenvalues : array_like
                It contains the energy value for each k point.
            connections : array_like
                The dot product information between k points.
            neighbors : array_like
                An array with the information about which are the neighbors of each k point.
            vectors : array_like
                Each k point in the vector representation on k-space.
            n_process : int
                Number of processes to use.
        '''
        self.nkx = nkx
        self.nky = nky
        self.nbnd = nbnd
        self.total_bands = nbnd
        self.nks = nks
        self.eigenvalues = eigenvalues
        self.connections = connections
        self.neighbors = neighbors
        self.vectors = None
        self.n_process = n_process
        self.logger = logger

    def make_BandsEnergy(self) -> np.ndarray:
        '''
        It sets the energy information in more convinient data structure
        
        Parameters
            None
        
        Returns
            BandsEnergy : array_like
                An array with the information about each energy value on k-space.
        '''
        bands_final, _ = np.meshgrid(np.arange(0, self.nbnd),
                                     np.arange(0, self.nks))
        BandsEnergy = np.empty((self.nbnd, self.nkx, self.nky), float)
        for bn in range(self.nbnd):
            count = -1
            zarray = np.empty((self.nkx, self.nky), float)
            for j in range(self.nky):
                for i in range(self.nkx):
                    count += 1
                    zarray[i, j] = self.eigenvalues[count,
                                                    bands_final[count, bn]]
            BandsEnergy[bn] = zarray
        return BandsEnergy

    def make_kpointsIndex(self) -> None:
        '''
        It computes the indices of each k point in their correspondence in k-space.
        '''
        My, Mx = np.meshgrid(np.arange(self.nky), np.arange(self.nkx))
        self.matrix = My*self.nkx+Mx
        counts = np.arange(self.nks)
        self.kpoints_index = np.stack([counts % self.nkx, counts//self.nkx],
                                      axis=1)

    def make_vectors(self, min_band: int=0, max_band: int=-1) -> None:
        '''
        It transforms the information into more convenient data structures.

        Parameters
            min_band : int
                An integer that gives the minimum band that clustering will use.
                    default: 0
            max_band : int
                An integer that gives the maximum band that clustering will use.
                    default: All

        Result
            self.vectors: [kx_b, ky_b, E_b]
                k = (kx, ky)_b: k point
                b: band number
            self.degenerados: It marks the degenerate points
            self.GRPAH: It is a graph in which each node represents a vector.
            self.energies: It contains the energy values for each band distributed
                        in a matrix.
        '''
        process_name = 'Making Vectors'
        self.logger.percent_complete(0, 100, title=process_name)

        ###########################################################################
        # Compute the auxiliar information
        ###########################################################################
        self.GRAPH = nx.Graph()     # Create the initail Graph
        self.min_band = min_band
        self.max_band = max_band
        nbnd = self.nbnd if max_band == -1 else max_band+1
        self.make_kpointsIndex()
        energies = self.make_BandsEnergy()
        self.logger.percent_complete(20, 100, title=process_name)

        ###########################################################################
        # Compute the vector representation of each k point
        ###########################################################################
        n_vectors = (nbnd-min_band)*self.nks
        ik = np.tile(self.kpoints_index[:, 0], nbnd-min_band)
        jk = np.tile(self.kpoints_index[:, 1], nbnd-min_band)
        bands = np.arange(min_band, nbnd)
        eigenvalues = self.eigenvalues[:, bands].T.reshape(n_vectors)
        self.vectors = np.stack([ik, jk, eigenvalues], axis=1)
        self.logger.percent_complete(100, 100, title=process_name)

        self.GRAPH.add_nodes_from(np.arange(n_vectors))     # Add the nodes, each node represent a k point
        
        ###########################################################################
        # Verify if any k point is a degenerate point
        ###########################################################################
        self.degenerados = []
        def obtain_degenerates(vectors: np.ndarray) -> list[Kpoint]:
            '''
            Find all degenerate k points present on vectors.

            Parameters
                vectors : array_like
                    An array with vector representation of k points.
            
            Returns
                degenerates : list[Kpoint]
                    It contains the degenerate points found.
            '''
            degenerates = []
            for i, v in vectors:
                degenerado = np.where(np.all(np.isclose(self.vectors[i+1:]-v, 0),
                                    axis=1))[0] # Verify which points have numerically the same value
                if len(degenerado) > 0:
                    self.logger.debug(f'Found degenerete point for {i}')
                    degenerates += [[i, d+i+1] for d in degenerado]
            return degenerates

        # Parallelize the verification process
        self.degenerados = self.parallelize('Finding degenerate points', obtain_degenerates, enumerate(self.vectors))

        if len(self.degenerados) > 0:
            self.logger.debug('\tDegenerate Points: ')
            for d in self.degenerados:
                self.logger.debug(f'\t\t{d}')

        self.ENERGIES = energies
        self.nbnd = nbnd-min_band
        self.bands_final = np.full((self.nks, self.total_bands), -1, dtype=int)

    def get_neigs(self, i: Kpoint) -> list[Kpoint]:
        '''
        Obtain the i's neighbors

        Parameters
            i : Kpoint
                The node index.
        
        Returns
            neighbors : list[Kpoint]
                List with the nodes that are neighbors of the node i.
        '''
        return list(self.GRAPH.neighbors(i))

    def find_path(self, i: Kpoint, j:Kpoint) -> bool:
        '''
        Verify if exist a path between two k points inside the graph

        Parameters
            i : Kpoint
            j : Kpoint
        
        Returns : bool
            If exists a path return True
        '''
        neighs = self.get_neigs(i)
        neigh = neighs.pop(0) if len(neighs) > 0 else None
        visited = [i] + [d for points in self.degenerados
                         for d in points if d not in [i, j]]
        while (neigh is not None and neigh != j and
               (neigh not in visited or len(neighs) > 0)):
            if neigh in visited:
                neigh = neighs.pop(0)
                continue
            visited.append(neigh)
            for k in self.get_neigs(neigh):
                if k not in visited:
                    neighs.append(k)
            neigh = neighs.pop(0) if len(neighs) > 0 else None
        return neigh == j if neigh is not None else False

    def make_connections(self, tol:float=0.95) -> None:
        '''
        This function evaluates the connection between each k point,
        and adds an edge to the graph if its connection is greater
        than a tolerance value (tol).

        <i|j>: The dot product between i and j represents its connection

        Parameters
            tol : float
                It is the minimum connection value that will be accepted as an edge.
                default: 0.95
        '''
        ###########################################################################
        # Find the edges on the graph
        ###########################################################################
        def connection_component(vectors:np.ndarray) -> list[list[Kpoint]]:
            '''
            Find the possible edges in the graph using the information of dot product.

            Parameters
                vectors : array_like
                    An array with vector representation of k points.
            
            Returns
                edges : list[list[Kpoint]]
                    List of all edges that was found.
            '''
            edges = []
            bands = np.repeat(np.arange(self.min_band, self.max_band+1), len(self.neighbors[0]))
            for i_ in vectors:
                bn1 = i_//self.nks + self.min_band  # bi
                k1 = i_ % self.nks
                neighs = np.tile(self.neighbors[k1], self.nbnd)
                ks = neighs + bands*self.nks
                ks = ks[neighs != -1]
                for j_ in ks:
                    k2 = j_ % self.nks
                    bn2 = j_//self.nks + self.min_band  # bj
                    i_neig = np.where(self.neighbors[k1] == k2)[0]
                    connection = self.connections[k1, i_neig,
                                                    bn1, bn2]  # <i|j>
                    '''
                    for each first neighbor
                    Edge(i,j) = 1 iff <i, j> ~ 1
                    '''
                    if connection > tol:
                        edges.append([i_, j_])
            return edges

        # Parallelize the edges calculation
        edges = self.parallelize('Computing Edges', connection_component, range(len(self.vectors)))
        # Establish the edges on the graph from edges array
        self.GRAPH.add_edges_from(edges)

        ###########################################################################
        # Solve problems that a degenerate point may cause
        ###########################################################################
        degnerates = []
        problems = []
        for d1, d2 in self.degenerados:
            '''
            The degenerate points may cause problems.
            The algorithm below finds its problems and solves them.
            '''
            if not self.find_path(d1, d2):
                # Verify if exist a path that connects two forbidden points
                # The points does not cause problems but are degenerated, then, they are signaled
                # The basis rotation program will solve them.
                degnerates.append([d1, d2])
                continue
            # Obtains the neighbors from each degenerate point that cause problems
            N1 = np.array(self.get_neigs(d1))
            N2 = np.array(self.get_neigs(d2))
            if len(N1) == 0 or len(N2) == 0:
                continue
            problem = {
                d1 : N1,
                d2 : N2,
            }
            self.logger.debug(f'\tProblem:\n\t{d1}: {N1}\n\t{d2}: {N2}\n')
            NKS = self.nks
            if len(N1) > 1 and len(N2) > 1:
                def n2_index(n1): return np.where(N2 % NKS == n1 % NKS)
                N = [[n1, N2[n2_index(n1)[0][0]]] for n1 in N1]
                flag = False
            else:
                if len(N1) == len(N2):
                    N = list(zip(N1, N2))
                else:
                    Ns = [N1, N2]
                    N_1 = Ns[np.argmin([len(N1), len(N2)])]
                    N_2 = Ns[np.argmax([len(N1), len(N2)])]
                    n2_index = np.where(N_2 % NKS == N_1[0] % NKS)[0][0]
                    N = [[N_1[0], N_2[n2_index]]] \
                        + [[n] for n in N_2 if n != N_2[n2_index]]
                    flag = True
            # Assign to a specific band each point and establish the corresponding edges
            n1 = np.random.choice(N[0])
            if flag:
                N1_ = [n1]
                N2_ = [N[0][np.argmax(np.abs(N[0]-n1))]]
                n2 = N2_[0]
                Ns = [N1_, N2_]
                for n in N[1:]:
                    n = n[0]
                    Ns[np.argmin(np.abs(np.array([n1, n2]) - n))].append(n)
            else:
                N1_ = [n[np.argmin(np.abs(n-n1))] for n in N]
                N2_ = [n[np.argmax(np.abs(n-n1))] for n in N]
            solution = {
                d1 : N1_,
                d2 : N2_
            }
            problems.append({
                'points' : [d1, d2],
                'problem' : problem,
                'solution' : solution
            })
            self.logger.debug(f'\tSolution:\n\t{d1}: {N1_}\n\t{d2}: {N2_}\n')
            for k in N1:
                self.GRAPH.remove_edge(k, d1)
            for k in N2:
                self.GRAPH.remove_edge(k, d2)

            for k in N1_:
                self.GRAPH.add_edge(k, d1)
            for k in N2_:
                self.GRAPH.add_edge(k, d2)
        
        self.degenerates = np.array(degnerates)

        ###########################################################################
        # Show the degenerate points that causes problems
        ###########################################################################
        self.solved_problems_info : list[str, list] = ['', []]
        self.solved_problems_info[0] = '\n\tThe number of points with forbidden paths between them is: ' + str(len(problems))
        if len(problems) > 0:
            self.solved_problems_info[0] += '\n\t    Problems in points:'
            self.logger.info('\t*** Points with forbidden paths found and solved ***')
            self.logger.info('\t    The problems and their solutions are:')
        
        calc_k_bn = lambda p: (p % self.nks, p // self.nks + self.min_band )
        for problem_dic in problems:
            d1, d2 = problem_dic['points']
            problem = problem_dic['problem']
            solution = problem_dic['solution']
            self.logger.info(f'\t    * Problem:')
            k1, bn1 = calc_k_bn(d1)
            _, bn2 = calc_k_bn(d2)
            self.solved_problems_info[1].append([d1, d2])
            self.logger.info(f'\n\t\tK-point: {k1} in bands: {bn1}, {bn2}')
            self.logger.info(f'\t\t   k: {k1}, band: {bn1} has edges with:')
            for k, bn in map(calc_k_bn, problem[d1]):
                self.logger.info(f'\t\t    k: {k} bn: {bn}')
            self.logger.info(f'\t\t   k: {k1}, band: {bn2} has edges with:')
            for k, bn in map(calc_k_bn, problem[d2]):
                self.logger.info(f'\t\t    k: {k} bn: {bn}')
            self.logger.info(f'\n\t      Solution:')
            self.logger.info(f'\t\t   k: {k1}, band: {bn1} has edges with:')
            for k, bn in map(calc_k_bn, solution[d1]):
                self.logger.info(f'\t\t    k: {k} bn: {bn}')
            self.logger.info(f'\t\t   k: {k1}, band: {bn2} has edges with:')
            for k, bn in map(calc_k_bn, solution[d2]):
                self.logger.info(f'\t\t    k: {k} bn: {bn}')

        if len(problems) > 0:
            self.logger.info('\n\t    Note that this solution may be wrong \n\t    but next iterations will correct it.\n')

            

    def parallelize(self, process_name: str, f: Callable, iterator: Union[list, np.ndarray], *args) -> np.ndarray:
        '''
        Create processes for some function f over an iterator.

        Parameters
            process_name : string
                The name of the process to be parallelized.
            f : Callable
                Function f(iterator: array_like, *args) to be applied.
                This function gets another iterator as a parameter and will compute the result for each element in the iterator.
            iterator : array_like
                The function f is applied to elements in the iterator array.
            verbose : bool
                If this flag is true the progress bar is shown.
        
        Return
            result : array_like
                It contains the result for each value inside the iterator.
                The information is not sorted.
        '''
        process = []
        iterator = list(iterator)
        N = len(iterator)

        ###########################################################################
        # Debug information and Progress bar
        ###########################################################################
        
        self.logger.debug(f'Starting Parallelization for {process_name} with {N} values')
        
        self.logger.percent_complete(0, N, title=process_name)

        ###########################################################################
        # Processes management
        ###########################################################################
        def parallel_f(result: np.ndarray, per: list[int], iterator: Union[list, np.ndarray], *args) -> None:
            '''
            Auxiliar function to help the parallelization

            Parameters:
                result : array_like
                    It is a shared memory list where each result is stored.
                per : list[int]
                    It is a shared memory list that contais the number of elements solved.
                iterator : array_like
                    The function f is applied to elements in the iterator array.
            '''
            value = f(iterator, *args)              # The function f is applied to the iterator
            if value is not None:
                # The function may not return anything
                result += f(iterator, *args)        # Store the output into result array
            per[0] += len(iterator)                 # The counter is actualized
            
            # Show on screen the progress bar
            self.logger.percent_complete(per[0], N, title=process_name)
        
        result = Manager().list([])             # Shared Memory list to store the result
        per = Manager().list([0])               # Shared Memory to countability the progress
        f_ = partial(parallel_f,  result, per)  # Modified function used to create processes

        n = N//self.n_process                                                   # Number or processes
        for i_start in range(self.n_process):
            # Division of the iterator array into n smaller arrays
            j_end = n*(i_start+1) if i_start < self.n_process-1\
                else n*(i_start+1) + N % self.n_process
            i_start = i_start*n
            p = Process(target=f_, args=(iterator[i_start: j_end], *args))      # Process creation
            p.start()                                                           # Initialize the process
            process.append(p)

        while len(process) > 0:
            p = process.pop(0)
            p.join()

        self.logger.info()
        return np.array(result)

    def get_components(self, alpha: float=0.5) -> None:
        '''
        The make_connections function constructs the graph, in which
        it can detect components well constructed.
            - A component is denominated solved when it has all
              k points attributed.
            - A cluster is a significant component that can not join
              with any other cluster.
            - Otherwise, It is a sample that has to be grouped with
              some cluster.
        
        Parameters
            alpha : float
                The weight of connection to consider for score calculation.
                    score = alpha*<i|j> + (1-alpha)*f(E_i)
        '''

        ###########################################################################
        # Identify connected components inside the GRAPH
        ###########################################################################
        self.logger.info('\n\n\t\tNumber of Components: '.rstrip('\n'))
        self.logger.info(f'\t\t{nx.number_connected_components(self.GRAPH)}')
        self.components = [COMPONENT(self.GRAPH.subgraph(c),
                                     self.kpoints_index,
                                     self.matrix)
                           for c in nx.connected_components(self.GRAPH)]    # Identify the components
        index_sorted = np.argsort([component.N
                                   for component in self.components])[::-1] # Sort the components by the number of nodes in decreasing order

        ###########################################################################
        # Identify the clusters and samples
        ###########################################################################
        self.solved : list[COMPONENT] = []
        clusters : list[COMPONENT] = []
        samples : list[COMPONENT] = []
        for i in index_sorted:
            # The first biggest components that can not join to the others are identified as clusters
            component = self.components[i]
            if component.N == self.nks:
                #  If the number of nodes inside the component equals the total number of k points, the cluster is considered solved
                self.solved.append(component)
                continue
            component.calculate_pointsMatrix()  # Computes the projection into k-space
            component.calc_boundary()           # Undersample the points by representative points identification
            if len(clusters) == 0:
                # The biggest component if it is not complete then it is the first cluster
                clusters.append(component)
                continue
            if not np.any([cluster.validate(component)
                           for cluster in clusters]):
                # Verification if the component can join other clusters.
                # If it can not, then it is a cluster.
                clusters.append(component)
            else:
                # If it can, then it is a sample.
                samples.append(component)
        self.logger.info(f'\t\tPhase 1: {len(self.solved)}/{self.nbnd} Solved')
        self.logger.info(f'\t\tInitial clusters: {len(clusters)} Samples: {len(samples)}')

        ###########################################################################
        # Assigning samples to clusters by selecting the best option
        ###########################################################################
        count = np.array([0, len(samples)])
        while len(samples) > 0:
            evaluate_samples = np.zeros((len(samples), 2))                          # Samples' scores storage
            for i_s, sample in enumerate(samples):
                # Comparison of each sample to each cluster
                scores = np.zeros(len(clusters))                                    # Storage the score of each cluster with the sample
                for j_s, cluster in enumerate(clusters):
                    if not cluster.validate(sample):
                        # If this sample can not join the cluster, the score is 0
                        continue
                    if len(sample.k_edges) == 0:
                        # Compute the edges
                        sample.calculate_pointsMatrix()
                        sample.calc_boundary()
                    scores[j_s] = sample.get_cluster_score(cluster,
                                                           self.min_band,
                                                           self.max_band,
                                                           self.neighbors,
                                                           self.ENERGIES,
                                                           self.connections,
                                                           alpha=alpha)             # Calculate the score
                evaluate_samples[i_s] = np.array([np.max(scores),
                                                np.argmax(scores)])                 # Store the best cluster's score

            for cluster in clusters:
                # Flag used to identify if the score must be calculated again
                cluster.was_modified = False
            arg_max = np.argmax(evaluate_samples[:, 0])                             # Obtain the best sample
            sample = samples.pop(arg_max)                                           # Select the sample
            score, bn = evaluate_samples[arg_max]                                   # Get the values
            bn = int(bn)
            count[0] += 1                                                           # Update the counter
            clusters[bn].join(sample)                                               # Join the sample to the best cluster               
            clusters[bn].was_modified = True
            self.logger.percent_complete(count[0], count[1], title='Clustering Samples')
            self.logger.debug(f'\t\t{count[0]}/{count[1]} Sample corrected: {score}')
            if clusters[bn].N == self.nks:
                #  If the number of nodes inside the component equals the total number of k points, the cluster is considered solved
                self.logger.debug('\n\tCluster Solved')
                self.solved.append(clusters.pop(bn))

        self.logger.info(f'\t\tPhase 2: {len(self.solved)}/{self.nbnd} Solved')

        if len(self.solved)/self.nbnd < 1:
            self.logger.info(f'\t\tNew clusters: {len(clusters)}')

        self.clusters : list[COMPONENT] = clusters

    def obtain_output(self) -> None:
        '''
        This function prepares the final data structures
        that are essential to other programs.
        '''

        ###########################################################################
        # Obtain the resultant bands' attribution and the k-point's signal.
        ###########################################################################
        solved_bands = []
        for solved in self.solved:
            bands = solved.get_bands()                                              # Getting the k-points' raw bands inside the solved cluster
            bn = solved.bands[0] + self.min_band                                    # Select the most repeated band and apply the initial band correction
            solved.bands = solved.bands[1:]                                         # Update the bands array
            while bn in solved_bands:
                # If the band to be solved is already solved, the next most repeated band is selected
                bn = solved.bands[0] + self.min_band                                # initial band correction
                solved.bands = solved.bands[1:]                                     # Update the bands array
            solved_bands.append(bn)                                                 # Append the solved band
            self.bands_final[solved.k_points, bn] = bands + self.min_band           # Update the resultant bands' attribution array

            for k in solved.k_points:
                # For each k-point is calculate the solution score
                bn1 = solved.bands_number[k] + self.min_band                        # The k-point's band
                connections = []                                                    # The array that store the dot-product with the k-point's neighbors
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    # Obtain the dot-product with each neighbor
                    if k_neig == -1:
                        continue
                    bn2 = solved.bands_number[k_neig] + self.min_band               # The neighbor's band
                    connections.append(self.connections[k, i_neig, bn1, bn2])       # <k, k neighbor>

                self.signal_final[k, bn] = evaluate_result(connections)             # Computes the k-point's signal

        clusters_sort = np.argsort([c.N for c in self.clusters])                    # Sort the remaining clusters
        for i_arg in clusters_sort[::-1]:
            cluster = self.clusters[i_arg]
            bands = cluster.get_bands()                                             # Getting the k-points' raw bands inside the cluster
            bn = cluster.bands[0] + self.min_band                                   # Select the most repeated band and apply the initial band correction
            cluster.bands = cluster.bands[1:]                                       # Update the bands array
            while bn in solved_bands and len(cluster.bands) > 0:
                # If the band to be solved is already solved, the next most repeated band is selected
                bn = cluster.bands[0] + self.min_band
                cluster.bands = cluster.bands[1:]

            if bn in solved_bands and len(cluster.bands) == 0:
                # If the cluster does not belong to any band is ignored
                break

            solved_bands.append(bn)                                                 # Append the solved band
            self.bands_final[cluster.k_points, bn] = bands + self.min_band          # Update the resultant bands' attribution array
            for k in cluster.k_points:
                # For each k-point is calculate the solution score
                bn1 = cluster.bands_number[k] + self.min_band                       # The k-point's band
                connections = []                                                    # The array that store the dot-product with the k-point's neighbors
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    # Obtain the dot-product with each neighbor
                    if k_neig == -1:
                        continue
                    if k_neig not in cluster.k_points:
                        # If the neighbor does not exist inside the cluster, the dot-product is 0
                        connections.append(0)
                        continue
                    bn2 = cluster.bands_number[k_neig] + self.min_band              # The neighbor's band
                    connections.append(self.connections[k, i_neig, bn1, bn2])       # <k, k neighbor>

                self.signal_final[k, bn] = evaluate_result(connections)             # Computes the k-point's signal


        ###########################################################################
        # Scoring the result.
        # Signaling and storage of degenerate k-points.
        ###########################################################################

        for d1, d2 in self.degenerates:
            # Signaling the numerically degenerate points Ei ~ Ej
            k1 = d1 % self.nks                                              # k point
            bn1 = d1 // self.nks + self.min_band                            # band
            k2 = d2 % self.nks                                              # k point
            bn2 = d2 // self.nks + self.min_band                            # band
            Bk1 = self.bands_final[k1] == bn1                               # Find in which  band the k-point was attributed
            Bk2 = self.bands_final[k2] == bn2                               # Find in which  band the k-point was attributed
            bn1 = np.argmax(Bk1) if np.sum(Bk1) != 0 else bn1               # Final band
            bn2 = np.argmax(Bk2) if np.sum(Bk2) != 0 else bn2               # Final band

            self.signal_final[k1, bn1] = DEGENERATE                         # Signal k_point as Degenerate
            self.signal_final[k2, bn2] = DEGENERATE                         # Signal k_point as Degenerate

        k_basis_rotation : list[Tuple[Kpoint, Kpoint, Band, list[Band]]] = []           # Storage pairs of points that are degenerates by dot product 0.5 < <i|j> < 0.8
        for bn in range(self.total_bands):
            # Search these degenerate points on each band
            # Calculating the score of the result
            score = 0
            for k in range(self.nks):
                # Evaluate each k-point
                if self.signal_final[k, bn] == NOT_SOLVED:
                    # If this k-point had not been solved the analysis can not be done
                    continue
                kneigs = self.neighbors[k]                                                  # k-point's neighbors
                flag_neig = kneigs != -1                                                    # Flag to obtain the allowed neighbors 
                i_neigs = np.arange(N_NEIGS)[flag_neig]                                     # Neighbors' index
                kneigs = kneigs[flag_neig]                                                  # Allowed neighbors
                flag_neig = self.signal_final[kneigs, bn] != NOT_SOLVED                     # Flag to obtain only attributed neighbors
                i_neigs = i_neigs[flag_neig]                                                # Update Neighbors' index
                kneigs = kneigs[flag_neig]                                                  # Update neighbors
                if len(kneigs) == 0:
                    # If there are no neighbors the k-point's score is 0
                    continue
                bn_k = self.bands_final[k, bn]                                              # K-point's band
                bn_neighs = self.bands_final[kneigs, bn]                                    # Neighbors' bands
                k = np.repeat(k, len(kneigs))                                               # Array with the same k-point
                bn_k = np.repeat(bn_k, len(kneigs))                                         # Array with the same K-point's band
                dps = self.connections[k, i_neigs, bn_k, bn_neighs]                         # The array with the dot-product between the k-point and their neighbors
                if np.any(np.logical_and(dps >= 0.5, dps <= 0.8)):
                    # It is considered degenerate if the k-point has some 
                    # neighbor's dot-product between 0.5 and 0.8
                    dps_deg = self.connections[k, i_neigs, bn_k]                                # All k-point dot-products
                    k = k[0]                                                                    # K-point
                    i_deg, bn_deg = np.where(np.logical_and(dps_deg >= 0.5, dps_deg <= 0.8))    # Find where the k-point dot-product is considered degenerate
                    k_deg = self.neighbors[k][i_deg+np.min(i_neigs)]                            # Identify the degenerate neighbors
                    i_sort = np.argsort(k_deg)                                                  # Sort the degenerate neighbors
                    k_deg = k_deg[i_sort]                                                       # Sort the degenerate neighbors
                    bn_deg = bn_deg[i_sort]                                                     # Sort the degenerate bands    
                    k_unique, index_unique = np.unique(k_deg, return_index=True)                # Identify unique neighbors
                    bn_unique = np.split(bn_deg, index_unique[1:])                              # Classify the bands by unique neighbor
                    len_bn = np.array([len(k_len) for k_len in bn_unique])                      # Look how many bands have each neighbor
                    if np.any(len_bn > 1):
                        # If for some neighbor exist more than one unique band,
                        # the k-point is degenerate
                        i_deg = np.where(len_bn > 1)[0]                                             # Obtain the neighbors
                        k_deg = k_unique[i_deg]                                                     # Obtain the neighbors
                        bns_deg = [bn_unique[j_deg] for j_deg in i_deg]                             # Get the unique bands
                        k_basis_rotation.append([k, k_deg, bn, bns_deg])                            # Append the information of the degenerate k-point
                score += np.mean(dps)                                                       # Update the band score
            score /= self.nks                                                               # Compute the mean socore
            self.final_score[bn] = score                                                    # Storage the band score

        degenerates = []
        for i, (k, k_deg, bn, bns_deg) in enumerate(k_basis_rotation[:-1]):
            # For each possible degenerate point have to exist a pair
            for k_, k_deg_, bn_, bns_deg_ in k_basis_rotation[i+1:]:
                # Comparison between each possible degenerate point.
                if k != k_ or not np.all(k_deg == k_deg_):
                    # The k_ point is not the k's pair
                    continue
                if not np.all([np.all(np.isin(bns, bns_deg_[j])) for j, bns in enumerate(bns_deg)]):
                    # If they do not belong to the same bands, The k_ point is not the k's pair
                    continue
                degenerates.append([k, bn, bn_])

        self.degenerate_final = []                                                 # Final degenerates k-points
        analyzed = []                                                              # K-points analyzed
        for i, (k, bn, bn_) in enumerate(degenerates):
            # For each possible degenerate point stored inside k_basis_rotation.
            # There are only a few that are true degenerates; the other ones are their neighbors
            if i in analyzed:
                # The degenerates[i] point was already analyzed
                continue
            analyzed.append(i)
            # It is necessary to search group of points that are degenerate
            # These points are stored in same_group
            same_group = [[k, bn, bn_]]
            for j, (k_, bn0, bn1) in enumerate(degenerates[i+1:]):
                # Comparison between each possible pair of degenerate point.
                ik, jk = self.kpoints_index[k]                                              # Obtain the matrix indices of k-space projection (k-point)
                ik_, jk_ = self.kpoints_index[k_]                                           # Obtain the matrix indices of k-space projection (k_-point)
                idif = np.abs(ik - ik_)                                                     # Manhattan distance i axes
                jdif = np.abs(jk - jk_)                                                     # Manhattan distance j axes
                if idif > 1 or jdif > 1 or not np.all(np.isin([bn, bn_], [bn0, bn1])):
                    # If the total Manhattan distance is more than 2 or the points do not belong to the same band,
                    # the analysis can not be done
                    continue
                analyzed.append(j+i+1)                                                      # The point k_ was analyzed
                same_group.append([k_, bn0, bn1])                                           # The k_-point belongs to the same group of k

            same_group = np.array(same_group)
            ks = same_group[:, 0]                                                           # K-points
            neighs = self.neighbors[ks]                                                     # K-points' neighbors
            points = [np.sum(neighs == k) for k in ks]                                      # How many points the k-point is neighbor?
            self.degenerate_final.append(same_group[np.argmax(points)])                     # There is only one degenerate point
        
        self.degenerate_final = np.array(self.degenerate_final)

    def print_report(self, signal_report: np.ndarray, description:str, show:bool=True) -> None:
        '''
        Shows on screen the report for each band.

        Parameters
            signal_report : array_like
                An array with the k-point's signal information.
            description : string
                Describes the table
            show : bool
                If it is true then the table is shown. Otherwise, the string and the report are returned.
        Return
            final_report : string
        '''
        final_report = f'\n\t====== {description} ======\n'
        bands_report = []
        MAX = np.max(signal_report) + 1
        ###########################################################################
        # Prepare the summary for each band
        ###########################################################################
        for bn in range(self.min_band, self.min_band+self.nbnd):
            band_result = signal_report[:, bn]                              # Obtain all k-point' signals for band bn
            report = [np.sum(band_result == s) for s in range(MAX)]         # Set the band report
            report.append(np.round(self.final_score[bn], 4))                # Set the final score
            bands_report.append(report)

            self.logger.debug(f'\t\t\tNew Band: {bn}\tnr fails: {report[0]}')
            if self.logger.level == logging.DEBUG:
                self.logger.debug(_bands_numbers(self.nkx, self.nky, self.bands_final[:, bn]))

        ###########################################################################
        # Set up the data representation
        ###########################################################################
        bands_report = np.array(bands_report)
        final_report += '\n\t\t Signaling: how many events ' + \
                        'in each band signaled.\n'
        bands_header = '\n\t\t Band | '

        header = list(range(MAX)) + ['Score']
        for signal, value in enumerate(header):
            # Make the header
            #  Band |    0    1   2     3     4      5 ...
            n_spaces = len(str(np.max(bands_report[:, signal])))-1
            bands_header += ' '*n_spaces+str(value) + '   '

        final_report += bands_header + '\n\t\t'
        final_report += '-'*len(bands_header)

        for bn, report in enumerate(bands_report):
            # Make the report
            #  bn    |    0    0   0     0     0   nks
            bn += self.min_band
            final_report += f'\n\t\t {bn}{" "*(4-len(str(bn)))} |' + ' '
            for signal, value in enumerate(report):
                if signal < MAX:
                    value = int(value)
                n_max = len(str(np.max(bands_report[:, signal])))
                n_spaces = n_max - len(str(value))
                final_report += ' '*n_spaces+str(value) + '   '
        final_report += '\n'
        if show:
            self.logger.info(final_report)              # Show on screen
            return None
        return final_report, bands_report
    
    def correct_signal(self) -> None:
        '''
        This function evaluates the k-point signal calculated on previous analysis and attributes
        a new signal value depending only on energy continuity.
        '''
        del self.GRAPH              # Clean memory
        OTHER = 3
        MISTAKE = 1

        ###########################################################################
        # Set up the necessary data structures
        ###########################################################################
        self.correct_signalfinal = np.copy(self.signal_final)                           # New array to store the corrected signal
        self.correct_signalfinal[self.signal_final == CORRECT] = CORRECT-1              # Change the CORRECT signal to CORRECT - 1

        ks_pC, bnds_pC = np.where(self.signal_final == POTENTIAL_CORRECT)               # Select the points marked as POTENTIAL_CORRECT
        ks_pM, bnds_pM = np.where(self.signal_final == POTENTIAL_MISTAKE)               # Select the points marked as POTENTIAL_MISTAKE

        ks = np.concatenate((ks_pC, ks_pM))                                             # Join all k-points
        bnds = np.concatenate((bnds_pC, bnds_pM))                                       # Join the k-points' bands

        error_directions = []                                                           # This array stores the k-point where the energy ccontinuity fails.     ! It is not used !
        directions = []                                                                 # This array stores the direction where the energy continuity fails.    ! It is not used !

        ###########################################################################
        # Correct the k-point's signal
        ###########################################################################
        for k, bn in zip(ks, bnds):
            # Iterate over all k-point signaled as potential points where the energy continuity may fail
            signal, scores = evaluate_point(k, bn, self.kpoints_index,
                                            self.matrix, self.signal_final, 
                                            self.bands_final, self.eigenvalues)         # Obtain the new signal
            self.correct_signalfinal[k, bn] = signal                                    # Store this new signal
            if signal == OTHER:
                # If the point was not marked as a correct or mistake signal, It is stored
                error_directions.append([k, bn])
                directions.append(scores)
            self.logger.debug(f'K point: {k} Band: {bn}    New Signal: {signal} Directions: {scores}')

        ###########################################################################
        # Create a new problem for another solver iteration
        ###########################################################################
        k_error, bn_error = np.where(self.correct_signalfinal == MISTAKE)           # Identify Mistakes
        k_other, bn_other = np.where(self.correct_signalfinal == OTHER)             # Identify k-points with some discontinuity
        other_same = self.correct_signalfinal_prev[k_other, bn_other] == OTHER      # Verify if these points were the same as the previous iteration
        k_ot = k_other[other_same]                                                  # Store these repeated k-points
        bn_ot = bn_other[other_same]                                                # Save their bands
        not_same = np.logical_not(other_same)                                       # Identify which points are different
        k_other = k_other[not_same]                                                 # The different k-points
        bn_other = bn_other[not_same]                                               # Their bands

        ks = np.concatenate((k_error, k_other))                                     # Join the k-points marked as a mistake or other signal
        bnds = np.concatenate((bn_error, bn_other))                                 # Join the k-points' bands

        bands_signaling = np.zeros((self.total_bands, *self.matrix.shape), int)     # The array used to identify the k-points' projection in k-space
        k_index = self.kpoints_index[ks]                                            # k-points' indeces
        ik, jk = k_index[:, 0], k_index[:, 1]                                       # unfold idices
        bands_signaling[bnds, ik, jk] = 1                                           # Marking the projection

        mean_fitler = np.ones((3,3))                                                # It is the kernel used to select the problems' boundary
        self.GRAPH = nx.Graph()                                                     # The new Graph
        self.GRAPH.add_nodes_from(np.arange(len(self.vectors)))                     # Set the nodes
        directions = np.array([[1, 0], [0, 1]])                                     # Auxiliary array with the directions to evaluate the edges' existence

        for bn, band in enumerate(bands_signaling[self.min_band: self.max_band+1]):
            # For each band construct the new graph
            bn += self.min_band                                                                     # Initial band correction
            if np.sum(band) > self.nks*0.05:
                # If there are more than 5% of marked points, the boundaries of 
                # the problem are considered a problem too.
                identify_points = correlate(band, mean_fitler, output=None,
                                            mode='reflect', cval=0.0, origin=0) > 0                 # The mean kernel is applied
            else:
                # Otherwise, just the marked points are considered
                identify_points = band > 0
            edges = []
            for ik, row in enumerate(identify_points):
                for jk, need_correction in enumerate(row):
                    # For each not identified point the graph is built
                    kp = self.matrix[ik, jk]                                                        # The k-point on position ik, jk in k-space
                    if need_correction and kp not in self.degenerate_final:
                        # If the point was identified as an error or as degenerate
                        # It does not have an edge in the graph
                        continue
                    for direction in directions:
                        # It is verified for each direction (Down, Right) if the points,
                        # and their neighbors belong to the same band
                        ikn, jkn = np.array([ik, jk]) + direction                                   # Neighbor's idices in k-space
                        if ikn >= self.matrix.shape[0] or jkn >= self.matrix.shape[1]:
                            # The neighbor is outside of the boundaries
                            continue
                        kneig = self.matrix[ikn, jkn]                                               # Neighbor k-point
                        if not identify_points[ikn, jkn]:
                            p = kp + (self.bands_final[kp, bn] - self.min_band)*self.nks            # The kpoint's node id
                            pn = kneig + (self.bands_final[kneig, bn] - self.min_band)*self.nks     # The neighbor's node id
                            edges.append([p, pn])                                                   # Establish an edge between nodes p (k-point) and pn (neighbor)
            edges = np.array(edges)
            self.GRAPH.add_edges_from(edges)                                                        # Build the identified edges
            self.correct_signalfinal_prev = np.copy(self.correct_signalfinal)                       # Save the currect result
            self.correct_signalfinal[k_ot, bn_ot] = CORRECT-1                                       # Signaling as CORRECT the repeated k-points

    def report(self):
        self.final_report += '*************************************************************************************************\n'
        self.final_report += '|                                       SOLUTION REPORT                                         |\n'
        self.final_report += '*************************************************************************************************\n\n'

        self.final_report += f'\n\tLegend of report:\n'
        self.final_report += EVALUATE_RESULT_HELP
        self.final_report += '\n'
        self.final_report += VALIDATE_RESULT_HELP
        self.final_report += '\n'
        report_s1, report_a1 = self.print_report(self.signal_final, 'Final Report for dot-product information', show=False)
        self.final_report += report_s1
        self.final_report += '\n'
        retport_s2, report_a2 = self.print_report(self.correct_signalfinal, 'Validation Report for energy continuity criteria', show=False)
        self.final_report += retport_s2
        self.final_report += '\n'

        p_report, problems = self.solved_problems_info

        self.final_report += '\n\tSummary:\n'
        self.final_report += p_report

        point2k_bn = lambda p: (p % self.nks, p // self.nks + self.min_band)

        for d1, d2 in problems:
            k1, bn1 = point2k_bn(d1)
            k2, bn2 = point2k_bn(d2)
            Bk1 = self.bands_final[k1] == bn1                               # Find in which  band the k-point was attributed
            Bk2 = self.bands_final[k2] == bn2                               # Find in which  band the k-point was attributed
            bn1 = np.argmax(Bk1) if np.sum(Bk1) != 0 else bn1               # Final band
            bn2 = np.argmax(Bk2) if np.sum(Bk2) != 0 else bn2               # Final band
        
            self.final_report += f'\n\t\t K-point: {k1} bands: {bn1}, {bn2}' # Report
        
        if len(problems) > 0:
            self.final_report += f'\n\t\tThese points were corrected.'
        
        filter_deg = lambda k1, bn1, bn2: np.any(np.all(np.array([k1, bn1, bn2]) == self.degenerate_final, axis=1))
        degenerates = []

        for i, (d1, d2) in enumerate(self.degenerates):
            k1, bn1 = point2k_bn(d1)
            k2, bn2 = point2k_bn(d2)
            Bk1 = self.bands_final[k1] == bn1                               # Find in which  band the k-point was attributed
            Bk2 = self.bands_final[k2] == bn2                               # Find in which  band the k-point was attributed
            bn1 = np.argmax(Bk1) if np.sum(Bk1) != 0 else bn1               # Final band
            bn2 = np.argmax(Bk2) if np.sum(Bk2) != 0 else bn2               # Final band
            if filter_deg(k1, bn1, bn2):
                degenerates.append([k1, bn1, bn2])

        if len(degenerates) > 0:
            n = len(degenerates)
            self.final_report += f'\n\n\tNumber of degenerate points: {n}\n'
            for k1, bn1, bn2 in degenerates:
                self.final_report += f'\n\t\t * K-point: {k1} Bands: {bn1}, {bn2}'
            self.final_report += f'\n\n\t   Run the basis correction program on these bands.'

        if len(self.degenerate_final) > 0:
            n = len(self.degenerate_final)
            self.final_report += f'\n\n\tFound {n} points with one or more neighbor with a dot-product between 0.5 and 0.8.'
            self.final_report += f'\n\t\tMay not be degenerate points under energy criteria.'
            self.final_report += f'\n\t\tSo they were not signaled and no corrections were applied.'
            self.final_report += f'\n\t\tHowever, they are saved in the degeneratefinal.npy file, in case they need analysis.'
            if self.logger.level == logging.DEBUG:
                self.final_report += '\n\t\tPoints:'
                i_sort = np.argsort([k for k, _, _ in self.degenerate_final])
                for k, bn1, bn2 in self.degenerate_final[i_sort]:
                    self.final_report += f'\n\t\t * K-point: {k} \tBands: {bn1}, {bn2}'

        TOL_USABLE = 0.95           # Minimum score to consider a band as usable.
        n_recomended = 0
        max_solved = 0

        for i, _ in enumerate(self.final_score):
            if np.sum(report_a1[i, NOT_SOLVED]) > 0:
                break
            max_solved += 1
        
        self.max_solved = max_solved

        n_max = max_solved
        for i, s in enumerate(self.final_score):
            if s <= TOL_USABLE or i > max_solved or np.sum(report_a2[i, [NOT_SOLVED, MISTAKE]]) > 0:
                break
            n_recomended += 1
        
        self.final_report += f'\n\n\tThe program solved {self.max_solved} bands.'
        self.final_report += f'\n\n\tBands from 0 up to {n_recomended-1} can be used.'

        if self.max_solved > n_recomended:
            self.final_report += f'\n\n\tNote that there may be more bands usable but a human verification is required.'
            n_max = n_recomended

        if len(degenerates) > 0:
            self.final_report += f'\n\n\tRun program basis rotation in the following manner:'
            self.final_report += f'\n\n\t\t $ berry basis {n_max - 1}'

        self.final_report += '\n\n*************************************************************************************************\n'

        return self.final_report

    def solve(self, step: float=0.1, min_alpha: float=0) -> None:
        '''
        This method is the main algorithm which iterates between solutions
        trying to find the best result for the material.

        Parameters
            step : float
                It is the iteration value which is used to relax the alpha value.
                (default 0.1)
            min_alpha : float
                The minimum alpha.
                (default 0)
        '''
        ###########################################################################
        # Initial preparation of data structures
        # The previous and best result are stored
        ###########################################################################
        ALPHA = 0.5   # The initial alpha is 0.5. 0.5*<i|j> + 0.5*f(E)
        COUNT = 0     # Counter iteration
        bands_final_flag = True
        self.final_report = ''
        self.bands_final_prev = np.copy(self.bands_final)
        self.best_bands_final = np.copy(self.bands_final)
        self.best_score = np.zeros(self.total_bands, dtype=float)
        self.final_score = np.zeros(self.total_bands, dtype=float)
        self.signal_final = np.zeros((self.nks, self.total_bands), dtype=int)
        self.correct_signalfinal_prev = np.full(self.signal_final.shape, -1, int)
        self.degenerate_best = None
        max_solved = 0  # The maximum number of solved bands

        ###########################################################################
        # Algorithm
        ###########################################################################
        while bands_final_flag and ALPHA >= min_alpha:
            COUNT += 1
            self.logger.info()
            self.logger.info(f'\n\n\t* Iteration: {COUNT} - Clustering samples for Alpha: {ALPHA} ')
            self.get_components(alpha=ALPHA)                    # Obtain components from a Graph

            self.logger.info('\n\t\tCalculating output')        
            self.obtain_output()                            # Compute the result
            self.print_report(self.signal_final, f'Report Number: {COUNT} considering dot-product information')                           # Print result

            self.logger.info('\n\t\tValidating result using energy continuity criteria')     
            self.correct_signal()                           # Evaluate the energy continuity and perform a new Graph
            self.print_report(self.correct_signalfinal, f'Validation Report Number: {COUNT} considering  energy continuity criteria')     # Print result
            
            # Verification if the result is similar to the previous one
            bands_final_flag = np.sum(np.abs(self.bands_final_prev - self.bands_final)) != 0
            self.bands_final_prev = np.copy(self.bands_final)

            # Verify and store the best result
            # To be a better result it has to be better score and all k points attributed for all the first max_solved bands
            solved = 0
            for bn, score in enumerate(self.final_score):
                best_score = self.best_score[bn]
                not_solved_prev = np.sum(self.correct_signalfinal_prev[:, bn] == NOT_SOLVED)
                not_solved = np.sum(self.correct_signalfinal[:, bn] == NOT_SOLVED)
                if score != 0 and score >= best_score and not_solved <= not_solved_prev:
                    solved += 1
                else:
                    break
            if solved >= max_solved:
                self.best_bands_final = np.copy(self.bands_final)
                self.best_score = np.copy(self.final_score)
                self.best_signal_final = np.copy(self.signal_final)
                self.degenerate_best = np.copy(self.degenerate_final)
                max_solved = solved
            else:
                self.bands_final = np.copy(self.best_bands_final)
                self.final_score = np.copy(self.best_score)
                self.signal_final = np.copy(self.best_signal_final)
                self.degenerate_final = np.copy(self.degenerate_best)        
                self.obtain_output()
                self.correct_signal()
            ALPHA -= step
        
        # The best result is maintained
        self.bands_final = np.copy(self.best_bands_final)
        self.final_score = np.copy(self.best_score)
        self.signal_final = np.copy(self.best_signal_final)
        self.degenerate_final = np.copy(self.degenerate_best)
        self.max_solved = max_solved
        self.logger.info(self.report())

class COMPONENT:
    '''
    This object contains the information that constructs a component,
    and also it has functions that are necessary to establish
    relations between components.

    Atributes
        GRAPH : Graph
            It is a graph object that contains nodes and edges defined as G = (V, E).
        N : int
            It is the number of nodes that the graph contains.
        m_shape : Tuple[int, int]
            The shape of the k-space representation.
        nks : int
            The number of k-points.
        kpoints_index : array_like
            Contains the k-points' indices on a k-space projection.
        matrix : array_like
            The matrix that contains in each position the k-point identification.
        position_matrix : None
            Contains the k-points' projection on k-space for points that belong to the graph.
        nodes : array_like
            It contains the node id for all graph's k-points.
        __id__ : string
            It identifies the component of the graph.
        was_modified : bool
            The flag that says if the component was modified in the last iteration.
                default = True
        scores : dict
            Stores the last score for some component with __id__ key.
    
    Methods
        calculate_pointsMatrix() : None
            Calculate the k-points' projection on a matrix representation of k-space.
        get_bands() : array_like
            Get the bands associated with each node.
        validate() : bool
            Verify if the component received as an argument can join with the current component.
        join() : None
            Join the component received as a parameter with the current component.
        calc_boundary() : None
            Only the boundary nodes are necessary. Therefore this function computes
                these essential nodes and uses them to compare components.
        get_cluster_score() : float
            This function returns the similarity between components taking
                into account the dot product of all essential points and their
                energy value.
    '''
    def __init__(self, component: nx.Graph, kpoints_index:np.ndarray, matrix: np.ndarray) -> None:
        '''
        Setup the component information.

        Parameters
            GRAPH : Graph
                It is a graph object that contains nodes and edges defined as G = (V, E).
            N : int
                It is the number of nodes that the graph contains.
            m_shape : Tuple[int, int]
                The shape of the k-space representation.
            nks : int
                The number of k-points.
            kpoints_index : array_like
                Contains the k-points' indices on a k-space projection.
            matrix : array_like
                The matrix that contains in each position the k-point identification.
            position_matrix : None
                Contains the k-points' projection on k-space for points that belong to the graph.
            nodes : array_like
                It contains the node id for all graph's k-points.
            __id__ : string
                It identifies the component of the graph.
            was_modified : bool
                The flag that says if the component was modified in the last iteration.
                    default = True
            scores : dict
                Stores the last score for some component with __id__ key.
        '''
        self.GRAPH = component
        self.N = self.GRAPH.number_of_nodes()
        self.m_shape = matrix.shape
        self.nks = self.m_shape[0]*self.m_shape[1]
        self.kpoints_index = np.array(kpoints_index)
        self.matrix = matrix
        self.positions_matrix = None
        self.nodes = np.array(self.GRAPH.nodes)

        self.__id__ = str(self.nodes[0])
        self.was_modified = True
        self.scores = {}

    def calculate_pointsMatrix(self) -> None:
        '''
        Calculate the k-points' projection on a matrix representation of k-space.
        '''
        self.positions_matrix = np.zeros(self.m_shape, int)                 # Position matrix for k-points projection
        index_points = self.kpoints_index[self.nodes % self.nks]            # Get the k-points' indices
        self.k_points = self.nodes % self.nks                               # Transform node id to k-point notation
        self.bands_number = dict(zip(self.nodes % self.nks,
                                     self.nodes//self.nks))                 # A dictionary that links the initial band to a k-point
        self.positions_matrix[index_points[:, 0], index_points[:, 1]] = 1   # Mark the k-point projection

    def get_bands(self) -> None:
        '''
        Get the bands associated with each node.

        Returns
            k_bands : array_like
                An array with bands information.
        '''
        self.k_points = self.nodes % self.nks                               # Get the k-point from node
        k_bands = self.nodes//self.nks                                      # Get the k-point's band from node
        self.bands_number = dict(zip(self.nodes % self.nks,
                                     self.nodes//self.nks))                 # A dictionary that links the initial band to a k-point
        bands, counts = np.unique(k_bands, return_counts=True)              # Count the number of nodes with each band
        self.bands = bands[np.argsort(counts)[::-1]]                        # Save in decreasing order
        return k_bands

    def validate(self, component : COMPONENT) -> bool:
        '''
        Verify if the component received as a parameter can join with the current component.

        Parameters
            component : COMPONENT
                It is another COMPONENT object that is trying to join the current component.
        
        Returns
            validate() : bool
                If it is true, the component does not have overlaying between k-points,
                and the sum of nodes does not exceed the total number of k-points
        '''
        if self.positions_matrix is None:
            # Calculates the k-space projection if it does not exist
            self.calculate_pointsMatrix()
        # Computes the overlaying between components by XOR operation between k-space projections
        N = np.sum(self.positions_matrix ^ component.positions_matrix)
        return (component.N <= self.nks - self.N and N == self.N+component.N)

    def join(self, component : COMPONENT) -> None:
        '''
        Join the component received as a parameter with the current component.

        Parameter
            component : COMPONENT
                It is another COMPONENT object to be joined to the current component.
        '''
        del component.scores                            # Clear the scores' information of the component
        self.was_modified = True                        # The Component is marked as modified
        G = nx.Graph(self.GRAPH)                        # Copy the Graph
        G.add_nodes_from(component.GRAPH)               # Add the new nodes
        self.GRAPH = G                                  # Set the new Graph
        self.N = self.GRAPH.number_of_nodes()           # Atualize the number of nodes
        self.nodes = np.array(self.GRAPH.nodes)
        # Update the other attributes
        self.calculate_pointsMatrix()
        self.calc_boundary()

    def calc_boundary(self) -> None:
        '''
        Only the boundary nodes are necessary. Therefore this function computes
        these essential nodes and uses them to compare components.
        '''
        if self.positions_matrix is None:
            # Calculates the k-space projection if it does not exist
            self.calculate_pointsMatrix()
        # Prewitt operator as the kernel for processing the positions
        # matrix to obtain the boundary points
        #       | -1 0 +1 |                       | -1 -1 -1 |      
        # Gx =  | -1 0 +1 |         Gy =  Gx^T =  |  0  0  0 |
        #       | -1 0 +1 |                       | +1 +1 +1 |
        Gx = np.array([[-1, 0, 1]]*3)
        Gy = np.array([[-1, 0, 1]]*3).T
        # Compute the convolution between the Prewitt Kernel and the position matrix
        # Ax = Gx*position_matrix   Ay = Gy*position_matrix
        # Boundary = sqrt( Ax^2 + Ay^2)
        Ax = correlate(self.positions_matrix, Gx, output=None,
                       mode='reflect', cval=0.0, origin=0)
        Ay = correlate(self.positions_matrix, Gy, output=None,
                       mode='reflect', cval=0.0, origin=0)
        self.boundary = np.sqrt(Ax**2+Ay**2)*self.positions_matrix
        # Maintain  all marked points
        self.boundary = (self.boundary > 0)
        self.k_edges = self.matrix[self.boundary]
        if len(self.k_edges) == 0:
            self.k_edges = self.nodes % self.nks

    def get_cluster_score(self, cluster : COMPONENT, min_band : int, max_band : int,
                          neighbors : np.ndarray, energies : np.ndarray, connections : np.ndarray, alpha : float = 0.5) -> float:
        '''
        This function returns the similarity between components taking
        into account the dot product of all essential points and their
        energy value.

        Parameters
            cluster : COMPONENT
                It is a component with which the similarity is calculated.
            min_band : int
                It is an integer that gives the minimum band used for clustering.
            max_band : int
                It is an integer that gives the maximum band used for clustering.
            neighbors : array_like
                It is an array that identifies the neighbors of each k point.
            energies : array_like
                It is an array of the energy values inside a matrix.
            connections : array_like
                It is an array with the dot product between k points
                    and his neighbors.
            alpha : float
                This is the weight given to the dotproduct value to compute the score.
                    score = alpha*<i|j> + (1-alpha)*f(E_i)
        Return
            score: float
                It is a float that represents the similarity between components.
        '''
        def difference_energy(bn1 : int, bn2 : int, iK1 : Tuple[int, int], iK2: Tuple[int, int], Ei : float = None) -> float:
            '''
            Compute the score for the k-point with their neighbor by comparing all possible energies with Ei energy.

            Parameters
                bn1 : int
                    k-point's band.
                bn2 : int
                    neighbor's band.
                iK1 : Tuple[int, int]
                    K-point's indices on k-space.
                iK2 : Tuple[int, int]
                    Neighbor's indices on k-space.
                Ei : float
                    Energy to comparison. The default is None
            
            Return
                score : float
                    If the neighbor's energy is the closest one to Ei the score is one, otherwise is between 0 and 1.
            '''
            ik1, jk1 = iK1                                                  # Unfold k-point's indices
            ik_n, jk_n = iK2                                                # Unfold neighbor's indices
            Ei = energies[bn1, ik1, jk1] if Ei is None else Ei              # The k-point Energy is used as default
            bands = np.arange(min_band, max_band+1)                         # Bands in analysis
            min_energy = np.min([np.abs(Ei-energies[bn, ik_n, jk_n])        # The energy difference between Ei and all possibilities
                                    for bn in bands])
            delta_energy = np.abs(Ei-energies[bn2, ik_n, jk_n])             # Actual energy difference
            return min_energy/delta_energy if delta_energy else 1           # score
        
        def fit_energy(bn1 : int, bn2 : int, iK1 : Tuple[int, int], iK2: Tuple[int, int]) -> float:
            '''
            Computes the best energy approximation for the neighbor's position.

            Parameters
                bn1 : int
                    k-point's band.
                bn2 : int
                    neighbor's band.
                iK1 : Tuple[int, int]
                    K-point's indices on k-space.
                iK2 : Tuple[int, int]
                    Neighbor's indices on k-space.
            
            Return
                score : float
                    Result of difference_energy function using the computed Energy.
            '''
            N = 4                                                               # Number of points to take account into the curve fitting
            ik1, jk1 = iK1                                                      # Unfold k-point's indices
            ik_n, jk_n = iK2                                                    # Unfold neighbor's indices

            ###########################################################################
            # Preparation of the points' indices
            ###########################################################################
            I = np.full(N+1,ik1)                                                # Repeat N+1 times the ik1 value  
            J = np.full(N+1,jk1)                                                # Repeat N+1 times the jk1 value  
            flag = ik1 == ik_n                                                  # Identify the neighbor's direction
            # Take the (i,j) indices of N+1 points
            i = I if flag else I + np.arange(0,N+1)*np.sign(ik1-ik_n)
            j = J if not flag else J + np.arange(0,N+1)*np.sign(jk1-jk_n)
            
            if not flag:
                # If the neighbor is in jk's direction then i is corrected to be inside boundaries.
                # The shape of js indices is corrected
                i = i[i >= 0]
                i = i[i < self.m_shape[0]]
                j = np.full(len(i), jk1)
            else:
                # If the neighbor is in ik's direction then j is corrected to be inside boundaries.
                # The shape of is indices is corrected
                j = j[j >= 0]
                j = j[j < self.m_shape[1]]
                i = np.full(len(j), ik1)

            ###########################################################################
            # Computes the best energy using the points above defined
            ###########################################################################
            ks = self.matrix[i, j]                                                  # Select k-points on positions delimited by the (i,j) indices
            f = lambda e: e in self.k_points                                        # Auxiliar lambda function to verify if an e point is inside component's k-points
            exist_ks = list(map(f, ks))                                             # Apply f to all ks
            ks = ks[exist_ks]                                                       # Maintain only the existent k-points
            if len(ks) <= 3:
                # It is necessary at least 3 points to fitting a second order curve
                # If there are not enough points is used the difference_energy's default
                return difference_energy(bn1, bn2, iK1, iK2)
            aux_bands = np.array([self.bands_number[kp] for kp in ks])              # Get the ks' bands
            bands = aux_bands + min_band                                            # Initial band correction
            # Use the existent k-points' indices
            i = i[exist_ks]
            j = j[exist_ks]
            Es = energies[bands, i, j]                                              # Get the ks' energies
            X = i if jk1 == jk_n else j                                             # Obtain the x values for Es.
            new_x = ik_n if jk1 == jk_n else jk_n                                   # Get the position to approximate the energy

            pol = lambda x, a, b, c: a*x**2 + b*x + c                               # Second order polynomial
            popt, pcov = curve_fit(pol, X, Es)                                      # Get the optimum parameters
            Enew = pol(new_x, *popt)                                                # Calculates the new energy
            # Ei = energies[bn1, ik1, jk1]
            # LOG.debug(f'Actual Energy: {Ei} Energy founded: {Enew} for {bn1} with {len(i)} points.')
            return difference_energy(bn1, bn2, iK1, iK2, Ei = Enew)                 # Score

        ###########################################################################
        # Computes the final score between components
        ###########################################################################
        if not cluster.was_modified:
            # If the cluster was not modified the previous result is maintained
            return self.scores[cluster.__id__]

        score = 0
        for k in self.k_edges:
            # Each k-point is compared with his respective neighbor
            # that belongs to the comparison component
            bn1 = self.bands_number[k] + min_band                                   # k-point's band
            ik1, jk1 = self.kpoints_index[k]                                        # k-point idices
            for i_neig, k_n in enumerate(neighbors[k]):
                # k-point's neighbors
                if k_n == -1 or k_n not in cluster.k_edges:
                    # If the neighbor is not a valid point the score for that point is 0
                    continue
                ik_n, jk_n = self.kpoints_index[k_n]                                # neighbor's indices
                bn2 = cluster.bands_number[k_n]+min_band                            # neighbor's band
                connection = connections[k, i_neig, bn1, bn2]                       # Dot product between k-point and his neighbor
                energy_val = fit_energy(bn1, bn2, (ik1, jk1), (ik_n, jk_n))         # Computes the energy continuity score
                score += alpha*connection + (1-alpha)*energy_val                    # Calculates the final k-point score
        score /= len(self.k_edges)*4                                                # Get the final score
        self.scores[cluster.__id__] = score                                         # Store the score
        return score
