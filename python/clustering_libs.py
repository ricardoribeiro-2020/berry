"""This module contains the algorithm behind band classification.
It determines which points belong to each band used for posterior calculations.
The algorithm uses machine learning techniques to cluster the data.
"""

import numpy as np
import networkx as nx
import contatempo
import time
import os
from scipy.ndimage import correlate
from write_k_points import bands_numbers
from multiprocessing import Process, Pool, Manager, Value, shared_memory
from functools import partial
from log_libs import log
from loaddata import version
from scipy.optimize import curve_fit

CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

N_NEIGS = 4

LOG = log('clustering', 'Band Clustering', version)

def evaluate_result(values):
    '''
    This function attributes the correspondent signal using
    the dot product between each neighbor.

    INPUT
    values: It is an array that contains the dot product
            between the k point and all neighbors.

    C -> Mean connection of each k point

    OUTPUT
    Value :                              Description
    0     :                        The point is not solved
    1     :  MISTAKE               c <= 0.2
    2     :  DEGENERATE            It is a degenerate point.
    3     :  POTENTIAL_MISTAKE     C <= 0.8
    4     :  POTENTIAL_CORRECT     0.8 < C < 0.9
    5     :  CORRECT               C > 0.9
    '''

    TOL = 0.9
    TOL_DEG = 0.8
    TOL_MIN = 0.2

    value = np.mean(values)
    if value > TOL:
        return CORRECT

    if value > TOL_DEG:
        return POTENTIAL_CORRECT

    if value > TOL_MIN and value < TOL_DEG:
        return POTENTIAL_MISTAKE

    return MISTAKE

def evaluate_point(k, bn, k_index, k_matrix, signal, bands, energies):
    N = 4 #Number of points to fit the curve
    mach_bn = bands[k, bn]
    sig = signal[k, bn]
    ik, jk = k_index[k]
    Ek = energies[k, mach_bn]

    def difference_energy(Ek, Enew):
        min_energy = np.min(np.abs(Enew-energies[k]))
        delta_energy = np.abs(Enew-Ek)
        return min_energy/delta_energy if delta_energy else 1

    directions = np.array([[1,0], [0,1], [-1,0], [0,-1]]) # Down, Right, Up, Left
    energy_vals = []
    for direction in directions:
        n = np.repeat(np.arange(1,N+1),2).reshape(N,2)
        kn_index = n*direction + np.array([ik, jk])
        i, j = kn_index[:, 0], kn_index[:, 1]
        flag = len(np.unique(i)) > 1
        if flag:
            i = i[i >= 0]
            i = i[i < k_matrix.shape[0]]
            j = np.full(len(i), j[0])
        else:
            j = j[j >= 0]
            j = j[j < k_matrix.shape[1]]
            i = np.full(len(j), i[0])
        
        ks = k_matrix[i, j] if len(i) > 0 else []
        if len(ks) == 0:
            energy_vals.append(1)
            continue
        if len(ks) <= 3:
            Eneig = energies[ks[0], bands[ks[0], bn]]
            energy_vals.append(difference_energy(Ek, Eneig))
            continue
        
        k_bands = bands[ks, bn]
        Es = energies[ks, k_bands]
        X = i if flag else j
        new_x = ik if flag else jk
        pol = lambda x, a, b, c: a*x**2 + b*x + c
        popt, pcov = curve_fit(pol, X, Es)
        Enew = pol(new_x, *popt)
        energy_vals.append(difference_energy(Ek, Enew))
    
    TOL = 0.9
    N_Neighs = 4
    
    energy_vals = np.array(energy_vals)
    scores = (energy_vals > TOL)*1
    score = np.sum(scores)

    CORRECT = 4
    MISTAKE = 1
    OTHER = 3
    
    if score == N_Neighs:
        return CORRECT, scores
    if score == 0:
        return MISTAKE, scores
    return OTHER, scores


class MATERIAL:
    '''
    This object contains all information about the material that
    will be used to solve their bands' problem.
    '''
    def __init__(self, nkx, nky, nbnd, nks, eigenvalues,
                 connections, neighbors, n_process=1):
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

    def make_BandsEnergy(self):
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

    def make_kpointsIndex(self):
        My, Mx = np.meshgrid(np.arange(self.nky), np.arange(self.nkx))
        self.matrix = My*self.nkx+Mx
        counts = np.arange(self.nks)
        self.kpoints_index = np.stack([counts % self.nkx, counts//self.nkx],
                                      axis=1)

    def make_vectors(self, min_band=0, max_band=-1):
        '''
        It transforms the information into more convenient data structures.

        INPUT
        min_band: An integer that gives the minimum band that clustering will use.
                  default: 0
        max_band: An integer that gives the maximum band that clustering will use.
                  default: All

        RESULT
        self.vectors: [kx_b, ky_b, E_b]
            k = (kx, ky)_b: k point
            b: band number
        self.degenerados: It marks the degenerate points
        self.GRPAH: It is a graph in which each node represents a vector.
        self.energies: It contains the energy values for each band distributed
                       in a matrix.
        '''
        process_name = 'Making Vectors'
        LOG.percent_complete(0, 100, title=process_name)

        self.GRAPH = nx.Graph()
        self.min_band = min_band
        self.max_band = max_band
        nbnd = self.nbnd if max_band == -1 else max_band+1
        self.make_kpointsIndex()
        energies = self.make_BandsEnergy()
        LOG.percent_complete(20, 100, title=process_name)

        n_vectors = (nbnd-min_band)*self.nks
        ik = np.tile(self.kpoints_index[:, 0], nbnd-min_band)
        jk = np.tile(self.kpoints_index[:, 1], nbnd-min_band)
        bands = np.arange(min_band, nbnd)
        eigenvalues = self.eigenvalues[:, bands].T.reshape(n_vectors)
        self.vectors = np.stack([ik, jk, eigenvalues], axis=1)
        LOG.percent_complete(100, 100, title=process_name)

        self.GRAPH.add_nodes_from(np.arange(n_vectors))

        self.degenerados = []
        
        def obtain_degenerates(vectors):
            degenerates = []
            for i, v in vectors:
                degenerado = np.where(np.all(np.isclose(self.vectors[i+1:]-v, 0),
                                    axis=1))[0]
                if len(degenerado) > 0:
                    LOG.debug(f'Found degenerete point for {i}')
                    degenerates += [[i, d+i+1] for d in degenerado]
            return degenerates
        
        '''
        for i, v in enumerate(self.vectors):
            LOG.percent_complete(50 + int(i*50/len(self.vectors)), 100, title=title)
            degenerado = np.where(np.all(np.isclose(self.vectors[i+1:]-v, 0),
                                  axis=1))[0]
            if len(degenerado) > 0:
                self.degenerados += [[i, d+i+1] for d in degenerado]
                LOG.debug(f'Found degenerete point for {i}')'''
        

        self.degenerados = self.parallelize('Finding degenerate points', obtain_degenerates, enumerate(self.vectors))

        if len(self.degenerados) > 0:
            LOG.info('Degenerate Points: ')
            for d in self.degenerados:
                LOG.info(f'\t{d}')

        self.ENERGIES = energies
        self.nbnd = nbnd-min_band

        self.bands_final = np.full((self.nks, self.total_bands), -1, dtype=int)

    def get_neigs(self, i):
        return list(self.GRAPH.neighbors(i))

    def find_path(self, i, j):
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

    def make_connections(self, tol=0.95):
        '''
        This function evaluates the connection between each k point,
        and adds an edge to the graph if its connection is greater
        than a tolerance value (tol).

        <i|j>: The dot product between i and j represents its connection

        INPUT
        tol: It is the minimum connection value that will be accepted as
             an edge.
             default: 0.95
        '''

        def connection_component(vectors):
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

        edges = self.parallelize('Computing Edges', connection_component, range(len(self.vectors)))

        self.GRAPH.add_edges_from(edges)

        for d1, d2 in self.degenerados:
            '''
            The degenerate points may cause problems.
            The algorithm below finds its problems and solves them.
            '''

            if not self.find_path(d1, d2):
                continue
            N1 = np.array(self.get_neigs(d1))
            N2 = np.array(self.get_neigs(d2))
            if len(N1) == 0 or len(N2) == 0:
                continue
            LOG.info(f'Problem:\n\t{d1}: {N1}\n\t{d2}:{N2}')
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

            LOG.info(f'Solution:\n\t{d1}: {N1_}\n\t{d2}:{N2_}')
            for k in N1:
                self.GRAPH.remove_edge(k, d1)
            for k in N2:
                self.GRAPH.remove_edge(k, d2)

            for k in N1_:
                self.GRAPH.add_edge(k, d1)
            for k in N2_:
                self.GRAPH.add_edge(k, d2)

    def parallelize(self, process_name, f, iterator, *args, verbose=True):
        process = []
        iterator = list(iterator)
        N = len(iterator)
        if verbose:
            LOG.debug(f'Starting Parallelization for {process_name} with {N} values')
        if verbose:
            LOG.percent_complete(0, N, title=process_name)

        def parallel_f(result, per, iterator, *args):
            value = f(iterator, *args)
            if value is not None:
                result += f(iterator, *args)
            per[0] += len(iterator)
            if verbose:
                LOG.percent_complete(per[0], N, title=process_name)
        
        result = Manager().list([])
        per = Manager().list([0])
        f_ = partial(parallel_f,  result, per)

        n = N//self.n_process
        for i_start in range(self.n_process):
            j_end = n*(i_start+1) if i_start < self.n_process-1\
                else n*(i_start+1) + N % self.n_process
            i_start = i_start*n
            p = Process(target=f_, args=(iterator[i_start: j_end], *args))
            p.start()
            process.append(p)

        while len(process) > 0:
            p = process.pop(0)
            p.join()

        if verbose:
            print()

        return np.array(result)

    def get_components(self, tol=0.5):
        '''
        The make_connections function constructs the graph, in which
        it can detect components well constructed.
            - A component is denominated solved when it has all
              k points attributed.
            - A cluster is a significant component that can not join
              with any other cluster.
            - Otherwise, It is a sample that has to be grouped with
              some cluster.
        '''

        LOG.info('\n\nNumber of Components: ')
        LOG.info(f'{nx.number_connected_components(self.GRAPH)}')
        self.components = [COMPONENT(self.GRAPH.subgraph(c),
                                     self.kpoints_index,
                                     self.matrix)
                           for c in nx.connected_components(self.GRAPH)]
        index_sorted = np.argsort([component.N
                                   for component in self.components])[::-1]
        self.solved = []
        clusters = []
        samples = []
        for i in index_sorted:
            component = self.components[i]
            if component.N == self.nks:
                self.solved.append(component)
                continue
            component.calculate_pointsMatrix()
            component.calc_boundary()
            if len(clusters) == 0:
                clusters.append(component)
                continue
            if not np.any([cluster.validate(component)
                           for cluster in clusters]):
                clusters.append(component)
            else:
                samples.append(component)
        LOG.info(f'    Phase 1: {len(self.solved)}/{self.nbnd} Solved')
        LOG.info(f'    Initial clusters: {len(clusters)} Samples: {len(samples)}')

        count = np.array([0, len(samples)])
        while len(samples) > 0:
            evaluate_samples = np.zeros((len(samples), 2))
            for i_s, sample in enumerate(samples):
                scores = np.zeros(len(clusters))
                for j_s, cluster in enumerate(clusters):
                    if not cluster.validate(sample):
                        continue
                    if len(sample.k_edges) == 0:
                        sample.calculate_pointsMatrix()
                        sample.calc_boundary()
                    scores[j_s] = sample.get_cluster_score(cluster,
                                                           self.min_band,
                                                           self.max_band,
                                                           self.neighbors,
                                                           self.ENERGIES,
                                                           self.connections,
                                                           tol=tol)
                evaluate_samples[i_s] = np.array([np.max(scores),
                                                np.argmax(scores)])

            for cluster in clusters:
                cluster.was_modified = False
            arg_max = np.argmax(evaluate_samples[:, 0])
            sample = samples.pop(arg_max)
            score, bn = evaluate_samples[arg_max]
            bn = int(bn)
            count[0] += 1
            clusters[bn].join(sample)
            clusters[bn].was_modified = True
            LOG.percent_complete(count[0], count[1], title='Clustering Samples')
            LOG.debug(f'{count[0]}/{count[1]} Sample corrected: {score}')
            if clusters[bn].N == self.nks:
                print('Cluster Solved')
                self.solved.append(clusters.pop(bn))

        LOG.info(f'    Phase 2: {len(self.solved)}/{self.nbnd} Solved')

        if len(self.solved)/self.nbnd < 1:
            LOG.info(f'    New clusnters: {len(clusters)}')


        labels = np.empty(self.nks*self.nbnd, int)
        count = 0
        for solved in self.solved:
            labels[solved.nodes] = count
            count += 1

        for cluster in clusters:
            # cluster.save_boundary(f'cluster_{count}') # Used for analysis
            labels[cluster.nodes] = count
            count += 1
        
        self.clusters = clusters
        return labels

    def obtain_output(self):
        '''
        This function prepares the final data structures
        that are essential to other programs.
        '''

        self.degenerate_final = []

        solved_bands = []
        for solved in self.solved:
            bands = solved.get_bands()
            bn = solved.bands[0] + self.min_band
            solved.bands = solved.bands[1:]
            while bn in solved_bands:
                bn = solved.bands[0] + self.min_band
                solved.bands = solved.bands[1:]
            solved_bands.append(bn)
            self.bands_final[solved.k_points, bn] = bands + self.min_band

            for k in solved.k_points:
                bn1 = solved.bands_number[k] + self.min_band
                connections = []
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    if k_neig == -1:
                        continue
                    bn2 = solved.bands_number[k_neig] + self.min_band
                    connections.append(self.connections[k, i_neig, bn1, bn2])

                self.signal_final[k, bn] = evaluate_result(connections)

        clusters_sort = np.argsort([c.N for c in self.clusters])
        for i_arg in clusters_sort[::-1]:
            cluster = self.clusters[i_arg]
            bands = cluster.get_bands()
            bn = cluster.bands[0] + self.min_band
            cluster.bands = cluster.bands[1:]
            while bn in solved_bands and len(cluster.bands) > 0:
                bn = cluster.bands[0] + self.min_band
                cluster.bands = cluster.bands[1:]

            if bn in solved_bands and len(cluster.bands) == 0:
                break

            solved_bands.append(bn)
            self.bands_final[cluster.k_points, bn] = bands + self.min_band
            for k in cluster.k_points:
                bn1 = cluster.bands_number[k] + self.min_band
                connections = []
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    if k_neig == -1:
                        continue
                    if k_neig not in cluster.k_points:
                        connections.append(0)
                        continue
                    bn2 = cluster.bands_number[k_neig] + self.min_band
                    connections.append(self.connections[k, i_neig, bn1, bn2])

                self.signal_final[k, bn] = evaluate_result(connections)

        for d1, d2 in self.degenerados:
            k1 = d1 % self.nks
            bn1 = d1 // self.nks + self.min_band
            k2 = d2 % self.nks
            bn2 = d2 // self.nks + self.min_band
            Bk1 = self.bands_final[k1] == bn1
            Bk2 = self.bands_final[k2] == bn2
            bn1 = np.argmax(Bk1) if np.sum(Bk1) != 0 else bn1
            bn2 = np.argmax(Bk2) if np.sum(Bk2) != 0 else bn2

            self.signal_final[k1, bn1] = DEGENERATE
            self.signal_final[k2, bn2] = DEGENERATE

            self.degenerate_final.append([k1, k2, bn1, bn2])

        self.degenerate_final = np.array(self.degenerate_final)

        for bn in range(self.total_bands):
            score = 0
            for k in range(self.nks):
                if self.signal_final[k, bn] == NOT_SOLVED:
                    continue
                kneigs = self.neighbors[k]
                flag_neig = kneigs != -1
                i_neigs = np.arange(N_NEIGS)[flag_neig]
                kneigs = kneigs[flag_neig]
                flag_neig = self.signal_final[kneigs, bn] != NOT_SOLVED
                i_neigs = i_neigs[flag_neig]
                kneigs = kneigs[flag_neig]
                if len(kneigs) == 0:
                    continue
                bn_k = self.bands_final[k, bn]
                bn_neighs = self.bands_final[kneigs, bn]
                k = np.repeat(k, len(kneigs))
                bn_k = np.repeat(bn_k, len(kneigs))
                dps = self.connections[k, i_neigs, bn_k, bn_neighs]
                score += np.mean(dps)
            score /= self.nks
            self.final_score[bn] = score

    def print_report(self, signal_report):
        final_report = '\t====== REPORT ======\n\n'
        bands_report = []
        MAX = np.max(signal_report) + 1
        for bn in range(self.min_band, self.min_band+self.nbnd):
            band_result = signal_report[:, bn]
            report = [np.sum(band_result == s) for s in range(MAX)]
            report.append(np.round(self.final_score[bn], 4))
            bands_report.append(report)

            LOG.info(f'\n  New Band: {bn}\tnr falis: {report[0]}')
            # bands_numbers(self.nkx, self.nky, self.bands_final[:, bn])

        bands_report = np.array(bands_report)
        final_report += '\n Signaling: how many events ' + \
                        'in each band signaled.\n'
        bands_header = '\n Band | '

        header = list(range(MAX)) + [' ']
        for signal, value in enumerate(header):
            n_spaces = len(str(np.max(bands_report[:, signal])))-1
            bands_header += ' '*n_spaces+str(value) + '   '

        final_report += bands_header + '\n'
        final_report += '-'*len(bands_header)

        for bn, report in enumerate(bands_report):
            bn += self.min_band
            final_report += f'\n {bn}{" "*(4-len(str(bn)))} |' + ' '
            for signal, value in enumerate(report):
                if signal < MAX:
                    value = int(value)
                n_max = len(str(np.max(bands_report[:, signal])))
                n_spaces = n_max - len(str(value))
                final_report += ' '*n_spaces+str(value) + '   '

        LOG.info(final_report)
        self.final_report = final_report
    
    def correct_signal(self):
        self.obtain_output()
        del self.GRAPH
        OTHER = 3
        MISTAKE = 1

        self.correct_signalfinal = np.copy(self.signal_final)
        self.correct_signalfinal[self.signal_final == CORRECT] = CORRECT-1

        ks_pC, bnds_pC = np.where(self.signal_final == POTENTIAL_CORRECT)
        ks_pM, bnds_pM = np.where(self.signal_final == POTENTIAL_MISTAKE)

        ks = np.concatenate((ks_pC, ks_pM))
        bnds = np.concatenate((bnds_pC, bnds_pM))

        error_directions = []
        directions = []

        for k, bn in zip(ks, bnds):
            signal, scores = evaluate_point(k, bn, self.kpoints_index,
                                            self.matrix, self.signal_final, 
                                            self.bands_final, self.eigenvalues)
            self.correct_signalfinal[k, bn] = signal
            if signal == OTHER:
                error_directions.append([k, bn])
                directions.append(scores)
            LOG.debug(f'K point: {k} Band: {bn}    New Signal: {signal} Directions: {scores}')

        k_error, bn_error = np.where(self.correct_signalfinal == MISTAKE)
        k_other, bn_other = np.where(self.correct_signalfinal == OTHER)
        other_same = self.correct_signalfinal_prev[k_other, bn_other] == OTHER
        k_ot = k_other[other_same]
        bn_ot = bn_other[other_same]
        not_same = np.logical_not(other_same)
        k_other = k_other[not_same]
        bn_other = bn_other[not_same]

        ks = np.concatenate((k_error, k_other))
        bnds = np.concatenate((bn_error, bn_other))

        bands_signaling = np.zeros((self.total_bands, *self.matrix.shape), int)
        k_index = self.kpoints_index[ks]
        ik, jk = k_index[:, 0], k_index[:, 1]
        bands_signaling[bnds, ik, jk] = 1

        mean_fitler = np.ones((3,3))
        self.GRAPH = nx.Graph()
        self.GRAPH.add_nodes_from(np.arange(len(self.vectors)))
        directions = np.array([[1, 0], [0, 1]])

        for bn, band in enumerate(bands_signaling[self.min_band: self.max_band+1]):
            _bn = bn
            bn += self.min_band
            if np.sum(band) > self.nks*0.05:
                identify_points = correlate(band, mean_fitler, output=None,
                                            mode='reflect', cval=0.0, origin=0) > 0
            else:
                identify_points = band > 0
            edges = []
            for ik, row in enumerate(identify_points):
                for jk, need_correction in enumerate(row):
                    kp = self.matrix[ik, jk]
                    if need_correction and kp not in self.degenerate_final:
                        continue
                    for direction in directions:
                        ikn, jkn = np.array([ik, jk]) + direction
                        if ikn >= self.matrix.shape[0] or jkn >= self.matrix.shape[1]:
                            continue
                        kneig = self.matrix[ikn, jkn]
                        if not identify_points[ikn, jkn]:
                            p = kp + (self.bands_final[kp, bn] - self.min_band)*self.nks
                            pn = kneig + (self.bands_final[kneig, bn] - self.min_band)*self.nks
                            edges.append([p, pn])
            edges = np.array(edges)
            self.GRAPH.add_edges_from(edges)
            self.correct_signalfinal_prev = np.copy(self.correct_signalfinal)
            self.correct_signalfinal[k_ot, bn_ot] = CORRECT-1

    def solve(self, step=0.1, min_tol=0):
        TOL = 0.5
        bands_final_flag = True
        self.bands_final_prev = np.copy(self.bands_final)
        self.best_bands_final = np.copy(self.bands_final)
        self.best_score = np.zeros(self.total_bands, dtype=float)
        self.final_score = np.zeros(self.total_bands, dtype=float)
        self.signal_final = np.zeros((self.nks, self.total_bands), dtype=int)
        self.correct_signalfinal_prev = np.full(self.signal_final.shape, -1, int)
        max_solved = 0

        while bands_final_flag and TOL >= min_tol:
            print()
            LOG.info(f'\n\n  Clustering samples for TOL: {TOL}')
            init_time = time.time()
            self.get_components(tol=TOL)
            LOG.info(f'{contatempo.tempo(init_time, time.time())}')

            LOG.info('  Calculating output')
            init_time = time.time()
            self.obtain_output()
            self.print_report(self.signal_final)
            LOG.info(f'{contatempo.tempo(init_time, time.time())}')

            LOG.info('  Validating result')
            init_time = time.time()
            self.correct_signal()
            self.print_report(self.correct_signalfinal)
            LOG.info(f'{contatempo.tempo(init_time, time.time())}')

            bands_final_flag = np.sum(np.abs(self.bands_final_prev - self.bands_final)) != 0
            self.bands_final_prev = np.copy(self.bands_final)

            solved = 0
            for bn, score in enumerate(self.final_score):
                best_score = self.best_score[bn]
                not_solved = np.sum(self.signal_final[:, bn] == NOT_SOLVED)
                if score >= best_score and not_solved == 0:
                    solved += 1
                else:
                    break
            if solved >= max_solved:
                self.best_bands_final = np.copy(self.bands_final)
                self.best_score = np.copy(self.final_score)
                self.best_signal_final = np.copy(self.signal_final)
            TOL -= step
        
        self.bands_final = np.copy(self.best_bands_final)
        self.final_score = np.copy(self.best_score)
        self.signal_final = np.copy(self.best_signal_final)
        
        self.print_report(self.signal_final)

class COMPONENT:
    '''
    This object contains the information that constructs a component,
    and also it has functions that are necessary to establish
    relations between components.
    '''
    def __init__(self, component: nx.Graph, kpoints_index, matrix):
        self.GRAPH = component
        self.N = self.GRAPH.number_of_nodes()
        self.m_shape = matrix.shape
        self.nks = self.m_shape[0]*self.m_shape[1]
        self.kpoints_index = np.array(kpoints_index)
        self.matrix = matrix
        self.positions_matrix = None
        self.nodes = np.array(self.GRAPH.nodes)

        self.__id__ = str(self.nodes[0])
        self.was_modified = False
        self.scores = {}

    def calculate_pointsMatrix(self):
        self.positions_matrix = np.zeros(self.m_shape, int)
        index_points = self.kpoints_index[self.nodes % self.nks]
        self.k_points = self.nodes % self.nks
        self.bands_number = dict(zip(self.nodes % self.nks,
                                     self.nodes//self.nks))
        self.positions_matrix[index_points[:, 0], index_points[:, 1]] = 1

    def get_bands(self):
        self.k_points = self.nodes % self.nks
        k_bands = self.nodes//self.nks
        self.bands_number = dict(zip(self.nodes % self.nks,
                                     self.nodes//self.nks))
        bands, counts = np.unique(k_bands, return_counts=True)
        self.bands = bands[np.argsort(counts)[::-1]]
        return k_bands

    def validate(self, component):
        if self.positions_matrix is None:
            self.calculate_pointsMatrix()
        N = np.sum(self.positions_matrix ^ component.positions_matrix)
        return (component.N <= self.nks - self.N and N == self.N+component.N)

    def join(self, component):
        del component.scores
        self.was_modified = True
        G = nx.Graph(self.GRAPH)
        G.add_nodes_from(component.GRAPH)
        self.GRAPH = G
        self.N = self.GRAPH.number_of_nodes()
        self.nodes = np.array(self.GRAPH.nodes)
        self.calculate_pointsMatrix()
        self.calc_boundary()

    def calc_boundary(self):
        '''
        Only the boundary nodes are necessary. Therefore this function computes
        these essential nodes and uses them to compare components.
        '''

        if self.positions_matrix is None:
            self.calculate_pointsMatrix()
        Gx = np.array([[-1, 0, 1]]*3)
        Gy = np.array([[-1, 0, 1]]*3).T
        Ax = correlate(self.positions_matrix, Gx, output=None,
                       mode='reflect', cval=0.0, origin=0)
        Ay = correlate(self.positions_matrix, Gy, output=None,
                       mode='reflect', cval=0.0, origin=0)
        self.boundary = np.sqrt(Ax**2+Ay**2)*self.positions_matrix
        self.boundary = (self.boundary > 0)
        self.k_edges = self.matrix[self.boundary]
        if len(self.k_edges) == 0:
            self.k_edges = self.nodes % self.nks

    def get_cluster_score(self, cluster, min_band, max_band,
                          neighbors, energies, connections, tol = 0.5):
        '''
        This function returns the similarity between components taking
        into account the dot product of all essential points and their
        energy value.
        INPUT
        cluster: It is a component with which the similarity is calculated.
        min_band: It is an integer that gives the minimum band used for clustering.
        max_band: It is an integer that gives the maximum band used for clustering.
        neighbors: It is an array that identifies the neighbors of each k point.
        energies: It is an array of the energy values inside a matrix.
        connections: It is an array with the dot product between k points
                     and his neighbors.
        OUTPUT
        score: It is a float that represents the similarity between components.
        '''
        def difference_energy(bn1, bn2, iK1, iK2, Ei = None):
            ik1, jk1 = iK1
            ik_n, jk_n = iK2
            Ei = energies[bn1, ik1, jk1] if Ei is None else Ei
            bands = np.arange(min_band, max_band+1)
            min_energy = np.min([np.abs(Ei-energies[bn, ik_n, jk_n])
                                    for bn in bands])
            delta_energy = np.abs(Ei-energies[bn2, ik_n, jk_n])
            return min_energy/delta_energy if delta_energy else 1
        
        def fit_energy(bn1, bn2, iK1, iK2):
            N = 4 # Number of points taking in account
            ik1, jk1 = iK1
            ik_n, jk_n = iK2
            I = np.full(N+1,ik1)
            J = np.full(N+1,jk1)
            flag = ik1 == ik_n
            i = I if flag else I + np.arange(0,N+1)*np.sign(ik1-ik_n)
            j = J if not flag else J + np.arange(0,N+1)*np.sign(jk1-jk_n)
            
            if not flag:
                i = i[i >= 0]
                i = i[i < self.m_shape[0]]
                j = np.full(len(i), jk1)
            else:
                j = j[j >= 0]
                j = j[j < self.m_shape[1]]
                i = np.full(len(j), ik1)


            ks = self.matrix[i, j]
            f = lambda e: e in self.k_points
            exist_ks = list(map(f, ks))
            ks = ks[exist_ks]
            if len(ks) <= 3:
                return difference_energy(bn1, bn2, iK1, iK2)
            aux_bands = np.array([self.bands_number[kp] for kp in ks])
            bands = aux_bands + min_band
            i = i[exist_ks]
            j = j[exist_ks]
            Es = energies[bands, i, j]
            X = i if jk1 == jk_n else j
            new_x = ik_n if jk1 == jk_n else jk_n

            pol = lambda x, a, b, c: a*x**2 + b*x + c
            popt, pcov = curve_fit(pol, X, Es)
            Enew = pol(new_x, *popt)
            Ei = energies[bn1, ik1, jk1]
            # LOG.debug(f'Actual Energy: {Ei} Energy founded: {Enew} for {bn1} with {len(i)} points.')
            return difference_energy(bn1, bn2, iK1, iK2, Ei = Enew)


        if cluster.was_modified:
            return self.scores[cluster.__id__]
        
        cluster.was_modified = False
        score = 0
        for k in self.k_edges:
            bn1 = self.bands_number[k] + min_band
            ik1, jk1 = self.kpoints_index[k]
            for i_neig, k_n in enumerate(neighbors[k]):
                if k_n == -1 or k_n not in cluster.k_edges:
                    continue
                ik_n, jk_n = self.kpoints_index[k_n]
                bn2 = cluster.bands_number[k_n]+min_band
                connection = connections[k, i_neig, bn1, bn2]
                energy_val = fit_energy(bn1, bn2, (ik1, jk1), (ik_n, jk_n))
                score += tol*connection + (1-tol)*energy_val
        score /= len(self.k_edges)*4
        self.scores[cluster.__id__] = score
        return score

    def save_boundary(self, filename):
        if not os.path.exists("boundaries/"):
            os.mkdir('boundaries/')
        with open('boundaries/'+filename+'.npy', 'wb') as f:
            np.save(f, self.boundary)
            np.save(f, self.positions_matrix)
