"""This module contains the algorithm behind band classification.
It determines which points belong to each band used for posterior calculations.
The algorithm uses machine learning techniques to cluster the data.
"""

import numpy as np
import networkx as nx
import os
from scipy.ndimage import correlate
from write_k_points import bands_numbers
from multiprocessing import Process
from log_libs import log
from loaddata import version

CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

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
        title = 'Making Vectors'
        LOG.percent_complete(0, 100, title=title)

        self.GRAPH = nx.Graph()
        self.min_band = min_band
        self.max_band = max_band
        nbnd = self.nbnd if max_band == -1 else max_band+1
        self.make_kpointsIndex()
        energies = self.make_BandsEnergy()
        LOG.percent_complete(20, 100, title=title)

        n_vectors = (nbnd-min_band)*self.nks
        ik = np.tile(self.kpoints_index[:, 0], nbnd-min_band)
        jk = np.tile(self.kpoints_index[:, 1], nbnd-min_band)
        bands = np.arange(min_band, nbnd)
        eigenvalues = self.eigenvalues[:, bands].T.reshape(n_vectors)
        self.vectors = np.stack([ik, jk, eigenvalues], axis=1)
        LOG.percent_complete(50, 100, title=title)

        self.GRAPH.add_nodes_from(np.arange(n_vectors))

        self.degenerados = []
        for i, v in enumerate(self.vectors):
            LOG.percent_complete(50 + int(i*50/len(self.vectors)), 100, title=title)
            degenerado = np.where(np.all(np.isclose(self.vectors[i+1:]-v, 0),
                                  axis=1))[0]
            if len(degenerado) > 0:
                self.degenerados += [[i, d+i+1] for d in degenerado]
                LOG.debug(f'Found degenerete point for {i}')
        if len(self.degenerados) > 0:
            LOG.info('Degenerate Points: ')
            for d in self.degenerados:
                LOG.info(f'\t{d}')

        self.ENERGIES = energies
        self.nbnd = nbnd-min_band

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

        def connection_component(i_start, j_end, tol=0.95):
            V = self.vectors[i_start:j_end]
            edges = []
            bands = np.repeat(np.arange(self.min_band, self.max_band+1), len(self.neighbors[0]))
            for i,_ in enumerate(V):
                i_ = i+i_start
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

            edges = np.array(edges)

            with open(f'temp/CONNECTIONS_{i_start}_{j_end}.npy', 'wb') as f:
                np.save(f, edges)
                LOG.debug(f'End Process_CONNECTIONS_{i_start}_{j_end}')

        edges = self.parallel_process('CONNECTIONS',
                                      connection_component,
                                      len(self.vectors), (tol))
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

    def clear_temp(self):
        os.rmdir("temp")

    def parallel_process(self, process_name, f, N, *args):
        if not os.path.exists("temp/"):
            os.mkdir('temp/')
        process = []
        LOG.debug(f'PARALLEL {process_name}')
        n = N//self.n_process
        for i_start in range(self.n_process):
            j_end = n*(i_start+1) if i_start < self.n_process-1\
                else n*(i_start+1) + n % self.n_process
            i_start = i_start*n
            LOG.debug(f'\t\tCreating Process {process_name}: {i_start}-{j_end}')
            process.append(Process(target=f, args=(i_start, j_end, *args)))

        print()
        join_process = []
        while len(process) > 0:
            p = process.pop(0)
            p.start()
            join_process.append(p)

        while len(join_process) > 0:
            p = join_process.pop(0)
            p.join()

        print()
        result = None
        for i_start in range(self.n_process):
            j_end = n*(i_start+1) if i_start < self.n_process-1\
                else n*(i_start+1) + n % self.n_process
            i_start = i_start*n
            with open(f'temp/{process_name}_{i_start}_{j_end}.npy', 'rb') as f:
                matrix = np.load(f)
            os.remove(f'temp/{process_name}_{i_start}_{j_end}.npy')
            LOG.debug(f'Joint {process_name}_{i_start}_{j_end} component')
            if len(matrix) == 0:
                continue
            result = np.concatenate((result,
                                     matrix)) if result is not None else matrix
        LOG.debug(f'\tEND {process_name}')

        return result

    def get_components(self):
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
                                                           self.connections)
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

        self.bands_final = np.full((self.nks, self.total_bands), -1, dtype=int)
        self.signal_final = np.zeros((self.nks, self.total_bands), dtype=int)
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

    def print_report(self):
        final_report = '\t====== FINAL REPORT ======\n\n'
        bands_report = []
        for bn in range(self.min_band, self.min_band+self.nbnd):
            band_result = self.signal_final[:, bn]
            report = [np.sum(band_result == s) for s in range(CORRECT+1)]
            bands_report.append(report)

            LOG.info(f'\n  New Band: {bn}\tnr falis: {report[0]}')
            bands_numbers(self.nkx, self.nky, self.bands_final[:, bn])

        bands_report = np.array(bands_report)
        final_report += '\n Signaling: how many events ' + \
                        'in each band signaled.\n'
        bands_header = '\n Band | '

        for signal in range(CORRECT+1):
            n_spaces = len(str(np.max(bands_report[:, signal])))-1
            bands_header += ' '*n_spaces+str(signal) + '   '

        final_report += bands_header + '\n'
        final_report += '-'*len(bands_header)

        for bn, report in enumerate(bands_report):
            bn += self.min_band
            final_report += f'\n {bn}{" "*(4-len(str(bn)))} |' + ' '
            for signal, value in enumerate(report):
                n_max = len(str(np.max(bands_report[:, signal])))
                n_spaces = n_max - len(str(value))
                final_report += ' '*n_spaces+str(value) + '   '

        LOG.info(final_report)
        self.final_report = final_report


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
                          neighbors, energies, connections):
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
                Ei = energies[bn1, ik1, jk1]
                bands = np.arange(min_band, max_band+1)
                min_energy = np.min([np.abs(Ei-energies[bn, ik_n, jk_n])
                                     for bn in bands])
                delta_energy = np.abs(Ei-energies[bn2, ik_n, jk_n])
                energy_val = min_energy/delta_energy if delta_energy else 1
                score += 0.5*connection + 0.5*energy_val
        score /= len(self.k_edges)*4
        self.scores[cluster.__id__] = score
        return score

    def save_boundary(self, filename):
        if not os.path.exists("boundaries/"):
            os.mkdir('boundaries/')
        with open('boundaries/'+filename+'.npy', 'wb') as f:
            np.save(f, self.boundary)
            np.save(f, self.positions_matrix)
