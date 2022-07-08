"""This module contains the algorithm behind band classification.
It determines which points belong to each band used for posterior calculations.
The algorithm uses machine learning techniques to cluster the data.
"""

import numpy as np
import networkx as nx
import os
from scipy.ndimage import correlate
from multiprocessing import Process


class MATERIAL:
    def __init__(self, nkx, nky, nbnd, nks, eigenvalues, connections, neighbors, n_process=1):
        self.nkx = nkx
        self.nky = nky
        self.nbnd = nbnd
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
        self.GRAPH = nx.Graph()
        self.min_band = min_band
        self.max_band = max_band
        nbnd = self.nbnd if max_band == -1 else max_band+1
        self.make_kpointsIndex()
        energies = self.make_BandsEnergy()

        n_vectors = (nbnd-min_band)*self.nks
        ik = np.tile(self.kpoints_index[:, 0], nbnd-min_band)
        jk = np.tile(self.kpoints_index[:, 1], nbnd-min_band)
        bands = np.arange(min_band, nbnd)
        eigenvalues = self.eigenvalues[:, bands].T.reshape(n_vectors)
        self.vectors = np.stack([ik, jk, eigenvalues], axis=1)

        self.GRAPH.add_nodes_from(np.arange(n_vectors))

        self.degenerados = []
        for i, v in enumerate(self.vectors):
            degenerado = np.where(np.all(np.isclose(self.vectors[i+1:]-v, 0),
                                  axis=1))[0]
            if len(degenerado) > 0:
                self.degenerados += [[i, d+i+1] for d in degenerado]
        if len(self.degenerados) > 0:
            print('Degenerate Points: ')
            for d in self.degenerados:
                print(f'\t{d}')

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
        def connection_component(i_start, j_end, tol=0.95):
            V = self.vectors[i_start:j_end]
            edges = []

            for i, P1 in enumerate(V):
                i_ = i+i_start
                bn1 = i_//self.nks + self.min_band  # bi
                k1 = i_ % self.nks
                for j, P2 in enumerate(self.vectors[i_start+i:]):
                    j_ = j+i_
                    bn2 = j_//self.nks + self.min_band  # bj
                    # 1.  Same k-point, different band.
                    if np.all(P1[:-1] == P2[:-1]) and bn1 != bn2:
                        '''
                        mi = mj
                        ni = nj
                        bi != bj
                        '''
                        continue
                    k2 = j_ % self.nks
                    if k2 in self.neighbors[k1]:
                        i_neig = np.where(self.neighbors[k1] == k2)[0]
                        connection = self.connections[k1, i_neig,
                                                      bn1, bn2]  # <i|j>
                        '''
                        for each first neighbor
                        Mij = 1 iff <i, j> ~ 1
                        '''
                        if connection > tol:
                            edges.append([i_, j_])

            edges = np.array(edges)

            with open(f'temp/CONNECTIONS_{i_start}_{j_end}.npy', 'wb') as f:
                np.save(f, edges)
                print(f'\t\tEnd Process_CONNECTIONS_{i_start}_{j_end}')

        edges = self.parallel_process('CONNECTIONS',
                                      connection_component,
                                      len(self.vectors), (tol))
        self.GRAPH.add_edges_from(edges)

        for d1, d2 in self.degenerados:
            if not self.find_path(d1, d2):
                continue
            N1 = np.array(self.get_neigs(d1))
            N2 = np.array(self.get_neigs(d2))
            if len(N1) == 0 or len(N2) == 0:
                continue
            print(f'Problem:\n\t{d1}: {N1}\n\t{d2}:{N2}')
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

            print(f'Solution:\n\t{d1}: {N1_}\n\t{d2}:{N2_}')
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
        print(f'\n\tPARALLEL {process_name}')
        n = N//self.n_process
        for i_start in range(self.n_process):
            j_end = n*(i_start+1) if i_start < self.n_process-1\
                else n*(i_start+1) + n % self.n_process
            i_start = i_start*n
            print(f'\t\tCreating Process {process_name}: {i_start}-{j_end}')
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
            print(f'\t\tJoint {process_name}_{i_start}_{j_end} component')
            result = np.concatenate((result,
                                     matrix)) if result is not None else matrix
        print(f'\tEND {process_name}')

        return result

    def get_components(self):
        print('\n\nNumber of Components: ', end='')
        print(f'{nx.number_connected_components(self.GRAPH)}')
        self.components = [COMPONENT(self.GRAPH.subgraph(c),
                                     self.kpoints_index,
                                     self.matrix)
                           for c in nx.connected_components(self.GRAPH)]
        index_sorted = np.argsort([component.N
                                   for component in self.components])[::-1]
        self.solved = []
        clusters = []
        samples = []
        for i_, i in enumerate(index_sorted):
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
        print(f'    Phase 1: {len(self.solved)}/{self.nbnd} Solved')
        print(f'    Initial clusters: {len(clusters)} Samples: {len(samples)}')

        tol = 1
        while len(samples) > 0:
            count = np.array([0, len(samples)])
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
            arg_evaluate = np.argsort(evaluate_samples[:, 0])[::-1]
            flag = True

            samples = [samples[i_arg] for i_arg in arg_evaluate]
            new_samples = []
            i_arg = 0
            while len(samples) > 0:
                sample = samples.pop(0)
                score, bn = evaluate_samples[arg_evaluate[i_arg]]
                bn = int(bn)
                if score < tol:
                    new_samples.append(sample)
                    new_samples += samples
                    break
                tol = score if score > tol else tol
                i_arg += 1
                flag = False
                count[0] += 1
                print(f'{tol}-{count[0]}/{count[1]} Sample corrected: {score}')
                clusters[bn].join(sample)
                if clusters[bn].N == self.nks:
                    print('Cluster Solved')
                    self.solved.append(clusters.pop(bn))

            if flag:
                tol -= 0.05
                flag = True
                count = np.array([0, len(samples)])

            samples = new_samples

        print(f'    Phase 2: {len(self.solved)}/{self.nbnd} Solved')

        if len(self.solved)/self.nbnd < 1:
            print(f'    New clusnters: {len(clusters)}', end='')
            print(f'    New Samples: {len(new_samples)}')

        labels = np.empty(self.nks*self.nbnd, int)
        count = 0
        for solved in self.solved:
            labels[solved.nodes] = count
            count += 1

        for cluster in clusters:
            cluster.save_boundary(f'cluster_{count}')
            labels[cluster.nodes] = count
            count += 1

        for sample in new_samples:
            sample.save_boundary(f'sample_{count}')
            labels[sample.nodes] = count
            count += 1
        
        self.clusters = clusters
        self.samples = new_samples

        return labels

    def obtain_output(self):
        self.bands_final = np.full((self.nks, self.nbnd), -1, dtype=int)
        solved_bands = []
        for solved in self.solved:
            bands = solved.get_bands()
            bn = solved.bands[0]
            solved.bands = solved.bands[1:]
            while bn in solved_bands:
                bn = solved.bands[0]
                solved.bands = solved.bands[1:]
            solved_bands.append(bn)
            self.bands_final[solved.nodes, bn] = bands


class COMPONENT:
    def __init__(self, component: nx.Graph, kpoints_index, matrix):
        self.GRAPH = component
        self.N = self.GRAPH.number_of_nodes()
        self.m_shape = matrix.shape
        self.nks = self.m_shape[0]*self.m_shape[1]
        self.kpoints_index = np.array(kpoints_index)
        self.matrix = matrix
        self.positions_matrix = None
        self.nodes = np.array(self.GRAPH.nodes)

    def calculate_pointsMatrix(self):
        self.positions_matrix = np.zeros(self.m_shape, int)
        index_points = self.kpoints_index[self.nodes % self.nks]
        self.bands_number = dict(zip(self.nodes % self.nks,
                                     self.nodes//self.nks))
        self.positions_matrix[index_points[:, 0], index_points[:, 1]] = 1
    
    def get_bands(self):
        k_bands = self.nodes//self.nks
        bands, counts = np.unique(k_bands, return_counts=True)
        self.bands = bands[np.argsort(counts)]
        return k_bands

    def validate(self, component):
        if self.positions_matrix is None:
            self.calculate_pointsMatrix()
        N = np.sum(self.positions_matrix ^ component.positions_matrix)
        return (component.N <= self.nks - self.N and N == self.N+component.N)

    def join(self, component):
        G = nx.Graph(self.GRAPH)
        G.add_nodes_from(component.GRAPH)
        self.GRAPH = G
        self.N = self.GRAPH.number_of_nodes()
        self.nodes = np.array(self.GRAPH.nodes)
        self.calculate_pointsMatrix()
        self.calc_boundary()

    def calc_boundary(self):
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
                # print(f'    {bn1}-{bn2}: {connection} ', end='')
                # print(f'{energy_val} = {0.3*connection + 0.7*energy_val}')
                score += 0.5*connection + 0.5*energy_val
        score /= len(self.k_edges)*4
        return score

    def save_boundary(self, filename):
        if not os.path.exists("boundaries/"):
            os.mkdir('boundaries/')
        with open('boundaries/'+filename+'.npy', 'wb') as f:
            np.save(f, self.boundary)
            np.save(f, self.positions_matrix)
