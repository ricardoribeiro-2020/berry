"""
This file defines the Material class, which encapsulates the core data structures and logic necessary to
process, analyze, and cluster the bands. It supports 1D, 2D, and 3D data, and constructs graphs based on 
energy proximity and degeneracies to identify components across k-point meshes and clustering them using
the dot product information.

Classes:
    - Material: Core class that handles eigenvalue and connection data and builds clustering logic.

Last Modified: [2025-05-06]
"""


import numba
import textwrap
import numpy as np
import networkx as nx

import numpy.typing as npt

from typing import Tuple, Union, List
from numba import prange
from tqdm import tqdm
from datetime import datetime

from berry._subroutines.clustering_utils import Component, compute_edges, solver_algorithm
from berry import log




import psutil, os
import sys

# Try importing pympler; if unavailable, we do shallow sizing
try:
    from pympler import asizeof
    _USE_PYMP = True
except ImportError:
    _USE_PYMP = False


def attribute_memory_usage(obj):
    """
    Given an object instance, return a dict {attribute_name: memory_in_bytes}.
    - If pympler is available, uses asizeof.asizeof(attr) (deep size).
    - Otherwise, for np.ndarray it uses arr.nbytes + overhead; else sys.getsizeof(attr).
    Works for classes with __dict__ or with __slots__.
    """
    mem_usage = {}
    seen_names = set()

    # 1) Handle standard attributes in __dict__
    if hasattr(obj, '__dict__'):
        for name, val in vars(obj).items():
            if _USE_PYMP:
                size = asizeof.asizeof(val)
            else:
                if isinstance(val, np.ndarray):
                    size = val.nbytes + sys.getsizeof(val)
                else:
                    size = sys.getsizeof(val)
            mem_usage[name] = size
            seen_names.add(name)

    # 2) Handle __slots__ (if defined and not in __dict__)
    cls = obj.__class__
    slots = getattr(cls, '__slots__', ())
    if slots:
        # __slots__ might be a single string or iterable
        slot_list = (slots,) if isinstance(slots, str) else slots
        for name in slot_list:
            if name in seen_names:
                continue
            try:
                val = getattr(obj, name)
            except AttributeError:
                continue
            if _USE_PYMP:
                size = asizeof.asizeof(val)
            else:
                if isinstance(val, np.ndarray):
                    size = val.nbytes + sys.getsizeof(val)
                else:
                    size = sys.getsizeof(val)
            mem_usage[name] = size

    return mem_usage


###########################################################################
# Type Definition
###########################################################################
Kpoint = int
Connection = float
Band = int
###########################################################################
# Constant Definition
###########################################################################
CORRECT = 2
DEGENERATE = 1
MISTAKE = 0

DEGENERATE_TOLERANCE = 0.71
MINIMUM_TOLERANCE = 0.5

EVALUATE_RESULT_HELP = '''
    ---------------------- Report considering dot-product information ----------------------------
            C -> Mean dot-product |<i|j>| of each k-point
            ----------------------------------------------------------------------------
            Value | Abbreviation |                Description
            ----------------------------------------------------------------------------
            0     |     MIS      |      MISTAKE               C < 0.5
            1     |     DEG      |      DEGENERATE            It is a degenerate point
            2     |     COR      |      CORRECT               C > 0.7
    -----------------------------------------------------------------------------------------------
'''
EVALUATE_RESULT_HEADER = ['MIS', 'DEG', 'COR']

def evaluate_result(values: Union[List[Connection], npt.NDArray]) -> int:
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
            1     :  MISTAKE               C < 0.5
            2     :  DEGENERATE            It is a degenerate point.
            5     :  CORRECT               C > 0.7
    '''

    values = np.array(values)
    value = np.mean(values) # Mean conection of each k point

    if value > DEGENERATE_TOLERANCE:
        return CORRECT

    if value >= MINIMUM_TOLERANCE and value <= DEGENERATE_TOLERANCE:
        return CORRECT

    return MISTAKE




class Material:
    """
    The Material class encapsulates the data strucures and logic of the clustering
    algorithm

    Attributes:
        dimensions (int): Number of spatial dimensions (1, 2, or 3).
        nk_i (list[int]): Number of k-points in each direction.
        number_of_bands (int): Total number of bands after applying min_band offset.
        nks (int): Total number of k-points.
        eigenvalues (np.ndarray): Eigenvalues for all bands at each k-point.
        connections (np.ndarray): Dot-product information for the bands.
        neighbors (np.ndarray): Neighboring k-points for each k-point.
        delta_k (float): Distance between neighboring k-points in reciprocal space.
        min_band (int): Starting band index to analyze.
        max_band (int): Final band index to analyze.
        n_process (int): Number of processes (used for parallelism with Numba).
        logger (log): Logger instance for debug and info messages.
        verbose (bool): If True, enables tqdm progress bar and more logging output.

    Main Methods:
        - make_kpoints_index: Generates index mappings for the k-point mesh.
        - make_bands: Reshapes eigenvalues into mesh format.
        - solver: Executes the main clustering and degeneracy analysis algorithm.
    """
    def __init__(self,
                 dimensions: int,
                 nk_i: List[int],
                 number_of_bands: int,
                 nks: int,
                 eigenvalues: npt.NDArray,
                 connections: npt.NDArray,
                 neighbors: npt.NDArray,
                 logger: log,
                 min_band: int,
                 max_band: int,
                 delta_k: float,
                 n_process: int=1,
                 verbose: bool = False) -> None:
        
        numba.set_num_threads(n_process)

        self.number_of_bands = number_of_bands - min_band
        self.bands = np.arange(0, self.number_of_bands)
        self.min_band = min_band
        self.max_band = max_band    # m.final_band
        self.dimensions = dimensions
        self.nks = nks


        self.eigenvalues = eigenvalues[:, min_band:]
        self.connections = connections

        self.nkx, self.nky, self.nkz = nk_i
        self.neighbors = neighbors
        self.delta_k = delta_k
        self.number_neighbors = self.dimensions * 2

        self.n_process = n_process
        self.logger = logger

        self.verbose = verbose

        
    def make_kpoints_index(self) -> None:
        '''
        self.mesh_kpoints_idex: np.ndarray
            The index of the kpoints in the meshgrid format.
            The shape of the array is (nkx, nky, nkz) for 3D, (nkx, nky) for 2D, and (nkx) for 1D.
        self.array_kpoints_index: np.ndarray
            The index of the kpoints in the array format.
            The shape of the array is (nks, 3) for 3D, (nks, 2) for 2D, and (nks) for 1D.
        '''
        if self.dimensions == 1:
            self.mesh_kpoints_index = np.arange(self.nkx, dtype=int)
            self.array_kpoints_index = np.arange(self.nks, dtype=int)
            return
    
        if self.dimensions == 2:
            My, Mx = np.meshgrid(np.arange(self.nky, dtype=int), np.arange(self.nkx,dtype=int))
            self.mesh_kpoints_index = My * self.nkx + Mx
            counts = np.arange(self.nks, dtype=int)
            self.array_kpoints_index = np.stack([counts % self.nkx, counts//self.nkx],
                                            axis=1)
            return
        
        self.mesh_kpoints_index = np.empty((self.nkx, self.nky, self.nkz), int)
        self.array_kpoints_index = np.empty((self.nks, 3), int)
        count = -1
        for k in prange(self.nkz):
            for j in prange(self.nky):
                for i in prange(self.nkx):
                    count += 1
                    self.mesh_kpoints_index[i, j, k] = count
                    self.array_kpoints_index[count] = [i, j, k]
        
    def make_bands(self) -> None:
        '''
        self.mesh_energies: np.ndarray
            The eigenvalues in the meshgrid format.
            The shape of the array is (number_of_bands, nkx, nky, nkz) for 3D, (number_of_bands, nkx, nky) for 2D, and (number_of_bands, nkx) for 1D.
            
        '''

        bands_final, _ = np.meshgrid(self.bands,
                                     np.arange(0, self.nks, dtype=int))

        self.bands_final = np.full((self.nks, self.number_of_bands), -1, dtype=int)
        
        if self.dimensions == 1:
            self.mesh_energies = np.empty((self.number_of_bands, self.nks), float)
            for bn in prange(self.number_of_bands):
                self.mesh_energies[bn] = self.eigenvalues[:, bn]
            return
        
        if self.dimensions == 2:
            self.mesh_energies = np.empty((self.number_of_bands, self.nkx, self.nky), float)
            for bn in prange(self.number_of_bands):
                for j in prange(self.nky):
                    for i in prange(self.nkx):
                        count = i + j * self.nkx
                        self.mesh_energies[bn, i, j] = self.eigenvalues[count,
                                                        bands_final[count, bn]]
            return
        
        self.mesh_energies = np.empty((self.number_of_bands, self.nkx, self.nky, self.nkz), float)
        for bn in prange(self.number_of_bands):
            for k in prange(self.nkz):
                for j in prange(self.nky):
                    for i in prange(self.nkx):
                        count = i + j * self.nkx + k * self.nkx * self.nky
                        self.mesh_energies[bn, i, j, k] = self.eigenvalues[count,
                                                                    bands_final[count, bn]]
                        
    def make_vectors(self) -> None:
        '''
        self.number_of_vectors: int
            The number of vectors in the array format.
        self.vectors: np.ndarray
            The vectors in the array format.
            The shape of the array is (number_of_vectors, 4) for 3D, (number_of_vectors, 3) for 2D, and (number_of_vectors, 2) for 1D.

            Each row of the array is [ik, jk, kk, eigenvalue].
        
        
        This function is not being used, it was an improvement for previous versions but it is not present in the current implementation
        It remains here If required for future improvements.
        '''
        self.number_of_vectors = self.number_of_bands * self.nks

        ik = np.tile(self.array_kpoints_index[:, 0], self.number_of_bands) if self.dimensions > 1 else np.tile(self.array_kpoints_index, self.number_of_bands)

        stack_aux = [ik]

        if self.dimensions >= 2:
            jk = np.tile(self.array_kpoints_index[:, 1], self.number_of_bands)
            stack_aux.append(jk)

        if self.dimensions == 3:
            kk = np.tile(self.array_kpoints_index[:, 2], self.number_of_bands)
            stack_aux.append(kk)

        eigenvalues = self.eigenvalues[:, self.bands].T.reshape(self.number_of_vectors)
        stack_aux.append(eigenvalues)

        self.vectors = np.stack(stack_aux, axis=1)

    def solver(self, initial_tol=0.99):
        """
        Perform the full band structure analysis, identifying energy-connected components
        and degenerate regions across a k-point mesh.

        This method constructs graphs representing energy connectivity between k-points 
        within each band. It identifies band connectivity through energy thresholds and 
        detects degenerate points (where energy differences are below a specified 
        threshold). Then, it iteratively solves for clusters (connected components) using
        dot-product information.

        Parameters
        ----------
        initial_tol : float, optional
            Initial tolerance used for determining whether k-points from different bands 
            are considered connected (default is 0.99). This influences the energy 
            similarity threshold when building edges between bands.
        """
            
        # -- Values that can be optimized --
        delta_E_tol = 3 * self.delta_k
        delta_E_deg_tol = 0.25 * self.delta_k
        self.logger.info(f"Initial tolerance: {initial_tol}")

        # -----------------------------------
        
        self.make_kpoints_index()
        self.make_bands()

        self.logger.info(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting clustering algorithm...")
        self.logger.info(f"Memory mesh_kpoints_index: {self.mesh_kpoints_index.nbytes / (1024 ** 2):.2f} MB")
        self.logger.info(f"Memory array_kpoints_index: {self.array_kpoints_index.nbytes / (1024 ** 2):.2f} MB")
        self.logger.info(f"Memory mesh_energies: {self.mesh_energies.nbytes / (1024 ** 2):.2f} MB")

        self.logger.info(f"Memory PID: {os.getpid()} - {psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2):.2f} MB")

        Component.set_global_data(self.dimensions, self.nks, self.mesh_kpoints_index, self.array_kpoints_index, self.mesh_energies)
        
        graphs = []
        components = []
        
        not_energy_points = []
        degenerate_points = []

        mask_energies = [np.ones(self.mesh_kpoints_index.shape, dtype=bool) for _ in range(self.number_of_bands)]
        mask_degenerates = [np.zeros(self.mesh_kpoints_index.shape, dtype=bool) for _ in range(self.number_of_bands)]
        max_band_energies = []

        self.degenerates = []

        for bn_i, band_i in enumerate(self.mesh_energies[:-1]):
            for bn_j, band_j in enumerate(self.mesh_energies[bn_i + 1:]):
                bn_j += bn_i + 1
                diff_energy = band_j - band_i
                mask_energy = np.abs(diff_energy) > delta_E_tol
                mask_degenerate = np.abs(diff_energy) < delta_E_deg_tol

                if np.all(mask_energy):
                    break
                
                axis = tuple(range(self.dimensions))
                mask_energy = mask_energy * 1 - mask_energy * np.sign(np.abs(np.sum(np.array(np.gradient(mask_energy * 1, axis=axis))**2, axis=0)))

                k_deg = self.mesh_kpoints_index[mask_degenerate]
                for k in k_deg:
                    self.degenerates.append([k, bn_i, bn_j])


                mask_energies[bn_i] = np.logical_and(mask_energies[bn_i], mask_energy)
                mask_energies[bn_j] = np.logical_and(mask_energies[bn_j], mask_energies[bn_i])

                mask_degenerates[bn_i] = np.logical_or(mask_degenerates[bn_i], mask_degenerate)
                mask_degenerates[bn_j] = np.logical_or(mask_degenerates[bn_j], mask_degenerates[bn_i])

            max_band_energies.append(bn_j - 1 if bn_j < self.bands[-1] else bn_j)

        max_band_energies.append(self.bands[-1])

        max_band_energies_corrected = []
        for bn_i, max_band_i in enumerate(max_band_energies):
            if max_band_i == bn_i:
                max_band_energies_corrected.append(max_band_i)
                continue
            bn_j = max_band_i
            max_band_j = max_band_energies[bn_j]
            while bn_j != max_band_j:
                bn_j = max_band_j
                max_band_j = max_band_energies[bn_j]

            max_band_energies_corrected.append(bn_j)
        
        max_band_energies = max_band_energies_corrected
        sequence = []
        prev_max_band = -1
        for bn_i, max_band_i in enumerate(max_band_energies):
            if prev_max_band < bn_i:
                sequence.append(f"{bn_i + self.min_band} - {max_band_i + self.min_band}" if max_band_i != bn_i else f"{bn_i + self.min_band}")
                prev_max_band = max_band_i

            for bn_j in self.bands[bn_i + 1: max_band_i + 1]:
                mask_energies[bn_j] = np.logical_and(mask_energies[bn_j], mask_energies[bn_i])
                mask_degenerates[bn_j] = np.logical_or(mask_degenerates[bn_j], mask_degenerates[bn_i])

        self.logger.info(textwrap.fill(
            "Band energy progression:\n  " + ', '.join(sequence),
            width=100,
            subsequent_indent='  '
        ))

        self.logger.info(f"Memory PID: {os.getpid()} - {psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3):.2f} GB")

 
        for bn_i in self.bands:
            self.logger.debug("Degenerate points: ", np.sum(mask_degenerates[bn_i]))
            k_points = self.mesh_kpoints_index[mask_energies[bn_i]]

            if np.sum(mask_degenerates[bn_i]) == self.nks:
                mask_degenerates[bn_i] = np.zeros_like(mask_degenerates[bn_i], dtype=bool)

            not_k_points = self.mesh_kpoints_index[~mask_energies[bn_i] ^ mask_degenerates[bn_i]]
            deg_k_points = self.mesh_kpoints_index[mask_degenerates[bn_i]]
            not_energy_points.append(not_k_points)
            degenerate_points.append(deg_k_points)

            bn_i_graph = nx.Graph()
            bn_i_graph.add_nodes_from([(k, bn_i) for k in k_points])

            edges = []
            for k in k_points.flatten():
                for k_neig in self.neighbors[k]:
                    if k_neig < k or k_neig not in k_points:
                        continue
                    edges.append([(k, bn_i), (k_neig, bn_i)])
         
            bn_i_graph.add_edges_from(edges)

            graphs.append(bn_i_graph)
        

        for bn_i in tqdm(self.bands, desc="Identifying Energy Components", unit="Components", disable=not self.verbose):
            not_k_points = not_energy_points[bn_i]

            if graphs[bn_i].number_of_nodes() == self.nks:
                components.append([Component(graphs[bn_i])])
                continue
            
            vectors_bn_i = np.column_stack([not_k_points, np.full((len(not_k_points), 1), bn_i)])

            do_parallel = self.n_process > 1 and len(vectors_bn_i) > self.nks

            edges_aux = compute_edges(vectors_bn_i, self.neighbors, self.connections, np.arange(bn_i, max_band_energies[bn_i] + 1), tol_to_be_an_edge=initial_tol, n_process=self.n_process, verbose=False, parallel=do_parallel)

            edges = []

            for (k, b), v, weight in edges_aux:
                k_n = v % self.nks
                b_n = v // self.nks

                if b_n > bn_i:
                    if k_n not in not_energy_points[b_n] or k_n in degenerate_points[b_n]:
                        continue
                    not_energy_points[b_n] = np.delete(not_energy_points[b_n], np.where(not_energy_points[b_n] == k_n)[0])
                else:
                    if k_n not in not_energy_points[bn_i] or k_n in degenerate_points[bn_i]:
                        continue
                
                edges.append([(k, b), (k_n, b_n)])

            graphs[bn_i].add_nodes_from([(k, bn_i) for k in not_k_points])
            graphs[bn_i].add_edges_from(edges)

            band_components = [Component(graphs[bn_i].subgraph(c)) for c in nx.connected_components(graphs[bn_i])]

            components.append(band_components)

        # Free memory
        del graphs

        degenerate_components = []

        memory_of_component = []
        for components_bn_i in components:
            for c in components_bn_i:
                memory_of_component.append(sys.getsizeof(c))

        self.logger.info(f"Memory PID: {os.getpid()} - {psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3):.2f} GB")

        for bn_i, degenerate_points_bn_i in enumerate(degenerate_points):
            if len(degenerate_points_bn_i) == 0:
                degenerate_components.append([])
                continue

            degenerate_graph = nx.Graph()
            degenerate_graph.add_nodes_from([(k, bn_i) for k in degenerate_points_bn_i])
    
            degenerate_components_bn_i = [Component(degenerate_graph.subgraph(c)) for c in nx.connected_components(degenerate_graph)]
            degenerate_components.append(degenerate_components_bn_i)

        for bn_i, connected_components_bn_i in enumerate(components):
            number_nodes = [c.number_of_nodes for c in connected_components_bn_i]
            self.logger.debug(f"Number of connected components {bn_i}: {len(connected_components_bn_i)} total: {np.sum(number_nodes)}")
        
        self.logger.info(f"Memory PID Before Start: {os.getpid()} - {psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3):.2f} GB")
        clusters, iterable_bands = solver_algorithm(components, degenerate_components, degenerate_points, self.connections, self.neighbors, self.mesh_kpoints_index, self.nks, max_band_energies, mask_degenerates, self.bands, n_process=self.n_process, verbose=self.verbose, logger=self.logger, tolerance=initial_tol)
        
        order_bands = [c.min_energy() for c in clusters]
        order_bands = np.argsort(order_bands)

        self.cluster_bands = [clusters[i] for i in order_bands]

        self.obtain_output()
        self.print_report(self.signal_final, "Intermediate Report")

        self.prev_signal = self.signal_final

        components = []
        degenerate_components = [[] for _ in self.bands]

        mesh_with_mistakes = [np.zeros_like(self.mesh_kpoints_index) for _ in self.bands]

        for bn_i, band in enumerate(self.cluster_bands):
            mask_degenerates[bn_i] = np.zeros_like(mask_degenerates[bn_i])
            
            mistakes = np.zeros_like(self.mesh_kpoints_index)
            marked_indexes = self.array_kpoints_index[self.signal_final[:, bn_i] == MISTAKE]
            if len(marked_indexes.shape) > 1:
                marked_indexes = tuple(marked_indexes.T)
            
            mistakes[marked_indexes] = 1

            mesh_with_mistakes[bn_i] = np.logical_or(mesh_with_mistakes[bn_i], mistakes)
            for bn_j in self.bands[bn_i + 1: max_band_energies[bn_i] + 1]:
                mesh_with_mistakes[bn_j] = np.logical_or(mesh_with_mistakes[bn_j], mesh_with_mistakes[bn_i])

        for bn_i, band in enumerate(self.cluster_bands):
            edges = []
            
            for bn_j in self.bands[bn_i + 1: max_band_energies[bn_i] + 1]:
                mesh_with_mistakes[bn_i] = np.logical_or(mesh_with_mistakes[bn_j], mesh_with_mistakes[bn_i])

            if np.sum(mesh_with_mistakes[bn_i]) == 0:
                components.append([band])
                continue

            for k, b in band.nodes:
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    if k_neig < k:
                        continue
                    b_neigh = band.nodes[k_neig, 1]
                    edges.append([(k, b), (k_neig, b_neigh)])
            
            band.graph.add_edges_from(edges)

            components.append(band.remove_points(mesh_with_mistakes[bn_i]))
            del band

        clusters, iterable_bands = solver_algorithm(components, degenerate_components, degenerate_points, self.connections, self.neighbors, self.mesh_kpoints_index, self.nks, max_band_energies, mask_degenerates, self.bands, n_process=self.n_process, verbose=self.verbose, logger=self.logger, energies=self.eigenvalues, tolerance=initial_tol)

        order_bands = [c.min_energy() for c in clusters]
        order_bands = np.argsort(order_bands)

        self.cluster_bands = [clusters[i] for i in order_bands]

        self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Clustering algorithm finished.")

        self.obtain_output()

        self._number_of_preserved_mistakes = np.sum(self.signal_final == MISTAKE)
        self.iterable_bands = iterable_bands

        self.signal_final[self.signal_final == MISTAKE] = CORRECT


    def obtain_output(self) -> None:
        '''
        This function prepares the final data structures
        that are essential to other programs.
        '''

        ###########################################################################
        # Obtain the resultant bands' attribution and the k-point's signal.
        ###########################################################################

        self.logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Obtaining the output...")

        self.signal_final = np.zeros((self.nks, self.number_of_bands), dtype=int)   # The k-point's signal

        for bn_i, band in enumerate(self.cluster_bands):
            
            band.nodes = band.nodes[band.nodes[:, 0].argsort()]                   # Sort the k-points inside the band

            for k, bn1 in band.nodes:
                self.bands_final[k, bn_i] = bn1

                connections = []                                                    # The array that store the dot-product with the k-point's neighbors
                for i_neig, k_neig in enumerate(self.neighbors[k]):
                    # Obtain the dot-product with each neighbor
                    if k_neig == -1:
                        continue
                    
                    bn2 = band.nodes[k_neig, 1]
                    connections.append(self.connections[k, i_neig, bn1, bn2])       # <k, k neighbor>

                self.signal_final[k, bn_i] = evaluate_result(connections)             # Computes the k-point's signal


        ###########################################################################
        # Scoring the result.
        # Signaling and storage of degenerate k-points.
        ###########################################################################

        self.degenerate_final = []                                                 # Final degenerates k-points
        for k, bn1, bn2 in tqdm(self.degenerates, desc="Scoring the result", unit="k-points", disable=not self.verbose):
            # Signaling the numerically degenerate points Ei ~ Ej

            Bk1 = self.bands_final[k] == bn1                               # Find in which  band the k-point was attributed
            Bk2 = self.bands_final[k] == bn2                               # Find in which  band the k-point was attributed
            bn1 = np.argmax(Bk1) if np.sum(Bk1) != 0 else bn1               # Final band
            bn2 = np.argmax(Bk2) if np.sum(Bk2) != 0 else bn2               # Final band

            k_neigs = self.neighbors[k]                                    # k-point's neighbors
            flag_neig = k_neigs != -1                                       # Flag to obtain the allowed neighbors
            i_neigs = np.arange(self.number_neighbors)[flag_neig]                         # Neighbors' index
            k_neigs = k_neigs[flag_neig]                                    # Allowed neighbors

            if len(k_neigs) == 0:
                # If there are no neighbors the k-point's score is 0
                continue
            bn1 = self.bands_final[k, bn1]                                 # K-point's band
            bn2 = self.bands_final[k, bn2]                                 # K-point's band
            dps = self.connections[k, i_neigs, bn1, bn2]        # Dot-product between the k-point and their neighbors
            if np.any(np.logical_and(dps >= 0.5, dps <= DEGENERATE_TOLERANCE)):
                # It is considered degenerate if the k-point has some 
                # neighbor's dot-product between 0.5 and DEGENERATE_TOLERANCE
                self.degenerate_final.append([k, bn1, bn2])                # Storage the degenerate k-point
                self.signal_final[k, bn1] = DEGENERATE                         # Signal k_point as Degenerate
                self.signal_final[k, bn2] = DEGENERATE                         # Signal k_point as Degenerate

        self.k_basis_rotation = []
        k_basis_rotation = []           # Storage pairs of points that are degenerates by dot product 0.5 < <i|j> < DEGENERATE_TOLERANCE
        self.final_score = np.zeros(self.number_of_bands)                            # The bands' score
        for bn in self.bands:
            # Search these degenerate points on each band
            # Calculating the score of the result
            score = 0
            for k in range(self.nks):
                kneigs = self.neighbors[k]                                                  # k-point's neighbors
                flag_neig = kneigs != -1                                                    # Flag to obtain the allowed neighbors 
                i_neigs = np.arange(self.number_neighbors)[flag_neig]                                     # Neighbors' index
                kneigs = kneigs[flag_neig]                                                  # Allowed neighbors
                
                if len(kneigs) == 0:
                    # If there are no neighbors the k-point's score is 0
                    continue
                bn_k = self.bands_final[k, bn]                                              # K-point's band
                bn_neighs = self.bands_final[kneigs, bn]                                    # Neighbors' bands
                k = np.repeat(k, len(kneigs))                                               # Array with the same k-point
                bn_k = np.repeat(bn_k, len(kneigs))                                         # Array with the same K-point's band
                dps = self.connections[k, i_neigs, bn_k, bn_neighs]                         # The array with the dot-product between the k-point and their neighbors
                if np.any(np.logical_and(dps >= 0.5, dps <= DEGENERATE_TOLERANCE)):
                    # It is considered degenerate if the k-point has some 
                    # neighbor's dot-product between 0.5 and DEGENERATE_TOLERANCE
                    dps_deg = self.connections[k, i_neigs, bn_k]                                # All k-point dot-products
                    k = k[0]                                                                    # K-point
                    i_deg, bn_deg = np.where(np.logical_and(dps_deg >= 0.5, dps_deg <= 0.8))    # Find where the k-point dot-product is considered degenerate
                    k_deg = self.neighbors[k][i_deg + np.min(i_neigs)]                            # Identify the degenerate neighbors
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
                k_deg = np.array(k_deg)
                k_deg_ = np.array(k_deg_)
                if k != k_ or k_deg.shape != k_deg_.shape or not np.all(k_deg == k_deg_):
                    # The k_ point is not the k's pair
                    continue
                if not np.all([np.all(np.isin(bns, bns_deg_[j])) for j, bns in enumerate(bns_deg)]):
                    # If they do not belong to the same bands, The k_ point is not the k's pair
                    continue
                degenerates.append([k, bn, bn_])


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

                if not np.all(np.isin([bn, bn_], [bn0, bn1])):
                    # If they do not belong to the same bands, the analysis can not be done
                    continue

                ik =  self.array_kpoints_index[k, 0] if self.dimensions > 1 else self.array_kpoints_index[k]       # Obtain the matrix indices of k-space projection (k-point)
                ik_ = self.array_kpoints_index[k_, 0] if self.dimensions > 1 else self.array_kpoints_index[k_]     # Obtain the matrix indices of k-space projection (k_-point)

                idif = np.abs(ik - ik_)                                                                 # Manhattan distance i axes

                if idif > 1:
                    # If the total Manhattan distance is more than 2,
                    # the analysis can not be done
                    continue

                if self.dimensions >= 2:
                    jk = self.array_kpoints_index[k, 1]                                                       # Obtain the matrix indices of k-space projection (k-point)
                    jk_ = self.array_kpoints_index[k_, 1]                                                     # Obtain the matrix indices of k-space projection (k_-point)
                    jdif = np.abs(jk - jk_)                                                             # Manhattan distance j axes
                    if jdif > 1:
                        # If the total Manhattan distance is more than 2,
                        # the analysis can not be done
                        continue
                
                if self.dimensions == 3:
                    kk = self.array_kpoints_index[k, 2]                                                       # Obtain the matrix indices of k-space projection (k-point)
                    kk_ = self.array_kpoints_index[k_, 2]                                                     # Obtain the matrix indices of k-space projection (k_-point)
                    kdif = np.abs(kk - kk_)                                                             # Manhattan distance k axes
                    if kdif > 1:
                        # If the total Manhattan distance is more than 2,
                        # the analysis can not be done
                        continue

                analyzed.append(j+i+1)                                                      # The point k_ was analyzed
                same_group.append([k_, bn0, bn1])                                           # The k_-point belongs to the same group of k

            same_group = np.array(same_group)
            ks = same_group[:, 0]                                                           # K-points
            neighs = self.neighbors[ks]                                                     # K-points' neighbors
            points = [np.sum(neighs == k) for k in ks]                                      # How many points the k-point is neighbor?
            self.k_basis_rotation.append(same_group[np.argmax(points)])                     # There is only one degenerate point
            
            k__deg, bn1_deg, bn2_deg = same_group[np.argmax(points)]
            self.signal_final[k__deg, bn1_deg] = DEGENERATE                         # Signal k_point as Degenerate
            self.signal_final[k__deg, bn2_deg] = DEGENERATE                         # Signal k_point as Degenerate
        
        self.k_basis_rotation += self.degenerate_final
        self.degenerate_final = np.array(self.degenerate_final)
        self.k_basis_rotation = np.array(self.k_basis_rotation)
        self.k_basis_rotation = np.unique(self.k_basis_rotation, axis=0)

    def print_report(self, signal_report: npt.NDArray, description:str, show:bool=True):
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
        for bn in self.bands:
            band_result = signal_report[:, bn]                              # Obtain all k-point' signals for band bn
            report = [np.sum(band_result == s) for s in range(MAX)]         # Set the band report
            report.append(np.round(self.final_score[bn], 4))                # Set the final score
            bands_report.append(report)

            self.logger.debug(f'\t\t\tNew Band: {bn}\tnr fails: {report[0]}')

        ###########################################################################
        # Set up the data representation
        ###########################################################################
        bands_report = np.array(bands_report)
        final_report += '\n\t\t Signaling: how many events ' + \
                        'in each band signaled.\n'
        bands_header = '\n\t\t Band | '
        
        header_text = EVALUATE_RESULT_HEADER
        header = header_text + ['Score']

        n_spaces = len(str(np.max(bands_report[:, -1]))) + 4

        bands_header += ''.join(f"{h:^{n_spaces}}" for h in header[:-1])
        bands_header += f'{header[-1]:>8}'


        final_report += bands_header + '\n\t\t'
        final_report += '-' * len(bands_header)

        for bn, report in enumerate(bands_report):
            # Make the report
            final_report += f'\n\t\t {bn+self.min_band:<4d} | '
            final_report += ''.join(f"{int(r):^{n_spaces}d}" for r in report[:-1])
            final_report += f'{report[-1]:>8.4f}'
        final_report += '\n'
        if show:
            self.logger.info(final_report)              # Show on screen
            return None
        return final_report, bands_report

    def report(self):
        self.final_report = ''
        self.final_report += '*************************************************************************************************\n'
        self.final_report += '|                                       SOLUTION REPORT                                         |\n'
        self.final_report += '*************************************************************************************************\n\n'

        self.final_report += f'\n\tLegend of report:\n'
        self.final_report += EVALUATE_RESULT_HELP
        self.final_report += '\n'
        # report_s1, report_a1 = self.print_report(self.signal_final, 'Final Report for dot-product information', show=False)
        # self.final_report += report_s1
        # self.final_report += '\n'
        retport_s2, report_a2 = self.print_report(self.signal_final, 'Final Report', show=False)
        self.final_report += retport_s2
        self.final_report += '\n'

        self.final_report += '\n\t{:-^30}\n\t{: <30}\n\t{:-^30}\n'.format('', 'Summary:', '')

        number_of_degenerates = len(self.degenerate_final)
        number_of_k_basis_rotation = len(self.k_basis_rotation)

        if number_of_k_basis_rotation > 0:
            self.final_report += f'\n\n\t\tNumber of degenerate points: {number_of_degenerates}\n'
            for k1, bn1, bn2 in self.degenerate_final:
                self.final_report += f'\n\t\t\t* K-point: {k1} Bands: {bn1 + self.min_band}, {bn2 + self.min_band}'
            self.final_report += f'\n\n\t\t  ' + textwrap.fill(
                        f"Degenerate points refer to instances where a point shares the same numerical energy value with another "
                        f"k-point, and the dot product with its neighboring points falls within the range of 0.5 to "
                        f"{DEGENERATE_TOLERANCE}. As a result, it is necessary for the rotation basis program to be executed for "
                        f"these points. A total of {number_of_degenerates} degenerate points were found and saved in the 'data' folder "
                        f"under the name 'degenerate_final.npy'. A total of {number_of_k_basis_rotation} points with dot products in the "
                        f"range of 0.5 to {DEGENERATE_TOLERANCE}—regardless of whether they share the same eigenvalue—were saved in "
                        f"'data/k_basis_rotation.npy'. These include all degenerate dot product points, but not all necessarily have "
                        f"identical numerical eigenvalues.",
                        width=110,
                        subsequent_indent='\t\t  ') + '\n'


        TOL_USABLE = 0.90           # Minimum score to consider a band as usable.
        n_recomended = self.min_band
        
        for i, s in enumerate(self.final_score):
            if s <= TOL_USABLE:
                break
            n_recomended += 1

        max_bands_used = []
        for bn_i, band in enumerate(self.cluster_bands):
            max_band = np.max(band.nodes[:, 1])
            max_bands_used.append(max_band)

        max_band_energies_corrected = []
        for bn_i, max_band_i in enumerate(max_bands_used):
            if max_band_i == bn_i:
                max_band_energies_corrected.append(max_band_i)
                continue
            bn_j = max_band_i
            max_band_j = max_bands_used[bn_j]
            while bn_j != max_band_j:
                bn_j = max_band_j
                max_band_j = max_bands_used[bn_j]

            max_band_energies_corrected.append(bn_j)

        iterable_bands = []

        bn_i = 0
        while bn_i <= self.bands[-1]:
            iterable_bands.append(self.min_band + np.arange(bn_i, max_band_energies_corrected[bn_i] + 1))
            bn_i = max_band_energies_corrected[bn_i] + 1

        if n_recomended in iterable_bands[-1]:
            message = textwrap.fill(
                f"It was solved up to band {n_recomended}. However, this band belongs to the group of bands "
                f"({', '.join(map(str, iterable_bands[-1]))}), which may be incomplete due to the absence of nearby higher-energy bands "
                f"that could overlap energetically. Therefore, it is recommended to consider only the bands from {self.min_band} "
                f"up to {np.max(iterable_bands[-2])}.",
                width=110,
                initial_indent='\n\n\t',
                subsequent_indent='\t  '
            )
        else:
            message = textwrap.fill(
                f"You can use the bands from {self.min_band} up to {n_recomended + self.min_band - 1}.",
                width=110,
                initial_indent='\n\n\t',
                subsequent_indent='\t  '
            )

        self.final_report += message

        self.final_report += f'\n\n\tThe information is stored in the `completed_bands.npy` file.'
        self.final_report += f'\n\t\t  Band: ' + ', '.join([str(bn+self.min_band) for bn in self.bands])

        if number_of_degenerates > 0:
            self.final_report += f'\n\n\tTo use the program basis rotation, you must run the program for the first {n_recomended + 1} bands.'
            self.final_report += f'\n\n\t\t `$ berry basis {n_recomended}`'

        self.final_report += '\n\n*************************************************************************************************\n'

        return self.final_report
