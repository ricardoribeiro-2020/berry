'''
Utility Classes and Functions

This file contains utility classes and functions that support the main algorithm in the core module. These 
utilities are designed to handle specific tasks like data manipulation, computation, and assisting in the 
implementation of the clustering logic.

Last Modified: [2025-12-11]

Main Classes:
----------
    - Component: Represents a component of the mesh, with methods for merging and checking intersections.
    - Cluster: Represents a cluster of components, with methods for merging and calculating similarity.


Main Functions:
----------
    - solver_iterable: This function handles the iterative process of solving clusters for a given set of bands. It 
    iterates through the components, handling normal and degenerate components, and applies energy and neighborhood 
    constraints to form clusters.

    - solver_algorithm: This function acts as the main orchestration engine, calling the `solver_iterable` for each 
    set of bands. It manages parallelism, logging, and verbose output, and ensures that the process runs efficiently 
    across multiple processors or sequentially.
'''

import numpy as np
import networkx as nx
import datetime
import time
import os

from functools import partial
from multiprocessing import Pool

from scipy.optimize import curve_fit

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from collections import deque


def print_bool_array(A):
    result_str = ''
    for row in A:
        result_str += ' '.join(['■' if x == 1 else '□' for x in row] ) + '\n'
    
    return result_str

def print_int_array(A):
    result_str = ''
    for row in A:
        # Number if greater than 0
        # otherwise ■ if -1 else □
        result_str += ' '.join([f'{x}' if x > 0 else '■' if x == -1 else '☒' for x in row]) + '\n'
    return result_str


def compute_edges_for_vi(vi, args):
    '''
    Compute the edges of the graph.
    '''

    neighbors, similarity, tol_to_be_an_edge, nks, total_bands, index_neighs, bands, neighs_points = args

    f_map = None
    tol_to_be_an_edge = f_map(tol_to_be_an_edge) if f_map is not None else tol_to_be_an_edge

    edges = []

    kp_i = vi[0]
    bn_i = vi[1]

    neighs = np.tile(neighbors[kp_i], total_bands)
    index_neighs = np.tile(index_neighs, total_bands)
    neighbors_vectors_indexes = neighs + nks * bands

    if neighs_points is None:
        neighbors_vectors_indexes = neighbors_vectors_indexes[neighs > kp_i]
        kp_i_index_neighs = index_neighs[neighs > kp_i]
    else:
        neighbors_vectors_indexes = neighbors_vectors_indexes[neighs != -1]
        kp_i_index_neighs = index_neighs[neighs != -1]


    neighbors_kpoints = neighbors_vectors_indexes % nks

    neighbors_vectors_indexes = neighbors_vectors_indexes[neighbors_kpoints != kp_i]
    neighbors_kpoints = neighbors_kpoints[neighbors_kpoints != kp_i]
    neighbors_bands = neighbors_vectors_indexes // nks

    similarity_vi_vj = similarity[kp_i, kp_i_index_neighs, bn_i, neighbors_bands]
    if f_map is not None:
        similarity_vi_vj = f_map(similarity_vi_vj)
    similarity_vj_indexes = neighbors_vectors_indexes[similarity_vi_vj > tol_to_be_an_edge]
    similarity_vi_vj = similarity_vi_vj[similarity_vi_vj > tol_to_be_an_edge]

    if neighs_points is not None:
        edges += [[vi, vj, weight] for vj, weight in zip(similarity_vj_indexes, similarity_vi_vj) if vj in neighs_points]
    else:
        edges += [[vi, vj, weight] for vj, weight in zip(similarity_vj_indexes, similarity_vi_vj)]

    # Free memory
    del neighbors_vectors_indexes, neighbors_kpoints, neighbors_bands, similarity_vi_vj, kp_i_index_neighs

    return edges

def compute_edges(vectors, neighbors, similarity, total_bands=None, tol_to_be_an_edge=0.9, n_process=1, neighs_points=None, verbose=False, parallel=True):
    '''
    Compute the edges of the graph.
    '''
    edges = []

    num_vectors = vectors.shape[0]

    nks = neighbors.shape[0]
    total_bands = np.arange(similarity[0, 0].shape[0]) if total_bands is None else total_bands
    number_of_neighbors = neighbors.shape[1]
    index_neighs = np.arange(neighbors[0].shape[0])
    bands = np.repeat(total_bands, number_of_neighbors)

    args = (neighbors, similarity, tol_to_be_an_edge, nks, total_bands.shape[0], index_neighs, bands, neighs_points)

    compute_edges_wrapper = partial(compute_edges_for_vi, args=args)

    if not parallel:
        edges = [compute_edges_for_vi(vi, args) for vi in vectors]
        edges = [item for sublist in edges for item in sublist]
        return edges

    if verbose:
        with Pool(n_process) as pool:
                edges += tqdm(pool.imap(compute_edges_wrapper, vectors, chunksize= 1000), total=num_vectors, desc='Computing edges')
    else:
        with Pool(n_process) as pool:
            edges += pool.map(compute_edges_wrapper, vectors)

    edges = [item for sublist in edges for item in sublist]
    
    return edges


class Component:
    """
    A class representing a component of a physical system described by a graph of k-points and energy bands.

    Attributes:
        graph (networkx.Graph): A graph representing the k-points and energy bands in the system.
        dimensions (int): The number of dimensions (1, 2, or 3) describing the k-point mesh.
        nks (int): The total number of k-points.
        mesh_kpoints_index (numpy.ndarray): The indices of k-points in the mesh grid.
        array_kpoints_index (numpy.ndarray): The k-point indices as an array.
        energies (numpy.ndarray): The energy values corresponding to k-points and bands.
        nodes (numpy.ndarray): The nodes (k-points and bands) of the graph.
        k_points (numpy.ndarray): The k-points of the system.
        k_bands (numpy.ndarray): The energy bands associated with each k-point.
        number_of_nodes (int): The number of nodes in the graph.
        mesh (numpy.ndarray): A binary mesh representing the k-points.
        energy_mesh (numpy.ndarray): The energy values on the mesh grid.
        gradient_energy (numpy.ndarray): The gradient of the energy mesh.
        gradient (numpy.ndarray): The gradient of the mesh.
        gradient_abs (numpy.ndarray): The absolute value of the gradient.
        min_energy (float): The minimum energy in the mesh.
        positive_gradient (numpy.ndarray): The positive part of the gradient.
        negative_gradient (numpy.ndarray): The negative part of the gradient.
        positive_boundary (numpy.ndarray): The positive boundary of the mesh.
        negative_boundary (numpy.ndarray): The negative boundary of the mesh.
        positive_neighs (numpy.ndarray): The positive neighbors of the mesh.
        negative_neighs (numpy.ndarray): The negative neighbors of the mesh.
        boundary (numpy.ndarray): A boolean mask indicating the boundary points.
        neighs (numpy.ndarray): A boolean mask indicating the neighboring points.

    Methods:
        __init__(graph, dimensions, number_of_kpoints, mesh_kpoints_index, array_kpoints_index, energies):
            Initializes the Component object with the provided parameters and computes mesh and gradient.
        
        compute_mesh_and_gradient():
            Computes the mesh and gradient based on the graph and dimensions.
        
        does_intersect(other):
            Checks if this component intersects with another component.
        
        to_cluster():
            Converts the component into a Cluster object.
        
        remove_points(mesh_to_remove):
            Removes k-points from the component and returns the resulting new components.
    """
    mesh_kpoints_index: np.ndarray = None
    array_kpoints_index: np.ndarray = None
    energies: np.ndarray = None
    dimensions: int = None
    nks: int = None

    @classmethod
    def set_global_data(cls, dimensions, nks, mesh_kpoints_index, array_kpoints_index, energies):
        """
        Supply the big arrays exactly once to the Component class.
        """
        cls.mesh_kpoints_index = mesh_kpoints_index
        cls.array_kpoints_index = array_kpoints_index
        cls.energies = energies

        cls.dimensions = dimensions
        cls.nks = nks

    def __init__(self, graph):
        self.graph = graph
        
        self.compute_boundary()

    def min_energy(self):
        """
        Returns the minimum energy in the component.
        """
        if Component.energies is None:
            raise ValueError("Energies are not set for this component.")
    
        coords_tuples = tuple(self.active_coords.T)

        return np.mean(Component.energies[self.nodes[:, 1], coords_tuples])
    
    def remove_repeated_k_points(self):
        'If there are repeated k-points in the component, remove them and return new components each with one of the repeated k-points.'
        k_points = self.nodes[:, 0]

        set_k_points = set(k_points)
        if len(set_k_points) == len(k_points):
            return [self]

        new_components = []

        unique_points = []
        unique_graph = nx.Graph()

        for k in set_k_points:
            mask = (k_points == k)
            if np.sum(mask) == 1:
                unique_points.append(self.nodes[mask][0])
                continue

            nodes_to_remove = self.nodes[mask]
            for k_ in nodes_to_remove:
                new_graph = nx.Graph()
                new_graph.add_node((k_[0], k_[1]))
                new_components.append(Component(new_graph))
            
        for point in unique_points:
            unique_graph.add_node((point[0], point[1]))
        
        new_components.append(Component(unique_graph))         

        return new_components

    def compute_boundary(self):
        """
        Populate self._boundary_mask and self._interior_mask from self.active_coords.
        We never allocate a full D-dim grid; we only inspect each node's ±1 neighbors to
        see whether those ±1 neighbors are also in this component.
        """
        self.nodes = np.array(self.graph.nodes)
        self.number_of_nodes = self.graph.number_of_nodes()

        self.active_coords = Component.array_kpoints_index[self.nodes[:, 0]]

        n = self.number_of_nodes
        D = Component.dimensions

        # Build a hash‐set of all active_coord tuples to test membership in O(1):
        if Component.dimensions > 1:
            self.coord_tuples = { tuple(self.active_coords[i]) for i in range(n) }
        else:
            self.coord_tuples = { self.active_coords[i] for i in range(n) }

        self.neighbor_coords = set()

        boundary = np.zeros(n, dtype=bool)
        interior = np.zeros(n, dtype=bool)

        for i in range(n):
            coord = self.active_coords[i]
            is_interior = True
            for dim in range(D):
                for offset in (-1, +1):
                    neighbor_coord = list(coord) if Component.dimensions > 1 else [coord]
                    neighbor_coord[dim] += offset
                    neighbor_coord = tuple(neighbor_coord) if Component.dimensions > 1 else neighbor_coord[0]

                    if neighbor_coord not in self.coord_tuples:
                        self.neighbor_coords.add(neighbor_coord)
                        boundary[i] = True
                        is_interior = False
                        break

                if not is_interior:
                    break
                
            if not boundary[i]:
                # We never saw a missing neighbor; so it’s truly interior:
                interior[i] = True


        self.sigma_E = np.zeros(np.sum(boundary), dtype=float)
        N_neighbor_for_sigma_E = 2

        for i, k in enumerate(np.arange(n)[boundary]):
            coord = self.active_coords[k]
            E_mesh = np.zeros((Component.dimensions, 2*N_neighbor_for_sigma_E + 1), dtype=float)
            Ek_index = (self.nodes[k, 1], ) + tuple(coord) if Component.dimensions > 1 else  (self.nodes[k, 1], coord)
            Ek = Component.energies[Ek_index]

            for dim in range(Component.dimensions):
                E_mesh[dim, N_neighbor_for_sigma_E] = Ek

                for offset in range(-N_neighbor_for_sigma_E, N_neighbor_for_sigma_E + 1):
                    if offset == 0:
                        continue
                    neighbor_coord = list(coord) if Component.dimensions > 1 else [coord]
                    neighbor_coord[dim] += offset
                    neighbor_coord = tuple(neighbor_coord) if Component.dimensions > 1 else neighbor_coord[0]
                    
                    if neighbor_coord in self.coord_tuples:
                        k_neighbor = np.where((self.active_coords == neighbor_coord).all(axis=1))[0][0] if Component.dimensions > 1 else np.where(self.active_coords == neighbor_coord)[0][0]
                        b_neighbor = self.nodes[k_neighbor, 1]
                        Ek_neighbor = Component.energies[(b_neighbor, ) + tuple(neighbor_coord) if Component.dimensions > 1 else (b_neighbor, neighbor_coord,)]
                        E_mesh[dim, N_neighbor_for_sigma_E + offset] = Ek_neighbor
                    else:
                        # Add None if the neighbor is not in the component
                        E_mesh[dim, N_neighbor_for_sigma_E + offset] = None

            # Compute the gradient of the energy mesh
            grad_E = [np.gradient(E_axis) for E_axis in E_mesh]
            abs_E = []

            for dim in range(Component.dimensions):
                grad_E_not_none = [g for g in grad_E[dim].flatten() if np.isfinite(g)]
                abs_E.append(np.mean(grad_E_not_none) if len(grad_E_not_none) > 0 else None)

            if len([a for a in abs_E if a is not None]) == Component.dimensions:
                self.sigma_E[i] = np.linalg.norm(abs_E)
            else:
                self.sigma_E[i] = None


        self._boundary_mask = boundary
        self._interior_mask = interior

    def does_intersect(self, other:'Component'):
        """
        Checks if this component intersects with another component.

        Parameters:
            other (Component): Another Component object to check for intersection.

        Returns:
            bool: True if the components intersect, False otherwise.
        """
        # Extract just the k_flat indices for each component:
        ours = set(self.nodes[:, 0].tolist())
        theirs = set(other.nodes[:, 0].tolist())
        # If intersection of k‐flats is nonempty, they intersect.
        return not ours.isdisjoint(theirs)
    
    def to_cluster(self):
        new_cluster = Cluster(self.graph)

        return new_cluster
    
    
    def compute_mesh_and_gradient(self):
        """
        !! This method is deprecated and will be removed in future versions. Use `compute_boundary` instead.

        Computes the mesh and gradient of the energy values based on the graph's k-points and energy bands.

        Depending on the dimensionality, the method calculates the mesh grid and gradient of the energies
        for the component and also identifies boundaries and neighbors.
        """

        self.mesh = np.zeros(Component.mesh_kpoints_index.shape, dtype=bool)
        #self.energy_mesh = np.zeros(Component.mesh_kpoints_index.shape, dtype=np.float64)

        if Component.dimensions == 1:
            self.mesh[self.nodes[:, 0]] = 1
            #self.energy_mesh[self.nodes[:, 0]] = Component.energies[self.nodes[:, 1], self.nodes[:, 0]]
            #gradient_energy = np.abs(np.gradient(self.energy_mesh))
            self.gradient = np.sign(np.gradient(self.mesh))

        elif Component.dimensions == 2:
            self.k_points_index = Component.array_kpoints_index[self.nodes[:, 0]]
            self.mesh[self.k_points_index[:, 0], self.k_points_index[:, 1]] = 1
            #self.energy_mesh[self.k_points_index[:, 0], self.k_points_index[:, 1]] = Component.energies[self.nodes[:, 1], self.k_points_index[:, 0], self.k_points_index[:, 1]]
            #gradient_energy = np.sum(np.array(np.gradient(self.energy_mesh, axis=(0, 1)))**2, axis=0)
            self.gradient = np.sign(np.array(np.gradient(self.mesh, axis=(0, 1))))

        else:
            self.k_points_index = Component.array_kpoints_index[self.nodes[:, 0]]
            self.mesh[self.k_points_index[:, 0], self.k_points_index[:, 1], self.k_points_index[:, 2]] = 1
            #self.energy_mesh[self.k_points_index[:, 0], self.k_points_index[:, 1], self.k_points_index[:, 2]] = Component.energies[self.nodes[:, 1], self.k_points_index[:, 0], self.k_points_index[:, 1], self.k_points_index[:, 2]]
            #gradient_energy = np.sum(np.array(np.gradient(self.energy_mesh, axis=(0, 1, 2)))**2, axis=0)
            self.gradient = np.sign(np.array(np.gradient(self.mesh, axis=(0, 1, 2))))

        if Component.dimensions == 1:
            gradient_abs = np.abs(self.gradient)
        else:
            gradient_abs = np.sign(np.sum(self.gradient**2, axis=0))

        #self.min_energy = np.min(self.energy_mesh[self.mesh == 1])

        positive_gradient = self.gradient.copy()
        positive_gradient[positive_gradient < 0] = 0
        negative_gradient = np.abs(self.gradient - positive_gradient)

        positive_boundary = self.mesh * positive_gradient
        negative_boundary = self.mesh * negative_gradient

        self.positive_neighs = positive_gradient - positive_boundary
        self.negative_neighs = negative_gradient - negative_boundary

        self.boundary = np.array(self.mesh * gradient_abs, dtype=bool)
        self.neighs = np.array(gradient_abs - self.boundary, dtype=bool)

    
    def remove_points(self, mesh_to_remove):
        """
        Removes k-points from the component based on the given mesh mask and returns the resulting components.

        Parameters:
            mesh_to_remove (numpy.ndarray): A binary mask indicating which k-points to remove.

        Returns:
            list: A list of new Component objects after removing the specified k-points.
        """
        k_points = Component.mesh_kpoints_index[mesh_to_remove == 1]

        # Unfrozen the graph to allow modifications
        self.graph = self.graph.copy()

        for k, b in self.nodes:
            if k in k_points.flatten():
                continue
            neighbor_coord_plus = list(Component.array_kpoints_index[k]) if Component.dimensions > 1 else [Component.array_kpoints_index[k]]
            for d in range(Component.dimensions):
                offset = 1
                if neighbor_coord_plus[d] + offset >= Component.mesh_kpoints_index.shape[d]:
                    continue
                neighbor_coord_plus[d] += offset
                k_neig = Component.mesh_kpoints_index[tuple(neighbor_coord_plus)]
                if k_neig in k_points.flatten():
                    continue
                b_neig = self.nodes[:, 1][self.nodes[:, 0] == k_neig]
                if len(b_neig) > 0:
                    self.graph.add_edge((k, b), (k_neig, b_neig[0]))

        nodes_to_remove = []

        for k in k_points.flatten():
            if not k in self.nodes[:, 0]:
                continue
            b = self.nodes[:, 1][self.nodes[:, 0] == k][0]
            nodes_to_remove.append((k, b))

        self.graph = self.graph.copy()
        self.graph.remove_nodes_from(nodes_to_remove)
        
        new_components = []
        for node in nodes_to_remove:
            new_graph = nx.Graph()
            new_graph.add_node(node)
            new_components.append(Component(new_graph))
        
        new_components += [Component(self.graph.subgraph(c)) for c in nx.connected_components(self.graph)]

        return new_components

    

class Cluster(Component):
    """
    A class representing a cluster of physical components, inheriting from the `Component` class.

    A `Cluster` is a collection of components that can merge with other components based on certain criteria.
    It inherits from `Component` and adds functionality for computing similarity between components,
    checking merge conditions, and handling outliers during clustering operations.

    Attributes:
        outliers (numpy.ndarray): An array to store outliers identified during clustering.
    
    Methods:
        __init__(graph, dimensions, number_of_kpoints, mesh_kpoints_index, array_kpoints_index, energies):
            Initializes the Cluster object and sets up the outliers array.
        
        can_merge(Sample: Component, verbose=False):
            Determines whether this cluster can merge with a given sample component based on specific conditions.

        enery_score(Sample: Component, energies, neighbors):
            Computes a score based on the energy differences between the cluster and a sample component.

        dot_product_score(Sample: Component, connections, neighbors):
            Computes a dot product score for comparing the cluster and sample component, handling outliers.

        compute_similarity_with_sample(Sample: Component, connections, neighbors, compute='all', energies=None):
            Computes the overall similarity between the cluster and a sample component, either based on energy or dot product score.

        merge(Sample: Component):
            Merges the current cluster with a sample component by combining their graphs and recalculating the mesh and gradient.
    """
    def __init__(self, graph):
        super().__init__(graph)

        self.outliers = np.array([], dtype=np.int8)

    def can_merge(self, Sample:Component, verbose=False):
        """
        Determines whether this cluster can merge with the provided sample component.

        The merge condition is based on:
            - The total number of nodes in the merged cluster not exceeding the total number of k-points.
            - The two clusters not intersecting.
            - At least one neighboring mesh point between the two clusters or shared boundaries between them.

        Parameters:
            Sample (Component): A sample component to check for merge compatibility.
            verbose (bool): If True, provides additional information about the merge conditions. (Not used)

        Returns:
            bool: True if the clusters can merge, False otherwise.
        """
        total_number_of_nodes = self.number_of_nodes + Sample.number_of_nodes
        if total_number_of_nodes > Component.nks:
            if verbose:
                return False, f"Total number of nodes exceeds total k-points"
            return False

        if self.does_intersect(Sample):
            if verbose:
                return False, "Clusters intersect." + f" Shared k-points"
            return False

        self_neighs_coords = self.neighbor_coords
        sample_coords = Sample.coord_tuples

        self_boundary_coords = {
            tuple(coord) if Component.dimensions > 1 else coord
            for coord in self.active_coords[self._boundary_mask]
        }

        sample_neighs_coords = set(Sample.neighbor_coords)

        neighs_intersect_sample_coords = not self_neighs_coords.isdisjoint(sample_coords)
        self_intersect_sample_neighs = not self_boundary_coords.isdisjoint(sample_neighs_coords)
        if verbose:
            if neighs_intersect_sample_coords:
                return True, "Clusters can merge: neighboring mesh points intersect."
            if self_intersect_sample_neighs:
                return True, "Clusters can merge: shared boundaries intersect."
            return False, "No neighboring mesh points or shared boundaries intersect."

        return neighs_intersect_sample_coords or self_intersect_sample_neighs

    def enery_score(self, Sample:Component, energies, neighbors):
        """
        Computes an energy score for comparing the cluster with a sample component.

        The energy score is based on the energy differences between the sample and cluster points.

        Parameters:
            Sample (Component): A sample component to compare against the cluster.
            energies (numpy.ndarray): Energy values for the k-points and bands.
            neighbors (numpy.ndarray): Neighboring k-points and bands for comparison.

        Returns:
            float: The energy score between the cluster and sample component (1 - energy difference).
        """
        cluster_points = self.nodes[self._boundary_mask]
        sample_points = Sample.nodes[Sample._boundary_mask]

        energies_differences = np.array(obtain_energy_difference(sample_points, cluster_points, self.sigma_E, energies, neighbors))

        if len(energies_differences) != 1:
            energies_score = np.mean(energies_differences)
        else:
            energies_score = energies_differences[0]
        
        return energies_score
    
    def dot_product_score(self, Sample:Component, connections, neighbors):
        """
        Computes a dot product score between the cluster and a sample component.

        This method also handles outliers, removes them from the score calculation, and updates the outliers array.

        Parameters:
            Sample (Component): A sample component to compare against the cluster.
            connections (numpy.ndarray): The connections between k-points and bands.
            neighbors (numpy.ndarray): The neighboring k-points and bands for comparison.

        Returns:
            float: The dot product score between the cluster and sample component.
        """

        cluster_points = self.nodes[self._boundary_mask]
        sample_points = Sample.nodes[Sample._boundary_mask]

        dot_products = np.array(obtain_dot_products(sample_points,  cluster_points, connections, neighbors))
        if len(dot_products) != 1:
            dot_product_score = np.mean(dot_products)
            dot_product_score_std = np.std(dot_products)
            find_outliers = np.abs(dot_products - dot_product_score) / dot_product_score_std > 1
            outliers = dot_products[find_outliers]
            self.outliers = np.concatenate((self.outliers, outliers))
            dot_product_score = np.mean(dot_products[~find_outliers])
        else:
            dot_product_score = dot_products[0]
        
        return dot_product_score

    
    def compute_similarity_with_sample(self, Sample:Component, connections, neighbors, compute='all', energies=None):
        """
        Computes the similarity between the cluster and a sample component.

        The similarity is based on either energy differences or dot product score (or both).

        Parameters:
            Sample (Component): The sample component to compare to the cluster.
            connections (numpy.ndarray): The connections between k-points and bands.
            neighbors (numpy.ndarray): The neighboring k-points and bands for comparison.
            compute (str): Specifies the type of score to compute ('all', 'dot_product', or 'energy').
            energies (numpy.ndarray, optional): Energy values for the k-points and bands (required if `compute` is 'energy').

        Returns:
            list: A list of similarity scores ([dot_product_score, energy_score]).
        """
        dot_product_score = np.nan
        energy_score = np.nan

        if compute == 'all':
            if energies is None:
                raise ValueError("Energies were not provided")
            dot_product_score = self.dot_product_score(Sample, connections, neighbors)
            energy_score = self.enery_score(Sample, energies, neighbors)
        elif compute == 'dot_product':
            dot_product_score = self.dot_product_score(Sample, connections, neighbors)
            return dot_product_score
        elif compute == 'energy':
            if energies is None:
                raise ValueError("Energies were not provided")
            energy_score = self.enery_score(Sample, energies, neighbors)
            return energy_score

        similarity = [dot_product_score, energy_score]
        
        return similarity
    
    def merge(self, Sample:Component):
        """
        Merges the current cluster with a sample component by combining their graphs and recalculating the mesh and gradient.

        Parameters:
            Sample (Component): The sample component to merge with the cluster.
        """
        self.graph = nx.compose(self.graph, Sample.graph)
        self.compute_boundary()


def obtain_dot_products(sample_points, cluster_points, connections, neighbors):
    '''
    Obtain the dot products between the cluster and the sample.
    '''

    dot_products = []

    for k_s, b_s in sample_points:
        for i_neigh, k in enumerate(neighbors[k_s]):
            if k not in cluster_points[:, 0]:
                continue
            k_c = np.where(cluster_points[:, 0] == k)[0][0]
            b_c = cluster_points[k_c, 1]
            dot_products.append(connections[k_s, i_neigh, b_s, b_c])
    return dot_products

def obtain_energy_difference(sample_points, cluster_points, sigma_E, energies, neighbors):
    '''
    Obtain the difference on energy between the cluster and the sample.
    '''
    energy_difference = []

    # Fallback sigma when a boundary point has no in-component neighbors to estimate
    # its local gradient. Use the mean of all valid boundary gradients so the scale
    # stays physically meaningful; only drop to the hard-coded 1e-3 if none are finite.
    finite_sigmas = sigma_E[np.isfinite(sigma_E) & (sigma_E > 0)]
    fallback_sigma = np.mean(finite_sigmas) if len(finite_sigmas) > 0 else 1e-3

    for k_s, b_s in sample_points:
        E_ks_bs = energies[k_s, b_s]
        for i_neigh, k in enumerate(neighbors[k_s]):
            if k not in cluster_points[:, 0]:
                continue
            k_c = np.where(cluster_points[:, 0] == k)[0][0]
            b_c = cluster_points[k_c, 1]
            E_kc_bc = energies[k, b_c]
            sigma_E_kc = sigma_E[k_c]
            if not np.isfinite(sigma_E_kc) or sigma_E_kc == 0:
                energy_difference.append(np.exp(- (np.abs(E_kc_bc - E_ks_bs) - fallback_sigma)**2 / (2 * fallback_sigma**2)))
            else:
                energy_difference.append(np.exp(- (np.abs(E_kc_bc - E_ks_bs) - sigma_E_kc)**2 / ( 2 * sigma_E_kc**2)))

    return energy_difference



def solver_iterable(bands_to_solve, components, degenerate_components, degenerate_points, connections, neighbors, mesh_kpoints_index, nks, max_band_energies, mask_degenerates, bands, logger, energies=None, tolerance=0.99, alpha=0.5, min_band=0):
    """
    ----------------------------------------------------------------------------------------------------------------------
    Main loop for solving clusters of components across multiple bands.
    ----------------------------------------------------------------------------------------------------------------------
    
    This function iterates over a list of bands and their associated components, merging components into clusters based on their 
    mesh compatibility and point intersections. It handles both normal and degenerate components, ensuring that clusters are 
    formed while adhering to energy and point allocation constraints. 

    Parameters:
    -----------
    bands_to_solve : list of int
        The list of band indices to be solved.
    
    components : list
        A ordered list where the i-th element corresponds to the list of components to be processed for the band i.
    
    degenerate_components : list
        A ordered list where the i-th element corresponds to the list of degenerate components for the band i.
    
    degenerate_points : list
        A list of degenerate points that may need to be handled separately during clustering.
    
    connections : array-like
        The connection matrix used to evaluate the relationships between components.
    
    neighbors : array-like
        The neighbor matrix used to evaluate the local neighborhood of each component.
    
    mesh_kpoints_index : array-like
        An index of mesh k-points used to map the components and their points.
    
    nks : int
        The number of k-points per band.
    
    max_band_energies : list of int
        The maximum energies for each band.
    
    mask_degenerates : array-like
        A list of masks to identify degenerate points in the components for the band.
    
    bands : list of int
        A list of all band indices to process.
    
    logger : Logger
        A logger instance to log progress and information.
    
    energies : array-like, optional
        The energies associated with the components, if available. Default is None.
    
    tolerance : float, optional
        The tolerance value for similarity scoring between components. Default is 0.99.

    Returns:
    --------
    clusters : list of Cluster
        A list of formed clusters after solving for all bands.
    """

    '''
    ----------------------------------------------------------------------------------------------------------------------
                                                        Main loop
    ----------------------------------------------------------------------------------------------------------------------
    '''
    clusters = []
    compute_energy_only = None

    if energies is None:
        compute_score = lambda sample: cluster_bn_i.compute_similarity_with_sample(sample, connections, neighbors, compute='dot_product')
        compute_energy_only = compute_score
    else:
        def compute_score_(sample, alpha=0.5):
            dot_product_score, energy_score = cluster_bn_i.compute_similarity_with_sample(sample, connections, neighbors, energies=energies, compute='all')
            return alpha * dot_product_score + (1 - alpha) * energy_score
        compute_score = partial(compute_score_, alpha=alpha)
        compute_energy_only = partial(compute_score_, alpha=0)

    for bn_i in bands_to_solve:
        # Sort the components by number of nodes
        components[bn_i] = sorted(components[bn_i], key=lambda x: x.number_of_nodes, reverse=True)

        '''
        ---------------------------------------------------------------------------
        1. Check if the first component is solved
        ---------------------------------------------------------------------------
        '''

        # Check if the first component is the full band

        if components[bn_i][0].number_of_nodes == nks:
            clusters.append(components[bn_i][0])
            # Check if there is more than one component
            # If there is, add the rest of the components to the next band
            logger.info(
                    f"[Band {bn_i + min_band}] Progress: {1:.2%}"
                )
            if len(components[bn_i][1:]) > 0 and bn_i + 1 <= max_band_energies[bn_i]:
                components[bn_i + 1] += components[bn_i][1:]
            else:
                # If the band bn_i + 1 is not in the range of the bands that are being solved, then raise an error
                if len(components[bn_i][1:]) > 0:
                    raise ValueError(f"Band {bn_i + min_band} has more than one component, but the first one is already the full band")
            if bn_i == max_band_energies[bn_i]:
                logger.info(f"All bands from the group {np.min(bands_to_solve) + min_band} - {bn_i + min_band} have been solved. | Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                bands_final_temp = np.full((nks, np.max(max_band_energies) + 1), -1, dtype=int)
                for cluster, bn in zip(clusters, bands_to_solve):
                    for k, b in cluster.nodes:
                        bands_final_temp[k, bn] = b

                # Save the bands_final_temp to a file
                if not os.path.exists('temp'):
                    os.makedirs('temp')
                np.save(f'temp/bands_final_temp_{np.min(bands_to_solve) + min_band}_{bn_i + min_band}.npy', bands_final_temp)
                logger.info(f"Temporary file temp/bands_final_temp_{np.min(bands_to_solve) + min_band}_{bn_i + min_band}.npy saved.")
            continue
        
        attribution_points = np.zeros_like(mesh_kpoints_index, dtype=int)

        # Identify the biggest component that is not the full band
        # This is the component that will be used to attribute the points

        cluster_bn_i = components[bn_i].pop(0).to_cluster()
        
        for coord in cluster_bn_i.active_coords:
            idx_coord = tuple(coord) if Component.dimensions > 1 else coord
            attribution_points[idx_coord] = -1

        samples_that_can_be_added = []

        '''
        ------------------------------------------------------------------------
        2. Identify the samples that can be added to the cluster
        ------------------------------------------------------------------------
        '''

        samples_that_can_be_added = []
        remaining_samples = []
        
        number_of_available_points = np.sum(cluster_bn_i.number_of_nodes)

        for bn_j in bands[bn_i: max_band_energies[bn_i] + 1]:

            for deg_sample in degenerate_components[bn_j]:
                number_of_available_points += deg_sample.number_of_nodes

            samples_that_can_be_added.append([])
            remaining_samples.append([])

            components[bn_j] = deque(components[bn_j])

            while len(components[bn_j]) > 0:
                sample = components[bn_j].popleft()
                number_of_available_points += sample.number_of_nodes
                # Check if the sample intersect with the cluster
                # If it does, then it is not possible to add it to the cluster
                if cluster_bn_i.does_intersect(sample):
                    remaining_samples[-1].append(sample)
                    continue

                # Check if the sample intersect with the degenerate points
                sample_mesh = np.zeros_like(mesh_kpoints_index, dtype=bool)
                for coord in sample.active_coords:
                    idx_coord = tuple(coord) if Component.dimensions > 1 else coord
                    sample_mesh[idx_coord] = True

                if np.sum(sample_mesh * mask_degenerates[bn_i]) != 0:
                    if sample.number_of_nodes != 1:
                        components[bn_j] += sample.remove_points(sample_mesh * mask_degenerates[bn_i])
                        number_of_available_points -= sample.number_of_nodes
                        del sample
                        continue
                    degenerate_components[bn_j].append(sample)
                    continue

                # If the sample does not intersect with the cluster, then it can be added to the cluster
                attribution_points += 1 * sample_mesh
                samples_that_can_be_added[-1].append(sample)

            components[bn_j] = remaining_samples[-1]

        if number_of_available_points != nks * (max_band_energies[bn_i] - bn_i + 1):
            logger.info(f"Number of available points: {number_of_available_points}")
            logger.info(f"Number of points in the cluster: {cluster_bn_i.number_of_nodes}")
            logger.info(f"Number of points in the band: {nks * (max_band_energies[bn_i] - bn_i + 1)}")
            raise ValueError(f"Number of available points is not equal to the number of points in the band")
        
        if bn_i + 1 <= max_band_energies[bn_i]:
            components[bn_i + 1] += components[bn_i]
            components[bn_i] = []
        else:
            if len(components[bn_i]) > 0:
                raise ValueError(f"Band {bn_i + min_band} has remaining points, but the next band is not in the range of the bands that are being solved")


        '''
        ------------------------------------------------------------------------
        3. From the samples that can be added, check which ones can be added 
            to the cluster because there are not more than one point in the 
            same position
        ------------------------------------------------------------------------
        '''

        number_of_available_points = np.sum(cluster_bn_i.number_of_nodes)
        for bn_j in bands[bn_i: max_band_energies[bn_i] + 1]:
            for sample in degenerate_components[bn_j]:
                number_of_available_points += np.sum(sample.number_of_nodes)
            for sample in components[bn_j]:
                number_of_available_points += np.sum(sample.number_of_nodes)


        number_of_to_be_added_points = 0

        for samples in samples_that_can_be_added:
            for sample in samples:
                number_of_to_be_added_points += np.sum(sample.number_of_nodes)

        did_change = True
        number_of_points_added = 0
        while did_change:

            remaining_samples = []
            did_change = False

            number_of_points_total = 0

            for j, samples in enumerate(samples_that_can_be_added):
                remaining_samples.append([])

                samples = deque(samples)
                while len(samples) > 0:
                    sample = samples.popleft()

                    sample_mesh = np.zeros_like(mesh_kpoints_index, dtype=bool)
                    for coord in sample.active_coords:
                        idx_coord = tuple(coord) if Component.dimensions > 1 else coord
                        sample_mesh[idx_coord] = True

                    if np.sum(attribution_points[sample_mesh] == 1) == sample.number_of_nodes and cluster_bn_i.can_merge(sample):
                        number_of_points_added += sample.number_of_nodes
                        cluster_bn_i.merge(sample)
                        did_change = True
                        continue
                    
                    number_of_points_total += sample.number_of_nodes 
                    remaining_samples[j].append(sample)
                    
                    # attribution_points[attribution_points * sample.mesh == 1] = -1
            samples_that_can_be_added = remaining_samples

        for samples in samples_that_can_be_added:
            for sample in samples:
                number_of_available_points += sample.number_of_nodes

        number_of_available_points += number_of_points_added

        if number_of_available_points != nks * (max_band_energies[bn_i] - bn_i + 1):
            logger.info(f"Number of available points: {number_of_available_points}")
            logger.info(f"Number of points in the cluster: {cluster_bn_i.number_of_nodes}")
            logger.info(f"Number of points in the band: {nks * (max_band_energies[bn_i] - bn_i + 1)}")
            raise ValueError(f"Number of available points is not equal to the number of points in the band")
        

        add_degenerates_points = False
        check_degenerates_points = True
        only_energy_score = False

        '''
        -----------------------------------------------------------------------
        4. Solve the cluster using the samples that can be added
        -----------------------------------------------------------------------
        '''
        last_logged_percent = -1

        remaining_samples_ = [[] for _ in range(len(samples_that_can_be_added))]
        logger.info(
                    f"[Band {bn_i + min_band}] Progress: {0:.2%} | Degenerates are being included: {not check_degenerates_points}"
                )
        while cluster_bn_i.number_of_nodes < nks:

            progress = cluster_bn_i.number_of_nodes / nks
            current_percent = int(progress * 100)

            if current_percent % 10 == 0 and current_percent != last_logged_percent:
                logger.debug(
                    f"[Band {bn_i + min_band}] Progress: {progress:.2%} {cluster_bn_i.number_of_nodes}| Degenerates are being included: {not check_degenerates_points}"
                )
            #    last_logged_percent = current_percent

            samples_that_can_be_added = remaining_samples
            remaining_samples = []

            add_degenerates_points = add_degenerates_points or cluster_bn_i.number_of_nodes >= nks - len(degenerate_points[bn_i])

            if add_degenerates_points:

                only_energy_score = True

                remaining_points = np.zeros_like(mesh_kpoints_index, dtype=bool)
                for coord in cluster_bn_i.active_coords:
                    idx_coord = tuple(coord) if Component.dimensions > 1 else coord
                    remaining_points[idx_coord] = True

                remaining_points = np.logical_not(remaining_points)


                for j, samples in enumerate(degenerate_components[bn_i: max_band_energies[bn_i] + 1]):
                    remaining_samples_bnj = []

                    samples = deque(samples)
                    while len(samples) > 0:
                        sample = samples.popleft()
                        sample_mesh = np.zeros_like(mesh_kpoints_index, dtype=bool)
                        for coord in sample.active_coords:
                            idx_coord = tuple(coord) if Component.dimensions > 1 else coord
                            sample_mesh[idx_coord] = True
                        if np.sum(sample_mesh * remaining_points) == 0:
                            remaining_samples_bnj.append(sample)
                            continue
                        samples_that_can_be_added[j].append(sample)

                    degenerate_components[bn_i + j] = remaining_samples_bnj

                check_degenerates_points = False
                add_degenerates_points = False


            scores = []
                    
            for i, samples in enumerate(samples_that_can_be_added):
                remaining_samples.append([])
                for j, sample in enumerate(samples):
                    sample_mesh = np.zeros_like(mesh_kpoints_index, dtype=bool)
                    for coord in sample.active_coords:
                        idx_coord = tuple(coord) if Component.dimensions > 1 else coord
                        sample_mesh[idx_coord] = True
                    check_degenerates = check_degenerates_points and np.sum(sample_mesh * mask_degenerates[bn_i]) != 0
                    if not cluster_bn_i.can_merge(sample) or check_degenerates:
                        remaining_samples[-1].append(sample)
                        continue
                    if only_energy_score:
                        score = compute_energy_only(sample)
                    else:
                        score = compute_score(sample)
                    scores.append([i, j, score])


            scores = np.array(scores)
            #logger.info(f"[Band {bn_i}] Number of samples that can be added in this iteration: {len(scores)}")

            appended_samples = []

            if len(scores) == 0:
                add_degenerates_points = True
                if not check_degenerates_points:
                    logger.info(
                            f"[{bn_i}] There is no more points to be attributed."
                            f" The cluster has {cluster_bn_i.number_of_nodes} points."
                        )

                    for ri, samples in enumerate(remaining_samples):
                        logger.info(f"Band {bn_i + ri + min_band}: {len(samples)} samples, {len(remaining_samples_[ri])} remaining samples.")
                    
                    cluster_mesh = np.zeros_like(mesh_kpoints_index, dtype=bool)
                    
                    for act_coord in cluster_bn_i.active_coords:
                        idx_coord = tuple(act_coord) if Component.dimensions > 1 else act_coord
                        cluster_mesh[idx_coord] = True
                    
                    remaining_points_mesh = np.logical_not(cluster_mesh) * 1
                    if np.sum(remaining_points_mesh) > np.sum(cluster_mesh):
                        remaining_points_mesh = cluster_mesh * 1
                    # logger.debug(print_bool_array(remaining_points_mesh))

                    new_remaining_samples = []
                    for samples in remaining_samples:
                        new_remaining_samples.append([])
                        for sample in samples:
                            new_components = sample.remove_points(remaining_points_mesh)
                            new_remaining_samples[-1] += new_components

                    remaining_samples = new_remaining_samples

                    new_remaining_samples_ = []
                    for ri, samples in enumerate(remaining_samples_):
                        new_remaining_samples_.append([])
                        for sample in samples:
                            new_components = sample.remove_points(remaining_points_mesh)
                            new_remaining_samples_[-1] += new_components
                        remaining_samples[ri] += new_remaining_samples_[-1]

                    remaining_samples_ = [[] for _ in range(len(samples_that_can_be_added))]
                    logger.debug(f"New remaining samples after removing points:")
                    total_samples_can_merge = 0
                    for ri, samples in enumerate(remaining_samples):
                        total_can_merge = 0
                        infos_print = []
                        for sample in samples:
                            can_merge, info_print =  cluster_bn_i.can_merge(sample, verbose=True)
                            if can_merge:
                                total_can_merge += 1
                            else:
                                infos_print.append(info_print)
                        total_samples_can_merge += total_can_merge
                        logger.debug(f"Band {bn_i + ri + min_band}: {len(samples)} samples remaining. Can merge with cluster: {total_can_merge}")
                        info_print = set(infos_print)
                        for info in info_print:
                            logger.debug(f"  - {info}")
                    if total_samples_can_merge == 0:
                        # sort the remaining samples by number of nodes
                        remaining_samples[0] = [s for s in sorted(remaining_samples[0], key=lambda x: x.number_of_nodes, reverse=True)]
                        cluster_bn_i_swap = remaining_samples[0].pop(0).to_cluster()
                        logger.debug(f"Swapping cluster with a component of size {cluster_bn_i_swap.number_of_nodes} (was {cluster_bn_i.number_of_nodes})")
                        remaining_samples[0].append(cluster_bn_i)
                        cluster_bn_i = cluster_bn_i_swap

                logger.debug("No samples that can be added, but there are still points to be added")
                continue
            
            scores = scores[scores[:, 2].argsort()[::-1]]
            first_score = scores[0][2]

            #logger.info(f"[Band {bn_i}] Best score in this iteration: {first_score:.4f}")

            # ------------------------------------------------------------------------------

            number_of_available_points = cluster_bn_i.number_of_nodes

            for bn_j in bands[bn_i: max_band_energies[bn_i] + 1]:
                for sample in degenerate_components[bn_j]:
                    number_of_available_points += sample.number_of_nodes
                for sample in components[bn_j]:
                    number_of_available_points += sample.number_of_nodes

            for samples in remaining_samples:
                for sample in samples:
                    number_of_available_points += sample.number_of_nodes

            for samples in remaining_samples_:
                for sample in samples:
                    number_of_available_points += sample.number_of_nodes

            for i, j, score in scores:
                i = int(i)
                j = int(j)
                sample = samples_that_can_be_added[i][j]
                number_of_available_points += sample.number_of_nodes
                already_added = any([sample.does_intersect(s) for s in appended_samples])

                if already_added:
                    remaining_samples_[i].append(sample)
                    continue
                
                if first_score > 0.7 and score < first_score * 0.7 or first_score <= 0.7 and score < first_score:
                    remaining_samples[i].append(sample)
                    continue

                #if not np.isclose(score, first_score, atol=1e-3):
                #    remaining_samples[i].append(sample)
                #    continue

                appended_samples.append(sample)
                cluster_bn_i.merge(sample)

            if number_of_available_points != nks * (max_band_energies[bn_i] - bn_i + 1):
                logger.info(f"Number of available points: {number_of_available_points}")
                logger.info(f"Number of points in the cluster: {cluster_bn_i.number_of_nodes}")
                logger.info(f"Number of points in the band: {nks * (max_band_energies[bn_i] - bn_i + 1)}")
                raise ValueError(f"Number of available points is not equal to the number of points in the band")

        for i, samples in enumerate(remaining_samples_):
            remaining_samples[i] += samples

        for j, samples in enumerate(remaining_samples):
            components[bn_i + j] += samples

        if len(components[bn_i]) > 0 and bn_i + 1 <= max_band_energies[bn_i]:
            components[bn_i + 1] += components[bn_i]
        else:
            if len(components[bn_i]) > 0:
                logger.info()
                raise ValueError(f"Band {bn_i + min_band} has remaining points, but the next band is not in the range of the bands that are being solved")

            
        number_of_available_points = np.sum(cluster_bn_i.number_of_nodes)
        
        for samples in components[bn_i + 1: max_band_energies[bn_i] + 1]:
            for sample in samples:
                number_of_available_points += sample.number_of_nodes
        
        for samples in degenerate_components[bn_i: max_band_energies[bn_i] + 1]:
            for sample in samples:
                number_of_available_points += sample.number_of_nodes

        if number_of_available_points != nks * (max_band_energies[bn_i] - bn_i + 1):
            logger.info(f"Number of available points: {number_of_available_points}")
            logger.info(f"Number of points in the cluster: {cluster_bn_i.number_of_nodes}")
            logger.info(f"Number of missing points: {nks * (max_band_energies[bn_i] - bn_i + 1) - number_of_available_points}")
            logger.info(f"Number of points in the band: {nks * (max_band_energies[bn_i] - bn_i + 1)}")
            raise ValueError(f"Number of available points is not equal to the number of points in the band")

        clusters.append(cluster_bn_i)

        logger.info(
            f"[Band {bn_i + min_band}] Progress: {1:.2%} | The band is completed | Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if bn_i == max_band_energies[bn_i]:
            logger.info(f"All bands from the group {np.min(bands_to_solve) + min_band} - {bn_i + min_band} have been solved. | Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            bands_final_temp = np.full((nks, np.max(max_band_energies) + 1), -1, dtype=int)
            for cluster, bn in zip(clusters, bands_to_solve):
                for k, b in cluster.nodes:
                    bands_final_temp[k, bn] = b

            # Save the bands_final_temp to a file
            if not os.path.exists('temp'):
                os.makedirs('temp')
            np.save(f'temp/bands_final_temp_{np.min(bands_to_solve) + min_band}_{bn_i + min_band}.npy', bands_final_temp)
            logger.info(f"Temporary file temp/bands_final_temp_{np.min(bands_to_solve) + min_band}_{bn_i + min_band}.npy saved.")



    return clusters


def solver_algorithm(components, degenerate_components, degenerate_points, connections, neighbors, mesh_kpoints_index, nks, max_band_energies, mask_degenerates, bands, logger, n_process=1, verbose=False, parallel=True, energies=None, tolerance=0.99, alpha=0.5, min_band=0):
    """
    ----------------------------------------------------------------------------------------------------------------------
    Main algorithm for solving clusters of components across multiple bands, with support for parallel processing.
    ----------------------------------------------------------------------------------------------------------------------
    
    This function orchestrates the solving of clusters across multiple bands by utilizing the `solver_iterable` function. It 
    handles the preparation of bands, manages parallel processing for faster computation, and integrates with a logger to track 
    the progress. It also includes options for verbosity and customization of the process.
    """

    '''
    ----------------------------------------------------------------------------------------------------------------------
                                                        Main loop
    ----------------------------------------------------------------------------------------------------------------------
    '''
    clusters = []
    solver_iterable_partial = partial(solver_iterable, components=components, degenerate_components=degenerate_components, degenerate_points=degenerate_points, connections=connections, neighbors=neighbors, mesh_kpoints_index=mesh_kpoints_index, nks=nks, max_band_energies=max_band_energies, mask_degenerates=mask_degenerates, bands=bands, logger=logger, energies=energies, tolerance = tolerance, alpha=alpha, min_band=min_band)

    iterable_bands = []

    bn_i = 0
    while bn_i <= bands[-1]:
        iterable_bands.append(np.arange(bn_i, max_band_energies[bn_i] + 1))
        if max_band_energies[bn_i] == bands[-1]:
            break
        bn_i = max_band_energies[bn_i] + 1

    
    if n_process == 1 or not parallel:
        clusters = []
        for iterable in iterable_bands:
            clusters += solver_iterable_partial(iterable)
        return clusters, iterable_bands

    for iterable in iterable_bands:
        logger.debug("Group Bands: " + ", ".join(map(str, iterable)) + " Number of Components: " + ", ".join(map(lambda x:str(len(components[x])), iterable)))

    if verbose:
        clusters = process_map(solver_iterable_partial, iterable_bands, max_workers=n_process, desc='Clustering Points')
    else:
        with Pool(n_process) as pool:
            clusters = pool.map(solver_iterable_partial, iterable_bands)

    clusters = [item for sublist in clusters for item in sublist]
    return clusters, iterable_bands


'''
----------------------------------------------------------------------------------------------------------
Not Used but implemented to otimized previous algorithms:
----------------------------------------------------------------------------------------------------------
'''


#class Graph_Method:
#    def __init__(self, nodes: np.ndarray, number_ks: int) -> None:
#        self.graph = nx.Graph()
#        self.graph.add_nodes_from(nodes)
#
#        self.nodes_index = nodes
#        self.k_values = nodes % number_ks
#        self.number_ks = number_ks
#
#        self.number_of_components = int(np.max(nodes // number_ks) + 1)
#    
#    def create_edges(self, edges) -> None:
#        # only add the edges if the nodes are already in the graph
#        for i, j, weight in edges:
#            self.graph.add_edge(i, j, weight=weight)
#
#    def find_points_with_forbiden_paths(self, degenerates):
#        '''
#        This function is not being used, it was an improvement for previous versions but it is not present in the current implementation
#        It remains here If required for future improvements.
#        '''
#        points_that_needs_correction = []
#
#        for i, j in degenerates:
#            if not nx.has_path(self.graph, i, j):
#                weights_of_node_i = [self.graph.get_edge_data(i, k)['weight'] for k in self.graph.neighbors(i)]
#                weights_of_node_j = [self.graph.get_edge_data(j, k)['weight'] for k in self.graph.neighbors(j)]
#
#                if np.all(np.isclose(weights_of_node_i, DEGENERATE_TOLERANCE, atol=1e-2)):
#                    points_that_needs_correction.append(i)
#                
#                if np.all(np.isclose(weights_of_node_j, DEGENERATE_TOLERANCE, atol=1e-2)):
#                    points_that_needs_correction.append(j)
#
#                continue
#            
#            print(f"Path between {i} and {j} found: {i % self.number_ks} bands {i // self.number_ks}, {j // self.number_ks}")
#
#            edges_to_remove = []
#            for k in self.graph.neighbors(i):
#                band_i = i // self.number_ks
#                band_k = k // self.number_ks
#                if band_i != band_k:
#                    edges_to_remove.append((i, k))
#            
#            self.graph.remove_edges_from(edges_to_remove)
#
#            edges_to_remove = []
#            for k in self.graph.neighbors(j):
#                band_j = j // self.number_ks
#                band_k = k // self.number_ks
#                if band_j != band_k:
#                    edges_to_remove.append((j, k))
#
#            self.graph.remove_edges_from(edges_to_remove)
#
#        
#        return points_that_needs_correction
#
#    def get_communities(self):
#        '''
#        This function is not being used, it was an improvement for previous versions but it is not present in the current implementation
#        It remains here If required for future improvements.
#        '''
#
#        number_of_connected_components = nx.number_connected_components(self.graph)
#
#        communities = nx.algorithms.community.greedy_modularity_communities(self.graph, weight='weight')
#
#        communities = [list(comm) for comm in communities]
#        print(f"Number of connected components: {number_of_connected_components}")
#        print(f"Number of communities: {len(communities)}")
#
#        self.number_of_more_than_ks = 0
#        sub_graphs = []
#
#        def get_subgraph(communitiy):
#            sub_grap = self.graph.subgraph(communitiy)
#            if sub_grap.number_of_nodes() <= self.number_ks:
#                return [sub_grap]
#            
#            self.number_of_more_than_ks += 1
#            new_communities = nx.algorithms.community.greedy_modularity_communities(sub_grap, weight='weight')
#            sub_graphs = []
#            for comm in new_communities:
#                sub_graphs += get_subgraph(comm)
#            return sub_graphs
#        
#        for comm in communities:
#            sub_graphs += get_subgraph(comm)
#        
#        print(f"Number of subgraphs with more than ks nodes: {self.number_of_more_than_ks}")
#        print(f"Number of final subgraphs: {len(sub_graphs)}")
#        
#        return sub_graphs



def evaluate_point(dimension:int, k, bn, k_index: np.ndarray, k_matrix: np.ndarray, bands: np.ndarray, energies: np.ndarray):
    '''
    Assign a signal value depending on energy continuity.

    Parameters
        dimension: int
            Dimension of the problem.
        k: Kpoint
            Integer that index the k point on analysis.
        bn: Band
            Integer that index the band number on analysis.
        k_index: array_like
            An array that contains the indices of each k point on the k-space matrix.
        k_matrix: array_like
            An array with the shape of the k-space. 
            It contains the value of each k point in their corresponding position.
        signal: array_like
            An array with the current signal value for each k point.
        bands: array_like
            An array with the information of current solution of band clustering.
        energies: array_like
            It contais the energy value for each k point.
    
    Returns
        (signal, scores): Tuple[int, list[int]]
    '''
    
    TOL = 0.9                       # Tolerance to consider that exist energy continuity
    N = 4                           # Number of points to fit the curve
    N_NEIGS = 2 * dimension         # Number of neighbors to consider the continuity

    mach_bn = bands[k, bn]          # original band

    if dimension == 1:
        ik = k_index[k]             # k point index on k-space
    elif dimension == 2:
        ik, jk = k_index[k]         # k point indices on k-space
    else:
        ik, jk, kk = k_index[k]     # k point indices on k-space

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


    if dimension == 1:
        directions = np.array([[1], [-1]])                     # Right, Left
        energy_vals = []

        ###########################################################################
        # Calculate the score for each direction
        ###########################################################################

        for direction in directions:
            # Iterates each direction and obtain N points to be used for fit the curve
            n = np.repeat(np.arange(1,N+1),2).reshape(N,2)
            kn_index = n*direction + np.array([ik])
            i = kn_index[:, 0]
            i = i[i >= 0]
            i = i[i < k_matrix.shape[0]]

            ks = k_matrix[i] if len(i) > 0 else []               # Identify the N k points
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
            X = i
            new_x = ik
            pol = lambda x, a, b, c: a*x**2 + b*x + c           # Second order polynomial
            popt, pcov = curve_fit(pol, X, Es)                  # Curve fitting
            Enew = pol(new_x, *popt)                            # Obtain Energy value
            energy_vals.append(difference_energy(Ek, Enew))     # Calculate score

    elif dimension == 2:

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
    
    else:
            
        directions = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]) # Down, Right, Up, Left, Front, Back
        energy_vals = []

        ###########################################################################
        # Calculate the score for each direction
        ###########################################################################

        for direction in directions:
            # Iterates each direction and obtain N points to be used for fit the curve
            n = np.repeat(np.arange(1,N+1),3).reshape(N,3)
            kn_index = n*direction + np.array([ik, jk, kk])
            i, j, k = kn_index[:, 0], kn_index[:, 1], kn_index[:, 2]
            flag_i = len(np.unique(i)) > 1
            flag_j = len(np.unique(j)) > 1

            if flag_i:
                i = i[i >= 0]
                i = i[i < k_matrix.shape[0]]
                j = np.full(len(i), j[0])
                k = np.full(len(i), k[0])
            elif flag_j:
                j = j[j >= 0]
                j = j[j < k_matrix.shape[1]]
                i = np.full(len(j), i[0])
                k = np.full(len(j), k[0])
            else:
                k = k[k >= 0]
                k = k[k < k_matrix.shape[2]]
                i = np.full(len(k), i[0])
                j = np.full(len(k), j[0])

            ks = k_matrix[i, j, k] if len(i) > 0 else []   # Identify the N k points
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
            X = i if flag_i else j if flag_j else k
            new_x = ik if flag_i else jk if flag_j else kk
            pol = lambda x, a, b, c: a*x**2 + b*x + c           # Second order polynomial
            popt, pcov = curve_fit(pol, X, Es)                  # Curve fitting
            Enew = pol(new_x, *popt)                            # Obtain Energy value
            energy_vals.append(difference_energy(Ek, Enew))     # Calculate score


    energy_vals = np.array(energy_vals)
    score = np.median(energy_vals) if len(energy_vals) > 0 else 0
    min_score = np.min(energy_vals) if len(energy_vals) > 0 else 0

    if score >= TOL and min_score >= 0.6:
        return 2, score
    return 0, score
