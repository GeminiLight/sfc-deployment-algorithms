import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp


class PhysicalNetwork(object):
    def __init__(self):
        # Load the data of network
        self.raw_nodes_data = pd.read_csv(
            'data/pn/nodes_data.csv', index_col=0)
        self.raw_edges_data = pd.read_csv(
            'data/pn/edges_data.csv', index_col=0)

        # The number of nodes and edges
        self.nodes_num = len(self.raw_nodes_data)  # the number of nodes
        self.edges_num = len(self.raw_edges_data)  # the number of edges

        # Constract to graph
        self.graph = self.data_to_graph()
        self.sparse_adjacency_matrix = self.get_sparse_adjacency_matrix()

        # Invariant edges information
        self.edges_list = np.array(
            [self.raw_edges_data['v_i'], self.raw_edges_data['v_j']], dtype=np.int32)
        self.edges_pair = list(self.graph.edges)

        # Statistic of cpu and betweeness
        self.node_benchmark = 100
        self.edge_benchmark = self.raw_nodes_data['bw_sum_max'].max()
        self.max_cpu_resource = self.raw_nodes_data['cpu_max'].max()
        self.max_rom_resource = self.raw_nodes_data['rom_max'].max()
        self.max_bw_resource = self.raw_edges_data['bw_max'].max()
        self.max_bw_sum_resource = self.raw_nodes_data['bw_sum_max'].max()
        # self.sum_nodes_cqu_max = self.raw_nodes_data['cpu_max'].sum()  # Return the sum of every node's cqu resource max value
        # self.sum_nodes_cqu_free = self.raw_nodes_data['cpu_free'].sum()  # Return the sum of every node's cqu resource free value
        # self.sum_edges_bw_max = self.raw_edges_data['bw_max'].sum()  # Return the sum of every node's bw resource max value
        # self.sum_edges_bw_free = self.raw_edges_data['bw_free'].sum()  # Return the sum of every node's bw resource free value

    # Initialization
    def data_to_graph(self):
        """Convert raw data to networkx graph"""
        # create physical graph
        G = nx.Graph()
        # add nodes attrs to graph
        nodes = self.raw_nodes_data.astype("int")
        nodes_list = [(n,
                       {'cpu_max': nodes['cpu_max'][n],
                        'cpu_free': nodes['cpu_free'][n],
                        'ram_max': nodes['ram_max'][n],
                        'ram_free': nodes['ram_free'][n],
                        'rom_max': nodes['rom_max'][n],
                        'rom_free': nodes['rom_free'][n],
                        'bw_sum_max': nodes['bw_sum_max'][n],
                        'bw_sum_free': nodes['bw_sum_free'][n]}) for n in range(self.nodes_num)]
        G.add_nodes_from(nodes_list)
        # add edges attrs to graph
        edges = self.raw_edges_data.astype("int")
        edges_list = [(edges['v_i'][e], edges['v_j'][e],
                       {'weight': edges['weight'][e],
                        'bw_max': edges['bw_max'][e],
                        'bw_free': edges['bw_free'][e]}) for e in range(self.edges_num)]
        G.add_edges_from(edges_list)
        return G

    def get_sparse_adjacency_matrix(self, format='csr'):
        '''Get the sparse adjacency matrix of graph'''
        A = nx.to_scipy_sparse_matrix(self.graph, format='csr')
        I = sp.sparse.identity(self.nodes_num, format='csr')
        return A + I

    ### data of nodes and edges ###
    @property
    def nodes(self):
        '''Return the information of nodes'''
        nodes_attrs = self.graph.nodes.data()
        nodes = [(n[1]['cpu_max'], n[1]['cpu_free'], n[1]['ram_max'], n[1]['ram_free'], n[1]['rom_max'], n[1]['rom_free'],
                  n[1]['bw_sum_max'], n[1]['bw_sum_free']) for n in nodes_attrs]
        return np.array(nodes, dtype=np.float32)

    @property
    def norm_nodes_data(self):
        '''Return the normalized information of nodes

        Normalize the attributions of nodes 
        Using the max_cpu_resource and max_bw_sum_resource as benchmarks
        '''
        norm_cpu = self.cpu_free_data / self.node_benchmark
        norm_ram = self.ram_free_data / self.node_benchmark
        norm_rom = self.rom_free_data / self.node_benchmark
        norm_bw_sum = self.bw_sum_free_data / self.edge_benchmark
        norm_nodes_data = np.vstack([norm_cpu, norm_ram, norm_rom, norm_bw_sum])
        return norm_nodes_data

    @property
    def edges(self):
        '''Return the information of edges'''
        edges_attrs = self.graph.edges.data()
        edges = [(e[0], e[1], e[2]['weight'], e[2]['bw_max'],
                  e[2]['bw_free']) for e in edges_attrs]
        return np.array(edges, dtype=np.float32)

    @property
    def state(self):
        '''Return the normalized state of physical network

        state = [norm_nodes_data, sparse_adjacency_matrix]
        It's the input of GCN layer where the features of nodes will be extracted
        '''
        return [self.norm_nodes_data, self.sparse_adjacency_matrix]

    @property
    def bw_free_data(self):
        edges_attrs_data = self.graph.edges.data('bw_free')
        bw_free_data = np.array([e[2] for e in edges_attrs_data])
        return bw_free_data

    @property
    def cpu_free_data(self):
        nodes_attrs_data = self.graph.nodes.data()
        cpu_free_data = np.array([n[1]['cpu_free'] for n in nodes_attrs_data])
        return cpu_free_data

    @property
    def ram_free_data(self):
        nodes_attrs_data = self.graph.nodes.data()
        ram_free_data = np.array([n[1]['ram_free'] for n in nodes_attrs_data])
        return ram_free_data

    @property
    def rom_free_data(self):
        nodes_attrs_data = self.graph.nodes.data()
        rom_free_data = np.array([n[1]['rom_free'] for n in nodes_attrs_data])
        return rom_free_data

    @property
    def bw_sum_free_data(self):
        nodes_attrs_data = self.graph.nodes.data()
        bw_sum_free_data = np.array([n[1]['bw_sum_free']
                                     for n in nodes_attrs_data])
        return bw_sum_free_data

    # statistic
    def sum_free_data(self):
        nodes_attrs_data = self.graph.nodes.data()
        nodes_free_data = np.array([[n[1]['cpu_free'], n[1]['ram_free'], n[1]['rom_free']] for n in nodes_attrs_data])
        sum_nodes_free_data = nodes_free_data.sum(axis=(0, 1))
        sum_edges_free_data = sum(self.bw_free_data)
        return sum_nodes_free_data + sum_edges_free_data

    ### Find shortest path ###
    def find_simple_paths(self, a, b):
        """Find the shortest simple paths of node a and b"""
        try:
            simple_paths = nx.shortest_simple_paths(self.graph, a, b)
            return simple_paths
        except:
            return []

    def find_shortest_path(self, a, b):
        """Find the shortest path (dijkstra) of node a and b"""
        try:
            shortest_path = nx.dijkstra_path(self.graph, a, b)
            return shortest_path
        except:
            return False

    ### Find candidate nodes ###
    def find_candidate_nodes(self, cpu_req=0, ram_req=0, rom_req=0, rtype='id', fiter_list=[]):
        if rtype == 'id':
            candidate_nodes_list = np.where((self.cpu_free_data >= cpu_req) & 
                    (self.ram_free_data >= ram_req) & (self.rom_free_data >= rom_req))
            candidate_nodes_list = np.setdiff1d(candidate_nodes_list, fiter_list)
        elif rtype == 'bool':
            cpu_subset = np.where(self.cpu_free_data >= cpu_req, True, False)
            ram_subset = np.where(self.ram_free_data >= ram_req, True, False)
            rom_subset = np.where(self.rom_free_data >= rom_req, True, False)
            candidate_nodes_list = np.logical_and(np.logical_and(cpu_subset, ram_subset), rom_subset)
            candidate_nodes_list[fiter_list] = False
        return candidate_nodes_list

    def is_candidate_node(self, nid, cpu_req=0, ram_req=0, rom_req=0):
        cpu_free = self.graph.nodes[nid]['cpu_free']
        ram_free = self.graph.nodes[nid]['ram_free']
        rom_free = self.graph.nodes[nid]['rom_free']
        if (cpu_free >= cpu_req) and (ram_free >= ram_req) and (rom_free >= rom_req):
            return True
        else:
            return False

    ### Update physical network ###
    def update_node(self, nid, attr, x):
        assert self.graph.nodes[nid][attr]+x >= 0
        self.graph.nodes[nid][attr] += x

    def update_edge(self, v_i, v_j, attr, x):
        assert self.graph[v_i][v_j][attr]+x >= 0
        self.graph[v_i][v_j][attr] += x

    def update_bw_with_path(self, path, x):
        """Update the bandwidth of PN with path selected by VN"""
        if path == None or len(path) == 1:
            return True
        for i in range(len(path)-1):
            self.graph[path[i]][path[i+1]]['bw_free'] += x
            self.graph.nodes[path[i]]['bw_sum_free'] += x
        self.graph.nodes[path[len(path)-1]]['bw_sum_free'] += x

    def update_bw_with_path_edges(self, path_edges, x):
        path = self.path_edges_to_path(path_edges)
        self.update_bw_with_path(path, x)

    def get_edge_id(self, i, j):
        if i > j:
            i, j = j, i
        return self.edges_pair.index((i, j))

    ### For GRC Algorithm ###
    def calc_grc_c(self):
        '''Return the data of free cpu resource of nodes'''
        nodes_attrs_data = self.graph.nodes.data()
        cpu_free_data = np.array([n[1]['cpu_free'] for n in nodes_attrs_data])
        rom_free_data = np.array([n[1]['rom_free'] for n in nodes_attrs_data])
        sum_nodes_data = sum(cpu_free_data) + sum(rom_free_data)
        return (cpu_free_data + rom_free_data) / sum_nodes_data

    def calc_grc_M(self):
        M = nx.attr_sparse_matrix(
            self.graph, edge_attr='bw_free', normalized=True, rc_order=self.graph.nodes).T
        return M

    ### reset ###
    def reset(self):
        pass


if __name__ == '__main__':
    pn = PhysicalNetwork()
    print(pn.find_candidate_nodes(cpu_req=90))
    print(pn.find_candidate_nodes(cpu_req=90, rtype='bool'))
    '''test get sum resource'''
    # print(pn.sum_nodes_cqu_max)
    # pn.update_node(0, 'cpu_free', 20)
    # print(pn.sum_nodes_cqu_free)
    # print(pn.sum_edges_bw_max)
    # pn.update_edge(0, 'bw_free', 20)
    # print(pn.sum_edges_bw_free)

    # tf.Tensor([82. 17. 27. 63.], shape=(4,), dtype=float32)
    # [True, True, True, True]
    # vnf_id: 2
    # tf.Tensor([ 6. 25. 78.  9.], shape=(4,), dtype=float32)
    # [True, True, True, True]

    '''test edge set'''
    # print(pn.get_edge_id(50, 45))
    # print(pn.get_edge_id(63, 80))

    # print(pn.edges)

    # start = time.time()
    # for i in range(10):
    #     pn.edges
    # end = time.time()
    # print('Running time: %s Seconds' %(float(end-start)))
    # paths = pn.find_simple_paths(0, 10)
    # for path in paths:
    #     print(path)

    '''test resource'''
    # start = time.time()
    # for i in range(10):
    #     drop_egdes = [(u, v) for (u, v, bw_free) in pn.graph.edges.data('bw_free') if bw_free < 80]
    # end = time.time()
    # print('Running time: %s Seconds' %(float(end-start)))
    # print(pn.max_cpu_resource)
    # print(pn.max_bw_sum_resource)

    '''test GCN'''
    # from spektral.layers import GraphConv
    # A = pn.sparse_adjacency_matrix
    # print(A)
    # D = GraphConv.preprocess(A).astype('f4')
    # print(D)
    # print(A[99, 12], A[12, 99])
    # D = nx.pagerank(pn.graph)
    # print(D)
    # paths_edges = [[(path[i], path[i+1]) for i in range(len(path)-1)] for path in paths]
    # print(paths_edges)

    '''test GRC'''
    # print(pn.grc_rank(type='dict'))
