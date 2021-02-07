import networkx as nx
import numpy as np
from network import Network
from utils import get_items, dict_to

class PhysicalNetwork(Network):
    def __init__(self, incoming_graph_data=None, num_nodes=None, **kwargs):
        super(PhysicalNetwork, self).__init__(incoming_graph_data, **kwargs)
        # Initialize the topology
        if num_nodes is not None:
            alpha = kwargs.get('alpha', 0.5)
            beta = kwargs.get('beta', 0.2)
            self.init(num_nodes, alpha, beta)

    def init(self, num_nodes, alpha=0.5, beta=0.2):
        self.generate_topology(num_nodes, strategy='waxman', alpha=alpha, beta=beta)

    def find_candidate_nodes(self, attrs=None, req_values=[], filter=[], rtype='id'):
        r"""
        Find candicate nodes according to the restrictions and filter.

        Args:
            req_dict
        Returns:

        """
        items = get_items(attrs, req_values)
        if rtype == 'id':
            candidate_nodes = list(self.nodes)
            for name, req in items:
                suitable_nodes = np.where(self.get_node_attr(name, rtype='array') >= req)
                candidate_nodes = np.intersect1d(candidate_nodes, suitable_nodes)
            candidate_nodes = np.setdiff1d(candidate_nodes, filter)
        elif rtype == 'bool':
            candidate_nodes = np.ones(self.num_nodes)
            for  name, req in items:
                suitable_nodes = np.where(self.get_node_attr(name, rtype='array') >= req, True, False)
                candidate_nodes = np.logical_and(candidate_nodes, suitable_nodes)
            candidate_nodes[filter] = False
        return candidate_nodes

class BatchPhysicalNetwork(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.batch = []
    
if __name__ == '__main__':
    pass