import networkx as nx
import numpy as np
from network import Network

class ServiceFunctionChain(Network):
    def __init__(self, incoming_graph_data=None, num_nodes=None, **kwargs):
        super(ServiceFunctionChain, self).__init__(incoming_graph_data, **kwargs)
        if num_nodes is not None:
            self.init(num_nodes)

    def init(self, num_nodes):
        self.generate_topology(num_nodes, stategy='path')


if __name__ == '__main__':
    pass