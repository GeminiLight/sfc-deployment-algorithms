import os
import sys
import abc
import copy
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt

file_path_dir = os.path.abspath('.')
if file_path_dir not in sys.path:
    sys.path.append(file_path_dir)

from generator.physical_network import PhysicalNetwork
from algo.grc.mapping import grc_rank, node_mapping, link_mapping
from generator.environment import Environment

class GRCEnv(object):
    """The environment for GRC appoarch."""
    def __init__(self, grc_d=0.95):
        super(GRCEnv, self).__init__()
        self.grc_d = grc_d


    def step(self, vn):
        """Agent interacts with Environment"""
        vn_pn_slots = node_mapping(vn, self.pn, d=self.grc_d)
        # FAILUER: Node Mapping
        if vn_pn_slots==False:
            self.pn = copy.deepcopy(self.pn_backup)
            return False
        vn_pn_paths = link_mapping(vn, self.pn, vn_pn_slots)
        # FAILUER: Edge Mapping
        if vn_pn_paths==False:
            self.pn = copy.deepcopy(self.pn_backup)
            return False
        # SUCCESS
        self.pn_backup = copy.deepcopy(self.pn)
        self.inservice += 1
        self.success += 1
        self.total_revenue += vn.revenue
        self.total_cost += vn.cost
        return True

if __name__ == '__main__':
    # new a env
    pass
    