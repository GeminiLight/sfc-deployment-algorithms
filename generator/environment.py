import os
import sys
import copy
from abc import abstractmethod, ABCMeta

file_path_dir = os.path.abspath('.')
if file_path_dir not in sys.path:
    sys.path.append(file_path_dir)

from generator.physical_network import PhysicalNetwork


class Environment(metaclass=ABCMeta):
    def __init__(self):
        super(Environment, self).__init__()
        # physical network
        self.pn = PhysicalNetwork()
        self.pn_backup = copy.deepcopy(self.pn)
        # record
        self.total_revenue = 0
        self.total_cost = 0
        self.success = 0
        self.inservice = 0

    def reset(self):
        """reset the environment"""
        self.pn = PhysicalNetwork()
        self.pn_backup = copy.deepcopy(self.pn)
        self.total_revenue = 0
        self.total_cost = 0
        self.success = 0
        self.inservice = 0
        return True

    def ready(self):
        """ready to embace a new sfc"""
        self.pn_backup = copy.deepcopy(self.pn)

    @abstractmethod
    def step(self, vn):
        """Agent interacts with Environment
        
        An abstract method which must be implemented

        Exmple:
        SUCCESS:
            self.pn_backup = copy.deepcopy(self.pn)
            self.inservice += 1
            self.success += 1
            self.total_revenue += vn.revenue
            self.total_cost += vn.cost
        FAILURE:
            self.pn = copy.deepcopy(self.pn_backup)
            return False
        """
        pass
        
    def release_resources(self, vn):
        """release its resources when a vn leaves """
        if len(vn.slots) == 0:
            return True
        for vid, pid in vn.slots.items():
            self.pn.update_node(pid, 'cpu_free', vn.cpu_data[vid])
            self.pn.update_node(pid, 'ram_free', vn.ram_data[vid])
            self.pn.update_node(pid, 'rom_free', vn.rom_data[vid])
        for eid, path in vn.paths.items():
            if len(path)==1:
                continue
            bw_req = vn.graph.edges[eid]['bw']
            self.pn.update_bw_with_path(path, bw_req)  # (path, bw_req)
        self.pn_backup = copy.deepcopy(self.pn)
        self.inservice -= 1
        return True


if __name__ == '__main__':
    pass
    

