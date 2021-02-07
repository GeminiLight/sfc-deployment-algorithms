import ast
import numpy as np
import pandas as pd
from service_fuction_chain import ServiceFunctionChain

class SfcSimulator(object):
    def __init__(self, num_sfcs, mode='single', **kwargs):
        """A simulator of SFC"""
        super(SfcSimulator, self).__init__()
        self.mode = mode
        self.num_sfcs = num_sfcs
        self.sfcs = []
        self.events = []
        self.set_config(kwargs)

    def set_config(self, config):
        # the node and edge attributes of sfcs
        self.node_attrs = config.get('node_attrs', ['cpu', 'ram', 'rom'])
        self.edge_attrs = config.get('edge_attrs', ['bw'])
        # the minximum and maximum of sfcs' length
        self.min_length = config.get('min_length', 2)
        self.max_length = config.get('max_length', 10)
        # the minximum and maximum of the node attributes' value of sfcs
        self.min_node_value = config.get('min_node_value', 2)
        self.max_node_value = config.get('max_node_value', 30)
        # the minximum and maximum of the edge attributes' value of sfcs
        self.min_edge_value = config.get('min_edge_value', 2)
        self.max_edge_value = config.get('max_edge_value', 30)
        # the arverge arrival rate of sfcs
        self.aver_arrival_rate = config.get('aver_arrival_rate', 20)
        # the lifetime of sfcs
        self.aver_lifetime = config.get('aver_lifetime', 500)

    def renew(self, sfcs=True, events=True):
        if sfcs == True:
            self.renew_sfcs()
        if events == True:
            self.renew_events()
        return self.sfcs, self.events

    def renew_sfcs(self):
        self.arrange_sfcs()
        for i in range(self.num_sfcs):
            sfc = ServiceFunctionChain(num_nodes=self.sfcs_length[i], id=i,
                                        arrival_time=self.sfcs_arrival_time[i],
                                        lifetime=self.sfcs_lifetime[i])
            sfc.generate_data(self.node_attrs, self.edge_attrs,
                                self.min_node_value, self.max_node_value,
                                self.min_edge_value, self.max_edge_value)
            self.sfcs.append(sfc)
        return self.sfcs

    def renew_events(self):
        arrival_list = [{'sfc_id': sfc.id, 'time': sfc.arrival_time, 'type': 1} for sfc in self.sfcs]
        leave_list = [{'sfc_id': sfc.id, 'time': sfc.arrival_time+sfc.lifetime, 'type': 0} for sfc in self.sfcs]
        event_list = arrival_list + leave_list
        self.events = sorted(event_list, key=lambda e: e.__getitem__('time'))
        return self.events

    def arrange_sfcs(self):
        self.sfcs_length = np.random.randint(self.min_length, self.max_length, self.num_sfcs)
        self.sfcs_lifetime = np.random.exponential(self.aver_lifetime, self.num_sfcs)
        self.sfcs_arrival_time = np.random.poisson(self.aver_arrival_rate, self.num_sfcs)

    def to_csv(self, sfc_path, event_path):
        sfc_list = []
        for sfc in self.sfcs:
            nodes_data = {n_attr: sfc.get_node_attr(n_attr) for n_attr in self.node_attrs}
            edges_data = {e_attr: sfc.get_edge_attr(e_attr) for e_attr in self.edge_attrs}
            sfc_info = {
                'id': sfc.id,
                'num_node': sfc.length,
                'lifetime': sfc.lifetime,
                'arrival_time': sfc.start_time,
                'nodes_data': nodes_data,
                'edges_data': edges_data,
            }
            '''Include other information'''
        pd_vns = pd.DataFrame.from_dict(sfc_list)
        pd_vns.to_csv(sfc_path)
        pd.DataFrame(self.events).to_csv(event_path)

    def from_csv(self, sfc_path, event_path):
        pass

if __name__ == '__main__':
    pass