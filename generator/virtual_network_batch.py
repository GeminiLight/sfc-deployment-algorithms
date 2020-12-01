import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx

file_path_dir = os.path.abspath('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(file_path_dir)

from generator.virtual_network import VirtualNetwork


class VirtualNetworkBatch(object):
    """Virtual Network Class

    Args:
        graph
        data
        state
        nodes_num 
        edges_num
        nodes_data: [cpu, rom]
        edges_data: [bw]
        attr_data [cpu | rom | bw]

    For GRC Algo:
        grc_c
        grc_M
    """
    def __init__(self, batch_size, min_length=2, max_length=15, min_request=2, max_request=30, vns_length=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.vns_length = vns_length
        self.min_length = min_length
        self.max_length = max_length
        self.min_request = min_request
        self.max_request = max_request
        # other
        self.id = kwargs.pop('id', 0)
        self.arrive_interval = kwargs.pop('arrive_interval', 0)
        self.aver_lifetime = kwargs.pop('aver_lifetime', 400)
        self.kwargs = kwargs
        self.vns = []
        if kwargs.pop('random', True):
            self.add_random_data()

    def add_random_data(self):
        if self.vns_length == None:
            self.vns_length = np.random.randint(self.min_length, self.max_length, size=(self.batch_size))
        self.add_other_attrs()
        for i in range(self.batch_size):
            vn = VirtualNetwork(self.vns_length[i], self.min_request, self.max_request)
            self.vns.append(vn)

    def add_other_attrs(self):
        self.lifetime = np.random.exponential(self.aver_lifetime)
        self.start_time = np.random.uniform(100) + self.arrive_interval * 100
        self.end_time = self.lifetime + self.start_time

    def load_from_pd(self, vn_data, batch=False):
        self.id = vn_data['id']
        self.vns_length = np.array([vn_data['nodes_num']] * self.batch_size)
        self.lifetime = vn_data['lifetime']
        self.start_time = vn_data['start_time']
        self.end_time = vn_data['end_time']
        for i in range(self.batch_size):
            vn = VirtualNetwork(random=False)
            vn.load_from_pd(vn_data)
            self.vns.append(vn)

    ### data ###
    @property
    def data(self):
        data = []
        for vn in self.vns:
            vn_data = vn.data
            pad = self.max_length - vn_data.shape[-1]
            data.append(np.pad(vn_data, ((0,0), (0,pad)), 'constant'))
        return np.array(data)

    @property
    def state(self):
        state = []
        for vn in self.vns:
            vn_state = vn.state
            pad = self.max_length - vn_state.shape[-1]
            state.append(np.pad(vn_state, ((0,0), (0,pad)), 'constant'))
        return np.array(state)

    @property
    def revenue(self):
        return np.array([vn.revenue for vn in self.vns])

    @property
    def cost(self):
        return np.array([vn.cost for vn in self.vns])

class VirtualNetworksBatchSimulator():
    def __init__(self, vns_num=100, batch_size=64, min_length=2, max_length=15, min_request=2, max_request=30,
                node_benchmark=100, edge_benchmark=1403, random_seed=1024, arrival_rate=20, aver_lifetime=400):
        self.vns_num = vns_num
        self.batch_size = batch_size
        self.min_length = min_length
        self.max_length = max_length
        self.min_request = min_request
        self.max_request = max_request
        self.node_benchmark = node_benchmark
        self.edge_benchmark = edge_benchmark
        self.arrival_rate = arrival_rate
        self.aver_lifetime = aver_lifetime
        self.vns_batch = []
        self.events = []

    def renew_vns_batch(self):
        '''Renew the vn information'''
        # self.vns_length = np.random.randint(self.min_length, self.max_length, self.vns_num)
        vnb_id = 0
        for i in range(int(np.ceil(self.vns_num/self.arrival_rate))):
            for j in range(self.arrival_rate):
                if vnb_id >= self.vns_num:
                    break
                vnb = VirtualNetworkBatch(self.batch_size, self.min_length, self.max_length, self.min_request, self.max_request, id=vnb_id, arrive_interval=i, aver_lifetime = self.aver_lifetime, random=True)
                self.vns_batch.append(vnb)
                vnb_id += 1
        return self.vns_batch

    def renew_events(self):
        '''Renew the occur events accordding to the vn information'''
        vn_start_list = [[vn.id, vn.start_time] for vn in self.vns_batch]
        vn_end_list = [[vn.id, vn.end_time] for vn in self.vns_batch]
        arrive_events = pd.DataFrame(vn_start_list, columns=['vn_id', 'time'])
        arrive_events['type'] = 1
        leave_events = pd.DataFrame(vn_end_list, columns=['vn_id', 'time'])
        leave_events['type'] = 0
        self.events = pd.concat([arrive_events, leave_events], ignore_index=True)
        self.events.sort_values("time", inplace=True)
        self.events.reset_index(inplace=True, drop=True)
        return self.events

    def load_from_csv(self, path='data/vn/', vns_data_file='vns_data.csv', events_data_file='events_data.csv'):
        vns_data = pd.read_csv(path + vns_data_file, index_col=0, header=0)
        self.events = pd.read_csv(path + events_data_file, index_col=0, header=0)
        self.vns_num = len(vns_data)
        for i in range(self.vns_num):
            vn_batch = VirtualNetworkBatch(self.batch_size, random=False)
            vn_data = vns_data.loc[i]
            vn_batch.load_from_pd(vn_data)
            self.vns_batch.append(vn_batch)


if __name__ == '__main__':
    vn = VirtualNetworkBatch(3)
    print(vn.data.shape)
    print(vn.state.shape)
    # print(vn.rom_data)
    # print(vn.calc_grc_c())
    # print(vn.calc_grc_M().todense())

    vns_simulator = VirtualNetworksBatchSimulator(100, batch_size=64)
    # vns_simulator.renew_vns_batch()
    vns_simulator.load_from_csv()

    # vns_simulator.renew_vns()
    # vns_simulator.renew_events()
    # vns_simulator.save()
    print(vns_simulator.vns_batch[0].data.shape)
    # vns_simulator.load()