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
    def __init__(self, batch_size, min_length=2, max_length=10, min_request=2, max_request=30, vn_length=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.vn_length = vn_length
        self.min_length = min_length
        self.max_length = max_length
        self.min_request = min_request
        self.max_request = max_request
        self.kwargs = kwargs
        self.vns = []
        self.add_random_data()

    def add_random_data(self):
        if self.vn_length == None:
            self.vn_length = np.random.randint(self.min_length, self.max_length, size=(self.batch_size))
        for i in range(self.batch_size):
            vn = VirtualNetwork(self.vn_length[i], self.min_request, self.max_request)
            self.vns.append(vn)

    def add_other_attrs(self, **kwargs):
        id = kwargs.pop('id', 0)
        arrive_interval = kwargs.pop('arrive_interval', 0)
        self.id = id
        self.lifetime = np.random.exponential(500)
        self.start_time = np.random.uniform(100) + arrive_interval * 100
        self.end_time = self.lifetime + self.start_time


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

class VirtualNetworksBatchSimulator():
    def __init__(self, vns_num, batch_size, min_length=2, max_length=10, min_request=2, max_request=30,
                cpu_benchmark=100, bw_benchmark=1500, random_seed=1024, arrival_rate=4):
        self.vns_num = vns_num
        self.batch_size = batch_size
        self.min_length = min_length
        self.max_length = max_length
        self.min_request = min_request
        self.max_request = max_request
        self.arrival_rate = arrival_rate
        self.vns_batch = []
        self.events = []

    def renew_vns_batch(self):
        '''Renew the vn information'''
        # self.vns_length = np.random.randint(self.min_length, self.max_length, self.vns_num)
        vnb_id = 0
        for i in range(int(np.ceil(self.vns_num/4))):
            for j in range(self.arrival_rate):
                if vnb_id >= self.vns_num:
                    break
                vnb = VirtualNetworkBatch(self.batch_size, self.min_length, self.max_length, self.min_request, self.max_request, id=vnb_id, arrive_interval=i, random=True)
                self.vns_batch.append(vnb)
                vnb_id += 1
        return self.vns

    def renew_events(self):
        '''Renew the occur events accordding to the vn information'''
        vn_start_list = [[vn.id, vn.start_time] for vn in self.vns_batch]
        vn_end_list = [[vn.id, vn.end_time] for vn in self.vns]
        arrive_events = pd.DataFrame(vn_start_list, columns=['vn_id', 'time'])
        arrive_events['type'] = 1
        leave_events = pd.DataFrame(vn_end_list, columns=['vn_id', 'time'])
        leave_events['type'] = 0
        self.events = pd.concat([arrive_events, leave_events], ignore_index=True)
        self.events.sort_values("time", inplace=True)
        self.events.reset_index(inplace=True, drop=True)
        return self.events

    def save(self, vns=True, events=True, path='data/vn/', id=0, result=True):
        '''Save the vn information or occure events to csv file'''
        if vns:
            # vns_path = path + str(id) + '_vns'  + '.csv'
            vns_path = path + 'vns_data'  + '.csv'
            vns_list = []
            for vn in self.vns:
                vn_info = {
                    'id': vn.id,
                    'nodes_num': vn.nodes_num,
                    'edges_num': vn.edges_num,
                    'lifetime': vn.lifetime,
                    'start_time': vn.start_time,
                    'end_time': vn.end_time,
                    'cpu_data': vn.cpu_data,
                    'rom_data': vn.rom_data,
                    'bw_data': vn.bw_data
                }
                '''Include other information'''
                if result==False:
                    vn_info['nodes_slot'] = vn.nodes_slot
                vns_list.append(vn_info)
            pd_vns = pd.DataFrame.from_dict(vns_list)
            pd_vns.to_csv(vns_path)
        if events:
            # events_path = path + str(id) + '_events' + '.csv'
            events_path = path + 'events_data' + '.csv'
            events_dict = []
            self.events.to_csv(events_path)


if __name__ == '__main__':
    vn = VirtualNetworkBatch(3)
    print(vn.data)
    print(vn.state)
    # print(vn.rom_data)
    # print(vn.calc_grc_c())
    # print(vn.calc_grc_M().todense())

    vns_simulator = VirtualNetworksBatchSimulator(2000, batch_size=64)
    vns_simulator.renew_vns_batch()

    # vns_simulator.renew_vns()
    # vns_simulator.renew_events()
    # vns_simulator.save()
    print()
    # vns_simulator.load()