import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx


class VirtualNetwork(object):
    """Virtual Network

    Take four resources (cpu, ram, rom, bw) into consideration

    Args:
        graph: a networkx graph
        nodes_data: [cpu, ram, rom]
        edges_data: [bw]
        data: [cpu, rom, ram, bw]
        state: [norm_cpu, norm_ram, norm_rom, norm_bw]
        nodes_num:
        edges_num:

    For DRL Algo:

    For GRC Algo:
        grc_c
        grc_M
    """
    def __init__(self, nodes_num=10, min_request=2, max_request=30, **kwargs):
        super().__init__()
        self.nodes_num = nodes_num
        self.edges_num = nodes_num-1
        self.min_request = min_request
        self.max_request = max_request
        # other
        self.id = kwargs.pop('id', 0)
        self.arrive_interval = kwargs.pop('arrive_interval', 0)
        self.aver_lifetime =  kwargs.pop('aver_lifetime', 500)
        self.node_benchmark = kwargs.pop('node_benchmark', 100)
        self.edge_benchmark = kwargs.pop('edge_benchmark', 1403)
        # record
        self.slots = {}
        self.paths = {}
        # random
        if kwargs.pop('random', True):
            self.add_random_data()

    ### Add info to VirtualNetwork ###
    def load_from_pd(self, pd):
        """Load the data from dataset"""
        self.id = pd['id']
        self.nodes_num = pd['nodes_num']
        self.edges_num = pd['edges_num']
        self.lifetime = pd['lifetime']
        self.start_time = pd['start_time']
        self.end_time = pd['end_time']
        self.cpu_data = np.array(pd['cpu_data'].replace('[', '').replace(']', '').split(), dtype=np.int32)
        self.ram_data = np.array(pd['ram_data'].replace('[', '').replace(']', '').split(), dtype=np.int32)
        self.rom_data = np.array(pd['rom_data'].replace('[', '').replace(']', '').split(), dtype=np.int32)
        self.bw_data = np.array(pd['bw_data'].replace('[', '').replace(']', '').split(), dtype=np.int32)
        self.graph = nx.path_graph(self.nodes_num)
        self.add_edge_attrs(random=False)
        self.add_node_attrs(random=False)

    def add_random_data(self):
        """Add random attrs to VN
        
        if random=True, the random data is added to VN when initializing"""
        self.graph = nx.path_graph(self.nodes_num)
        self.add_edge_attrs()
        self.add_node_attrs()
        self.add_other_attrs()

    def add_edge_attrs(self, random=True):
        """Add edge attrs to VN

        random [bool]:
            True: the random data is added to VN (For initializing randomly).
            False: the data of bw_data is add to VN (For renewing VN data).
        """
        if random:
            self.bw_data = np.random.randint(self.min_request, self.max_request+1, size=(self.edges_num))
        i = 0
        for v in self.graph.edges:
            self.graph.edges[v]['weight'] = 1
            self.graph.edges[v]['bw'] = self.bw_data[i]
            i += 1

    def add_node_attrs(self, random=True):
        """Add nodes attrs to VN

        random [bool]:
            True: the random data is added to VN (For initializing randomly).
            False: the data of bw_data is add to VN (For renewing VN data).
        """
        if random:
            self.cpu_data = np.random.randint(self.min_request, self.max_request+1, size=(self.nodes_num))
            self.ram_data = np.random.randint(self.min_request, self.max_request+1, size=(self.nodes_num))
            self.rom_data = np.random.randint(self.min_request, self.max_request+1, size=(self.nodes_num))
        i = 0
        for n in self.graph.nodes:
            self.graph.nodes[n]['cpu'] = self.cpu_data[i]
            self.graph.nodes[n]['ram'] = self.ram_data[i]
            self.graph.nodes[n]['rom'] = self.rom_data[i]
            bw_sum = 0
            for neighbor in self.graph.adj[n]:
                bw_sum += self.graph.edges[n, neighbor]['bw']
            self.graph.nodes[n]['bw_sum'] = bw_sum
            i += 1

    def add_other_attrs(self, random=True, id=None, aver_lifetime=None, arrive_interval=None):
        """Add other attrs of VN

        random [bool]:
            True: the random data is added to VN (For initializing randomly).
            False: the data of bw_data is add to VN (For renewing VN data).
        """
        if random:
            self.lifetime = np.random.exponential(self.aver_lifetime)
            self.arrive_interval = arrive_interval
            self.start_time = np.random.uniform(100) + self.arrive_interval * 100
            self.end_time = self.lifetime + self.start_time
            return True
        if id != None:
            self.id = id
        if aver_lifetime != None:
            self.aver_lifetime = aver_lifetime
            self.lifetime = np.random.exponential(self.aver_lifetime)
        # Note: if the default value is set 'False', the case that arrive_interval=0 will be ignored
        if arrive_interval != None:
            self.arrive_interval = arrive_interval
            self.start_time = np.random.uniform(100) + self.arrive_interval * 100
            self.end_time = self.lifetime + self.start_time

    ### data ###
    @property
    def data(self):
        """data: [cpu, ram, rom, bw]"""
        cpu_data = self.cpu_data
        ram_data = self.ram_data
        rom_data = self.rom_data
        bw_data = self.bw_data
        bw_data = np.insert(bw_data, 0, 0)
        data = np.vstack([cpu_data, ram_data, rom_data, bw_data])
        return data

    ### For DRL ###
    @property
    def state(self):
        """state: [norm_cpu, norm_ram, norm_rom, norm_bw]"""
        norm_cpu_data = self.cpu_data / self.node_benchmark
        norm_ram_data = self.ram_data / self.node_benchmark
        norm_rom_data = self.rom_data / self.node_benchmark
        norm_bw_data = np.insert(self.bw_data, 0, 0) / self.edge_benchmark
        norm_data = np.vstack([norm_cpu_data, norm_ram_data, norm_rom_data, norm_bw_data])
        return norm_data
    
    @property
    def nodes_data(self):
        """nodes_data: [cpu, ram, rom]"""
        return np.vstack([self.cpu_data, self.ram_data, self.rom_data])

    @property
    def edges_data(self):
        """edges_data: [bw]"""
        bw_data = self.bw_data 
        return np.insert(bw_data, 0, 0)

    ### statistic ###
    def sum_data(self):
        """The summation of all resources of VN"""
        return self.data.sum(axis=(0, 1))

    @property
    def revenue(self):
        if (self.slots == None) or (self.slots == {}):
            return 0
        return self.data.sum(axis=(0, 1))

    @property
    def cost(self):
        if (self.slots == None) or (self.slots == {}):
            return 0
        nodes_cost = self.nodes_data.sum(axis=(0, 1))
        edges_cost = 0
        for edge, path in self.paths.items():
            edges_cost += self.graph.edges[edge]['bw'] * (len(path)-1)
        return nodes_cost + edges_cost

    ### For GRC ###
    def calc_grc_c(self):
        """norm_node_data = node_data / max(node_data)"""
        node_data_sum = self.cpu_data + self.ram_data + self.rom_data
        sum_nodes_request = sum(node_data_sum)
        return node_data_sum / sum_nodes_request
    
    def calc_grc_M(self):
        M = nx.attr_sparse_matrix(self.graph, edge_attr='bw', normalized=True, rc_order=self.graph.nodes).T
        return M


class VirtualNetworksSimulator():
    """[summary]"""
    def __init__(self, vns_num=100, 
                min_length=2, max_length=15, 
                min_request=2, max_request=30,
                node_benchmark=100, edge_benchmark=1403, 
                arrival_rate=20, aver_lifetime=400, 
                random_seed=1024, **kwargs):
        self.vns_num = vns_num
        # section
        self.min_length = min_length
        self.max_length = max_length
        self.min_request = min_request
        self.max_request = max_request
        # benchmark
        self.node_benchmark = node_benchmark
        self.edge_benchmark = edge_benchmark
        self.arrival_rate = arrival_rate
        self.aver_lifetime = aver_lifetime
        # data
        self.vns = []
        self.events = []

    def renew_vns(self):
        '''Renew the vn information'''
        self.vns_length = np.random.randint(self.min_length, self.max_length, self.vns_num)
        vn_id = 0
        for i in range(int(np.ceil(self.vns_num/self.arrival_rate))):
            for j in range(self.arrival_rate):
                if vn_id >= self.vns_num:
                    break
                vn = VirtualNetwork(self.vns_length[vn_id], self.min_request, self.max_request, 
                                    node_benchmark=self.node_benchmark, edge_benchmark=self.edge_benchmark,
                                    id=vn_id, arrive_interval=i, aver_lifetime=self.aver_lifetime)
                self.vns.append(vn)
                vn_id += 1
        return self.vns

    def renew_events(self):
        '''Renew the occur events accordding to the vn information'''
        vn_start_list = [[vn.id, vn.start_time] for vn in self.vns]
        vn_end_list = [[vn.id, vn.end_time] for vn in self.vns]
        arrive_events = pd.DataFrame(vn_start_list, columns=['vn_id', 'time'])
        arrive_events['type'] = 1
        leave_events = pd.DataFrame(vn_end_list, columns=['vn_id', 'time'])
        leave_events['type'] = 0
        self.events = pd.concat([arrive_events, leave_events], ignore_index=True)
        self.events.sort_values("time", inplace=True)
        self.events.reset_index(inplace=True, drop=True)
        return self.events

    def renew_others(self, arrival_rate=None, aver_lifetime=None):
        '''renew the arrival_rate and aver_lifetime not modifying the date'''
        if not arrival_rate == None:
            self.arrival_rate = arrival_rate
        if not aver_lifetime == None:
            self.aver_lifetime = aver_lifetime
        vn_id = 0
        for i in range(int(np.ceil(self.vns_num/self.arrival_rate))):
            for j in range(self.arrival_rate):
                if vn_id >= self.vns_num:
                    break
                self.vns[vn_id].add_other_attrs(random=False, aver_lifetime=aver_lifetime, arrive_interval=i)
                vn_id += 1

    def save(self, vns=True, events=True, path='data/vn/', id=0, result=False, vns_data_file='vns_data.csv', events_data_file='events_data.csv',):
        '''Save the vn information or occure events to csv file'''
        if vns:
            # vns_path = path + str(id) + '_vns'  + '.csv'
            vns_path = path + vns_data_file
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
                    'ram_data': vn.ram_data,
                    'rom_data': vn.rom_data,
                    'bw_data': vn.bw_data
                }
                '''Include other information'''
                if result==True:
                    vn_info['nodes_slot'] = vn.nodes_slot
                vns_list.append(vn_info)
            pd_vns = pd.DataFrame.from_dict(vns_list)
            pd_vns.to_csv(vns_path)
        if events:
            # events_path = path + str(id) + '_events' + '.csv'
            events_path = path + events_data_file
            events_dict = []
            self.events.to_csv(events_path)

    def load_from_csv(self, path='data/vn/', vns_data_file='vns_data.csv', events_data_file='events_data.csv'):
        vns_data = pd.read_csv(path + vns_data_file, index_col=0, header=0)
        self.events = pd.read_csv(path + events_data_file, index_col=0, header=0)
        self.vns_num = len(vns_data)
        for i in range(self.vns_num):
            vn = VirtualNetwork(random=False)
            vn_data = vns_data.loc[i]
            vn.load_from_pd(vn_data)
            self.vns.append(vn)


if __name__ == '__main__':
    # print(vn.cpu_data)
    # print(vn.rom_data)
    # print(vn.calc_grc_c())
    # print(vn.calc_grc_M().todense())
    vns_num = 2000
    min_length = 2
    max_length = 15
    min_request = 2
    max_request = 30
    arrival_rate = 20
    aver_lifetime = 400
    vns_simulator = VirtualNetworksSimulator(vns_num, min_length=min_length, max_length=max_length, 
                                            min_request=min_request, max_request=max_request, 
                                            arrival_rate=arrival_rate, aver_lifetime=aver_lifetime)
    # vns_simulator.renew_vns()
    # vns_simulator.renew_events()
    # vns_simulator.save(vns_data_file=f'vns_data_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}.csv', 
    #                     events_data_file=f'events_data_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}.csv')
    
    vns_simulator.load_from_csv(vns_data_file=f'vns_data_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}.csv', 
                        events_data_file=f'events_data_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}.csv')
    
    '''renew the arrival rate'''
    new_arrival_rate=18
    vns_simulator.renew_others(arrival_rate=new_arrival_rate)
    vns_simulator.renew_events()
    vns_simulator.save(vns_data_file=f'vns_data_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{new_arrival_rate}_{aver_lifetime}.csv', 
                        events_data_file=f'events_data_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{new_arrival_rate}_{aver_lifetime}.csv')
