import os
import sys
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

file_path_dir = os.path.dirname(os.path.dirname(__file__))
path = os.path.join(file_path_dir, 'generator')
sys.path.append(file_path_dir)


class VNBatchGenerator(object):
    """
        Implementation of a random service chain generator

        Attributes:
            state[batch_size, max_length]           -- Batch of random service chains
            length[batch_size]            -- Array containing services length
    """

    def __init__(self, batch_size, min_length, max_length, 
                min_request=2, max_request=30, cpu_benchmark=100, bw_benchmark=1500, 
                **kwargs):
        """
        Args:
            batch_size(int)         -- Number of service chains to be generated
            min_length(int)         -- Minimum service length
            max_length(int)         -- Maximum service length
            vocab_size(int)         -- Size of the VNF dictionary
        """
        # global settings
        self.batch_size = batch_size
        self.min_length = min_length
        self.max_length = max_length
        self.min_request = min_request
        self.max_request = max_request
        self.cpu_benchmark = cpu_benchmark
        self.bw_benchmark = bw_benchmark

        # random_seed = kwargs.pop('random_seed', 1024)
        # np.random.seed(random_seed)

        # SFC = (state, liftime, start_time)
        # state = (cpu_requet, bw_request, pending_num)

        self.data = np.zeros(
            (self.batch_size, 3, self.max_length),  dtype='int32')
        self.state = np.zeros(
            (self.batch_size, 3, self.max_length),  dtype='float32')
        
        self.length = np.zeros(self.batch_size, dtype='int32')
        self.lifetime = 0
        self.start_time = 0
        self.end_time = 0

    def normalize_data(self):
        # cup_bw_state
        benchmark = [[[self.cpu_benchmark]*self.max_length, [self.bw_benchmark]*self.max_length]]*self.batch_size
        cup_bw_state = (self.data[:, 0:2, :] / benchmark)
        sfc_length = [[self.length[i]] for i in range(self.batch_size)]
        # pending_num_state
        pending_num_data = self.data[:, 2, :] / sfc_length
        pending_num_state = np.expand_dims(pending_num_data, axis=1)
        # state
        state = np.concatenate([cup_bw_state, pending_num_state], axis=1)
        return state

    def renew_info(self, sfc_id=0, arrive_interval=0):
        """ Generate new batch of service chain """
        # Clean attributes
        self.data = np.zeros(
            (self.batch_size, 3, self.max_length), dtype='int32')
        self.state = np.zeros(
            (self.batch_size, 3, self.max_length), dtype='float64')        
        self.length = np.zeros(self.batch_size,  dtype='int32')
        self.lifetime = 0
        self.start_time = 0
        self.end_time = 0

        # genarate random services chain
        self.id = sfc_id
        self.lifetime = np.random.exponential(500)
        self.start_time = np.random.uniform(100) + arrive_interval * 100
        self.end_time = self.lifetime + self.start_time

        for batch in range(self.batch_size):
            # random length
            self.length[batch] = np.random.randint(
                self.min_length, self.max_length+1, dtype='int32')
            curr_length = self.length[batch]

            # random attribution
            self.data[batch][0] = np.pad(np.ceil(np.random.uniform(
                self.min_request, self.max_request, curr_length)), (0, 10 - curr_length), 'constant', constant_values=(0, 0))
            self.data[batch][1] = np.pad(np.ceil(np.random.uniform(
                self.min_request, self.max_request, curr_length-1)), (1, 10 - curr_length), 'constant', constant_values=(0, 0))
            self.data[batch][2] = np.pad(np.arange(curr_length)[::-1], (0, 10 - curr_length), 'constant', constant_values=(0, 0))
        self.state = self.normalize_data()


class VNSBatchSimulator():
    '''
    '''
    def __init__(self, sfcs_num, batch_size, 
                min_length, max_length, min_request=2, max_request=30,
                cpu_benchmark=100, bw_benchmark=1500, random_seed=1024):
        self.sfcs_num = sfcs_num
        np.random.seed(random_seed)
        self.sfc_generator = VNBatchGenerator(batch_size, min_length, max_length, 
                                                min_request=min_request, max_request=max_request, 
                                                cpu_benchmark=cpu_benchmark, bw_benchmark=bw_benchmark, 
                                                random_seed=random_seed)
        self.sfcs = []
        self.events = None
        
    
    def renew_sfcs(self):
        '''Renew the SFC information'''
        sfc_id = 0
        for i in range(int(np.ceil(self.sfcs_num/4))):
            for j in range(4):
                self.sfc_generator.renew_info(sfc_id=sfc_id, arrive_interval=i)
                sfc_info = {
                    'id': self.sfc_generator.id,
                    'data': self.sfc_generator.data,
                    'state': self.sfc_generator.state,                    
                    'length': self.sfc_generator.length,
                    'lifetime': self.sfc_generator.lifetime,
                    'start_time': self.sfc_generator.start_time,
                    'end_time': self.sfc_generator.end_time,
                }
                self.sfcs.append(sfc_info)
                sfc_id += 1
                if sfc_id >= self.sfcs_num:
                    break
        return self.sfcs

    def renew_events(self):
        '''Renew the occur events accordding to the SFC information'''
        sfc_start_list = [[sfc['id'], sfc['start_time']] for sfc in self.sfcs]
        sfc_end_list = [[sfc['id'], sfc['end_time']] for sfc in self.sfcs]
        arrive_events = pd.DataFrame(sfc_start_list, columns=['sfc_id', 'time'])
        arrive_events['type'] = 1
        leave_events = pd.DataFrame(sfc_end_list, columns=['sfc_id', 'time'])
        leave_events['type'] = 0
        self.events = pd.concat([arrive_events, leave_events], ignore_index=True)
        self.events.sort_values("time", inplace=True)
        self.events.reset_index(inplace=True, drop=True)
        return self.events

    def save(self, sfcs=True, events=True, path='data/sfc/', id=0):
        '''Save the SFC information or occure events to csv file'''
        if sfcs:
            sfcs_path = path + str(id) + '_sfcs'  + '.csv'
            pd_sfcs = pd.DataFrame.from_dict(self.sfcs)
            pd_sfcs.to_csv(sfcs_path)
        if events:
            events_path = path + str(id) + '_events' + '.csv'
            self.events.to_csv(events_path)

def key_start_time(sfc):
        return sfc['start_time']

def key_end_time(sfc):
    return sfc['end_time']


if __name__ == '__main__':
    sfcs_num = 10
    batch_size = 100
    min_length = 2
    max_length = 10
    vocab_size = 8

    sfc_simulation = SFCRequestsSimulator(sfcs_num, batch_size, min_length, max_length)
    sfcs = sfc_simulation.renew_sfcs()
    events = sfc_simulation.renew_events()
    # sfc_simulation.save()
    # print(sfc_simulation.events.loc[0]['type'])
    