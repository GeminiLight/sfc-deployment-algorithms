import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


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
            # for i in range(self.length[batch]):
            # vnf_cqu_request = np.ceil(np.random.uniform(self.min_request, self.max_request, 1))
            # vnf_bw_request  = np.ceil(np.random.uniform(self.min_request, self.max_request, 1))
            # print(np.ceil(np.random.uniform(self.min_request, self.max_request, i)))
        # return self.data


if __name__ == "__main__":

    # Define generator
    sfc_num = 10
    batch_size = 5
    min_length = 2
    max_length = 10
    vocab_size = 8

    sfc = SFCBatchGenerator(batch_size, min_length, max_length)
    sfc.renew_info()
    print('data')
    print(sfc.data[:, 0:2, :])
    print('state')
    print(sfc.state)
    # print('length')
    # print(sfc.length)
    print([sfc.id, sfc.lifetime, sfc.start_time, sfc.end_time])
    # print(np.ceil(np.random.uniform(0, 30, (20, 8))))