import os
import sys
import numpy as np


nodes = [(0, {'cpu': 20}), (1, {'cpu': 35}), (2, {'cpu': 23}), (3, {'cpu': 36}), (4, {'cpu': 41}), (5, {'cpu': 46}), (6, {'cpu': 45}), (7, {'cpu': 23}), (8, {'cpu': 49}), (9, {'cpu': 43})]
node_current_embedding = np.array([i for i in range(10)], dtype=np.float32)
x = np.array([n[1]['cpu'] for n in nodes], dtype=np.float32)

x = np.stack([x, node_current_embedding], axis=1)
# x = np.array([n[1]['cpu'] for n in nodes], dtype=np.float32).reshape(len(nodes), 1)

edges = [(0, 2, {'bw': 42, 'weight': 1}), (1, 2, {'bw': 28, 'weight': 1}), (4, 5, {'bw': 26, 'weight': 1}), (6, 9, {'bw': 37, 'weight': 1})]
y = [[e[0] for e in edges], [e[1] for e in edges]]

w = [e[2]['weight'] for e in edges]

# print(x)
# print(y)
# print(w)
'''gcn'''
# import tensorflow as tf
# import tf_geometric as tfg

# # output = tfg.nn.gcn(tf.constant(x), tf.constant(y), w, tf.constant([1, 64]))
# # print(output)

# g = tfg.Graph(x, edge_index=np.array(y, dtype=np.float32), edge_weight=np.array(w, dtype=np.float32))

# print("Processed Graph Desc: \n", g)
# print("Processed Graph Desc: \n", g.x)
# print("Processed Edge Index:\n", g.edge_index)

# gcn = tfg.layers.GCN(units=64, activation=tf.nn.relu)
# f = tf.keras.layers.Flatten()
# output = gcn([tf.constant(x), tf.constant(y)])
# q = f(output)
# print(tf.reshape(q, shape=[-1]))


# file_path_dir = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(file_path_dir)

# from config import *
# args = parser.parse_args()
# print(args.embedding_size)

'''shortest path'''

# import networkx as nx

# G = nx.Graph()
# G.add_nodes_from([0, 1, 2, 3, 4])
# G.add_edge(0, 2)
# try:
#     shortest_path = nx.dijkstra_path(G, 0, 2)
# except nx.exception.NetworkXNoPath:
#     print('NetworkXNoPath')
# else:
#     print(shortest_path)


'''vn state'''
# import os
# import sys

file_path_dir = os.path.dirname(os.path.dirname(__file__))
path = os.path.join(file_path_dir, 'generator')
sys.path.append(file_path_dir)

from env import PhysicalNetworkEnv, PNBatchEnv
from generator.vnf_chain_batch_generator import VNFChainBatchGenerator
from generator.physical_network_loader import PhysicalNetwork

# batch_size = 64

# # vn state 
# vn_batch = VNFChainBatchGenerator(batch_size, 2, 10)
# vn_batch.getNewState()
# vn_batch_state = vn_batch.state
# vn_batch_length = vn_batch.service_length
# print(vn_batch_state)
# vnf_state = [[vn_batch_state[bid][aid][0] for aid in range(3)] for bid in range(batch_size)]
# print(vnf_state)

'''args_parse'''
# from config import get_config
# args, _ = get_config()
# print(args.batch_size)

'''spektral'''
from spektral.datasets import citation
from spektral.layers import GraphConv
from spektral.utils.data import numpy_to_disjoint
from tensorflow.keras.layers import Input

from spektral.datasets import citation
A, X, y, train_mask, val_mask, test_mask = citation.load_data('cora')

N = A.shape[0]
F = X.shape[-1]
n_classes = y.shape[-1]

print('A', A.shape)
print('X', X[0])
# print('y', y)
# print('N', N)

from spektral.layers import GraphConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.regularizers import l2

dropout = 0.5 
channels = 16
l2_reg = 5e-4 / 2

# Preprocessing operations
fltr = GraphConv.preprocess(A).astype('f4')
X = X.toarray()

X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)


dropout_1 = Dropout(dropout)(X_in)
graph_conv_1 = GraphConv(channels,
                         activation='relu',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False)
# dropout_2 = Dropout(dropout)(graph_conv_1)
# graph_conv_2 = GraphConv(n_classes,
#                          activation='softmax',
#                          use_bias=False)([dropout_2, fltr_in])
# Build model
# model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)

print(graph_conv_1([X, fltr]))
# print(output)