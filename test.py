import torch
import networkx as nx
from deepsnap.graph import Graph

G = nx.Graph()
G.add_node(0, node_feature=torch.tensor([1,2,3]))
G.add_node(1, node_feature=torch.tensor([4,5,6]))
G.add_edge(0, 1)
H = Graph(G)
H.node_feature