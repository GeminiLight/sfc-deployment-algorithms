import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt




# self.val1 = kwargs.pop('val1', "default value")
# self.val2 = kwargs.pop('val2', "default value")

class PhysicalNetworkGenerator(object):
    def __init__(self, nodes_num=100, edges_num=500, wm_alpha=0.5, wm_beta=0.2, auto_save=False, auto_draw=False, **kwargs):
        self.nodes_num = nodes_num
        self.edges_num = edges_num
        self.wm_alpha = wm_alpha
        self.wm_beta = wm_beta

        random_seed = kwargs.pop('random_seed', 1024)
        np.random.seed(random_seed)

        self.save_path_nodes = kwargs.pop('save_path_nodes', 'data/pn/nodes_data.csv')
        self.save_path_edges = kwargs.pop('save_path_edges', 'data/pn/edges_data.csv')
        self.min_resoure = kwargs.pop('min_resoure', 50)
        self.max_resoure = kwargs.pop('max_resoure', 100)

        self.graph = nx.generators.geometric.waxman_graph(nodes_num, wm_alpha, wm_beta)

        self.add_edge_attrs()
        self.add_node_attrs()

        self.nodes = self.graph.nodes
        self.edges = self.graph.edges

        # self.auto_save = kwargs.pop('auto_save', False)
        # self.auto_draw = kwargs.pop('auto_draw', False)

        if (auto_save):
            self.save_data()
        if (auto_draw):
            self.draw_graph()


    def add_edge_attrs(self):
        for v in self.graph.edges:
            self.graph.edges[v]['weight'] = 1
            self.graph.edges[v]['bw_max'] = self.graph.edges[v]['bw_free'] = np.random.randint(self.min_resoure, self.max_resoure+1)

    def add_node_attrs(self):
        for n in self.graph.nodes:
            self.graph.nodes[n]['cpu_max'] = self.graph.nodes[n]['cpu_free'] = np.random.randint(self.min_resoure, self.max_resoure+1)
            self.graph.nodes[n]['ram_max'] = self.graph.nodes[n]['ram_free'] = np.random.randint(self.min_resoure, self.max_resoure+1)
            self.graph.nodes[n]['rom_max'] = self.graph.nodes[n]['rom_free'] = np.random.randint(self.min_resoure, self.max_resoure+1)
            bw_sum = 0
            for neighbor in self.graph.adj[n]:
                bw_sum += self.graph.edges[n, neighbor]['bw_free']
            self.graph.nodes[n]['bw_sum_max'] = self.graph.nodes[n]['bw_sum_free'] = bw_sum  

    def save_data(self):
        # save the data of nodes to 'nodes_data.csv'
        n_index = self.graph.nodes

        pos = list(nx.get_node_attributes(self.graph, 'pos').values())
        pos_x = [p[0] for p in pos]
        pos_y = [p[1] for p in pos]
        cpu_max = list(nx.get_node_attributes(self.graph, 'cpu_max').values())
        cpu_free = list(nx.get_node_attributes(self.graph, 'cpu_free').values())
        ram_max = list(nx.get_node_attributes(self.graph, 'ram_max').values())
        ram_free = list(nx.get_node_attributes(self.graph, 'ram_free').values())  
        rom_max = list(nx.get_node_attributes(self.graph, 'rom_max').values())
        rom_free = list(nx.get_node_attributes(self.graph, 'rom_free').values())
        bw_sum_max = list(nx.get_node_attributes(self.graph, 'bw_sum_max').values())
        bw_sum_free = list(nx.get_node_attributes(self.graph, 'bw_sum_free').values())
        
        n_data = {
            'pos_x': pos_x,
            'pos_y': pos_y,
            'cpu_max': cpu_max,
            'cpu_free': cpu_free,
            'ram_max': ram_max,
            'ram_free': ram_free,
            'rom_max': rom_max,
            'rom_free': rom_free,
            'bw_sum_max': bw_sum_max,
            'bw_sum_free': bw_sum_free
        }
        
        nodes = pd.DataFrame(n_data, index=n_index)
        nodes.to_csv(self.save_path_nodes)

        # save the data of edges to 'edges_data.csv'
        v = self.graph.edges
        v_i = [e[0] for e in v]
        v_j = [e[1] for e in v]
        weight = list(nx.get_edge_attributes(self.graph, 'weight').values())
        bw_max = list(nx.get_edge_attributes(self.graph, 'bw_max').values())
        bw_free = list(nx.get_edge_attributes(self.graph, 'bw_free').values())
        
        v_data = {
            'v_i': v_i,
            'v_j': v_j,
            'weight': weight,
            'bw_max': bw_max,
            'bw_free': bw_free,
        }

        edges = pd.DataFrame(v_data)
        edges.to_csv(self.save_path_edges)

    def draw_graph(self):
        plt.figure(1)
        nx.draw(self.graph, with_labels=True)
        plt.show()


if __name__ == '__main__':
    G = PhysicalNetworkGenerator(auto_save=True, auto_draw=False)
    cpu_max = G.nodes('cpu_max')
    cpu_max = [n[1] for n in cpu_max]
    print(max(cpu_max))
    # print(G.edges(data=True))
    