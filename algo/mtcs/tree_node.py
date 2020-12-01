import copy
import math
import numpy as np
import networkx as nx


class TreeNode(object):
    def __init__(self, pid, vid, parent=None, path=[]):
        """
        """
        # state
        self.vid = vid  # the id of virtual network node
        self.pid = pid  # the id of physical network node
        # node
        self.parent = parent
        self.children = []  # chirdren nodes
        self.path = []
        # value
        self.value = 0  # win value
        self.visit = 0  # visit visit

    def set_parent(self, parent):
        self.parent = parent
    
    def add_children(self, chid_node):
        self.children.append(chid_node)
    
    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.children == []


class MCTS(object):
    def __init__(self, d, pn, vn):
        self.d = d  # exploration constant
        self.pn = pn
        self.vn = vn
        self.root = TreeNode(None, None, 0)
        self.selected_nodes = []

    def select(self, pn, vn, node, strategy='greedy'):
        """select a children node from all candicate nodes according to the strategy
        
        UTD = (child_value/ child_visit) + D * sqrt(In(parent_visit) / child_visit)

        strategy [greedy | random]:
            greedy: 
            random: 
        """
        utd = 0
        parent_visit = node.visit
        selected_node = node.chirdren[0]

        for n in node.chirdren:
            child_value = n.value
            child_visit = n.visit
            # calculate utd
            if child_visit == 0:
                curr_utd = np.inf
            else:
                curr_utd = (child_value/ child_visit) + self.d * math.sqrt((math.log(parent_visit) / child_visit))
            # update selected node
            if curr_utd > utd:
                selected_node = n
        
        pid = selected_node.pid
        vid = selected_node.vid
        path = selected_node.vid

        # update PN
        cpu_req = vn.graph.node[vid]['cpu']
        ram_req = vn.graph.node[vid]['cpu']
        rom_req = vn.graph.node[vid]['cpu']
        bw_req = vn.graph.edges[vid, vid+1]['bw']
        pn.update_node(vid, 'cpu', -cpu_req)
        pn.update_node(vid, 'ram', -ram_req)
        pn.update_node(vid, 'rom', -rom_req)
        pn.update_bw_with_path(path, -bw_req)
        return selected_node

    def expand(self, node):
        """[summary]
        """
        vid = node.vid
        # node constraint
        cpu_req = vn.graph.node[vid]['cpu']
        ram_req = vn.graph.node[vid]['cpu']
        rom_req = vn.graph.node[vid]['cpu']
        
        candicate_nodes = self.pn.find_candicate_nodes(cpu_req, ram_req, rom_req, filter=self.selected_nodes)
        
        # edge constraint & update resource
        # sub graph
        bw_req = vn.graph.edges[vid, vid+1]['bw']
        edges_data_bw_free = pn.graph.edges.data('bw_free')
        available_egdes = [(u, v) for (u, v, bw_free) in edges_data_bw_free if bw_free >= bw_req]
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(available_egdes)
        
        # find chirdren node
        for candi_pid in candicate_nodes:
            try:
                # finded shortest path
                path = nx.dijkstra_path(temp_graph, node.pid, candi_pid)
                # update link resource
                pn.update_bw_with_path(path, -bw_req)
                vn_pn_paths[(vid, vid+1)] = path
                # update node resource
                pn.update_node(candi_pid, 'cpu', -cpu_req)
                pn.update_node(candi_pid, 'ram', -ram_req)
                pn.update_node(candi_pid, 'rom', -rom_req)
                # add to chirdren node list
                node.append(TreeNode(candi_pid, vid+1, node, path=path))
            except:
                # FAILURE
                vn.slots = {}
                vn.paths = {}
        return True

    def backpropagate(self, node, flag):
        curr_node = node
        while(curr_node.parent != None):
            curr_node.parent.visit += 1
            curr_node.parent.value += 1
            curr_node = self.parent
        return True

    def simulate(self):
        pn = copy.deepcopy(self.pn)
        vn = copy.deepcopy(self.vn)
        curr_node = self.root
        
        # Select
        while(curr_node.children != []):
            selected_nodes = self.select(pn, vn, curr_node)
            self.selected_nodes.append(selected_nodes.pid)
        
        # Expand
        while(selected_nodes.vid > vn.node_num):
            flag = self.expand(pn, vn, curr_node)
            if not flag:
                self.backpropagate(selected_nodes, flag)
            selected_nodes = self.select(pn, vn, curr_node)


        # Backprogate
        if flag:
            # FAILURE
            self.backpropagate(selected_nodes, flag)
            2
        else:
            # SUCCESS
            1

    def decision(self):
        pass





if __name__ == '__main__':
    pass