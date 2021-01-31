import copy
import math
import time
import random
import numpy as np
import networkx as nx
import scipy as sp

class TreeNode(object):
    def __init__(self, pid, vid, path=[], state=None, selected_nodes=[]):
        """
        """
        # state
        self.vid = vid  # the id of virtual network node
        self.pid = pid  # the id of physical network node
        self.state = state

        # node
        self.children = []  # children nodes
        self.path = path
        self.selected_nodes = selected_nodes

        # value
        self.value = 0  # win value
        self.visit = 0  # visit visit
        self.selected = False

    def add_children(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return self.children == []

    def get_selected_nodes(self):
        return self.selected_nodes

class MCTS(object):
    """[summary]

    Args:
        object ([type]): [description]
    """
    def __init__(self, d, pn, vn):
        self.d = d  # exploration constant
        self.pn = pn
        self.vn = vn
        self.root = TreeNode(None, 0,  state=copy.deepcopy(pn))
        self.selected_nodes = []

    def is_terminate(self, vid):
        return vid >= self.vn.nodes_num

    @property
    def max_vid(self):
        return self.vn.nodes_num-1

    def select(self, curr_node, strategy='greedy'):
        """select a children curr_node from all candicate nodes according to the strategy
        
        UTD = (child_value/ child_visit) + D * sqrt(In(parent_visit) / child_visit)

        strategy [greedy | random]:
            greedy: 
            random: 
        """
        if curr_node.children == []:
            return False

        utd = 0
        parent_visit = curr_node.visit
        next_node = curr_node.children[0]

        ### fisit fit ###
        # for n in curr_node.children:
        #     if n.selected == False:
        #         curr_utd = np.inf
        #     else:
        #         # calculate utd
        #         child_value = n.value
        #         child_visit = n.visit
        #         curr_utd = (child_value / child_visit) + self.d * math.sqrt((math.log(parent_visit) / child_visit))
        #     # sample_next_state selected curr_node
        #     if curr_utd > utd:
        #         utd = curr_utd
        #         next_node = n
        
        ### random ###
        utd_dict = {}
        for id, n in enumerate(curr_node.children):
            if n.selected == False:
                curr_utd = np.inf
            else:
                # calculate utd
                child_value = n.value
                child_visit = n.visit
                curr_utd = (child_value / child_visit) + self.d * math.sqrt((math.log(parent_visit) / child_visit))
            # sample_next_state selected curr_node
            utd_dict[id] = curr_utd
        max_utd_id = max(utd_dict, key=utd_dict.get)
        return curr_node.children[max_utd_id]

    def find_candidate_nodes(self, curr_node=None, curr_state=None, curr_vid=None, curr_selected_node=None):
        """Find the candidate nodes for current node.
        
        Input:
            Tree Node with vid, state and selected nodes or input separately

        Return:
            The candidate nodes with sufficent resource.
        """
        if curr_node != None:
            pn = curr_node.state
            vid = curr_node.vid
            curr_selected_node = curr_node.selected_nodes
        else:
            pn = curr_state
            vid = curr_vid
            curr_selected_node = curr_selected_node
        
        # node constraint
        cpu_req = self.vn.graph.nodes[vid]['cpu']
        ram_req = self.vn.graph.nodes[vid]['ram']
        rom_req = self.vn.graph.nodes[vid]['rom']
        candidate_nodes = pn.find_candidate_nodes(cpu_req, ram_req, rom_req, 
                                                    filter=curr_selected_node)
        if vid == 0:
            return candidate_nodes

        # consider the basic bandiwidth constraint
        new_candidate_nodes = []
        bw_req = self.vn.graph.edges[vid-1, vid]['bw']
        for n in candidate_nodes:
            adj = pn.graph.adj[n]
            bw_free = [e['bw_free'] for e in adj.values()]
            if max(bw_free) >= bw_req:
                new_candidate_nodes.append(n)
        return candidate_nodes

    def expand(self, curr_node):
        """Expand the sub tree for children"""
        candidate_nodes = self.find_candidate_nodes(curr_node)
        vid = curr_node.vid
        count = 0
        for node in candidate_nodes:
            selected_nodes = copy.deepcopy(curr_node.selected_nodes)
            selected_nodes.append(node)
            curr_node.add_children(TreeNode(node, vid+1, selected_nodes=selected_nodes))
            count += 1
            if count >= 50:
                break
        return True

    def sample_next_state(self, curr_node, next_node):
        """sample_next_state physical network state"""
        # state
        next_pn = copy.deepcopy(curr_node.state)
        next_pid = next_node.pid
        vid = curr_node.vid
        # sample_next_state node resource of PN
        cpu_req = self.vn.graph.nodes[vid]['cpu']
        ram_req = self.vn.graph.nodes[vid]['ram']
        rom_req = self.vn.graph.nodes[vid]['rom']
        next_pn.update_node(next_pid, 'cpu_free', -cpu_req)
        next_pn.update_node(next_pid, 'ram_free', -ram_req)
        next_pn.update_node(next_pid, 'rom_free', -rom_req)
        if vid >= self.vn.nodes_num-1:
            policy = next_node.selected_nodes
            next_pn, reward = self.link_mapping(next_pn, policy)
            return next_pn, reward
        else:
            return next_pn, 0

    def link_mapping(self, pn, policy, return_paths=False):
        """temppet to place according to current policy """
        revenue = 0
        cost = 0
        paths = {}
        for i in range(len(policy)-1):
            # request
            bw_req = self.vn.graph.edges[i, i+1]['bw']
            # sub_gragh
            edges_data_bw_free = pn.graph.edges.data('bw_free')
            available_egdes = [(u, v) for (u, v, bw_free)
                            in edges_data_bw_free if bw_free >= bw_req]
            temp_graph = nx.Graph()
            temp_graph.add_edges_from(available_egdes)
            try:
                # find shortest path
                path = nx.dijkstra_path(temp_graph, policy[i], policy[i+1])
                paths[(i, i+1)] = path
                revenue += bw_req
                cost += bw_req * len(path)
                # sample_next_state link resource
                pn.update_bw_with_path(path, -bw_req)
            except:
                # FAILURE
                if return_paths:
                    return pn, -1000, {}
                return pn, -1000
        # SUCCESS
        if return_paths:
            return pn, (revenue - cost), paths
        return pn, (revenue - cost)

    def simulate(self, curr_node):
        """return reward in a simulation"""
        # # SUCCESS & STOP
        # if curr_node.vid >= self.vn.nodes_num:
        #     return True
        # find the children node for curr node (UCT)
        if not curr_node.children:
            self.expand(curr_node)

        # FAILURE: no children
        if len(curr_node.children) == 0:
            return -np.inf
        # CONTINUE: select children
        else:
            # select node: calculate uct value
            next_node = self.select(curr_node)
        next_pn, reward = self.sample_next_state(curr_node, next_node)
        
        # STOP -> terminate
        if curr_node.vid >= self.vn.nodes_num-1:
            next_node.selected = True
            next_node.value += reward
            next_node.visit += 1
            return reward
        
        # First Simulate
        # children node never be selected
        if next_node.state is None:
            next_node.state = next_pn
            next_node.selected = True
            reward = self.rollout(next_node)
        # Ever Simulate
        # children node ever be selected
        # UCT -> next_node
        else:
            reward = self.simulate(next_node)
        
        next_node.value += reward
        next_node.visit += 1
        return reward

    def rollout(self, curr_node):
        """simulate current node -> end -> reward"""
        temp_curr_state = copy.deepcopy(curr_node.state)
        policy = copy.deepcopy(curr_node.selected_nodes)
        temp_curr_vid = curr_node.vid
        # vid = 9: STOP
        while temp_curr_vid < self.vn.nodes_num:
            # find children node for current node
            candidate_nodes = self.find_candidate_nodes(curr_state=temp_curr_state, curr_vid=temp_curr_vid, curr_selected_node=policy)
            # FAILURE: no children
            # reward = -np.inf -> terminate
            if len(candidate_nodes) == 0:
                return -np.inf
            # find path between the current node and the next node
            next_pid = random.choice(candidate_nodes)
            policy.append(next_pid)

            # sample_next_state node
            temp_curr_state.update_node(next_pid, 'cpu_free', -self.vn.graph.nodes[temp_curr_vid]['cpu'])
            temp_curr_state.update_node(next_pid, 'ram_free', -self.vn.graph.nodes[temp_curr_vid]['ram'])
            temp_curr_state.update_node(next_pid, 'rom_free', -self.vn.graph.nodes[temp_curr_vid]['rom'])
            temp_curr_vid += 1
        
        pn, reward = self.link_mapping(temp_curr_state, policy)

        return reward

    def decision(self, expand_num, curr_node):
        """find the best candicate node for current node"""
        # simulate * expand_num
        while(expand_num > 0):
            # simulate
            reward = self.simulate(curr_node)

            # FAILURE -> terminate
            # next node = None
            if reward == -np.inf:
                return None
            curr_node.value += reward
            curr_node.visit += 1
            expand_num -= 1
        
        # select a node from current node's children nodes
        # chird_value / chird_visit
        max_prob = -np.inf
        next_node = None
        for chird in curr_node.children:
            if chird.selected:
                prob = chird.value / chird.visit
                if prob > max_prob:
                    max_prob = prob
                    next_node = chird
        return next_node

    def run(self, expand_num):
        """place the current vn"""
        # state
        # state = {
        #     'vn': copy.deepcopy(self.vn),
        #     'pn': copy.deepcopy(self.pn)
        # }
        nodes_map = {}
        vid = 0
        curr_node = self.root
        terminate = False
        while not terminate:
            if vid >= self.vn.nodes_num:
                # TERMINATE
                terminate = True
                break
            # place VNF one by one
            # state: {vn[vid], pn}
            next_node = self.decision(expand_num, curr_node)
            if next_node == None:
                # FAILURE
                terminate = True
                break
            else:
                nodes_map[curr_node.vid] = next_node.pid
                curr_node = next_node
                vid += 1
        
        # FAILURE
        if len(nodes_map) < self.vn.nodes_num:
            return False, self.pn
        # SUCCESS
        elif len(nodes_map) == self.vn.nodes_num:
            # node mapping
            for vid, pid in nodes_map.items():
                self.pn.update_node(pid, 'cpu_free', -self.vn.graph.nodes[vid]['cpu'])
                self.pn.update_node(pid, 'ram_free', -self.vn.graph.nodes[vid]['ram'])
                self.pn.update_node(pid, 'rom_free', -self.vn.graph.nodes[vid]['rom'])
            policy = list(nodes_map.values())
            pn_state, reward, paths = self.link_mapping(self.pn, policy, return_paths=True)
            if reward == -1000:
                self.vn.slots = {}
                self.vn.paths = {}
                return False, self.pn
            self.vn.slots = nodes_map
            self.vn.paths = paths
            return True, self.pn
            
if __name__ == '__main__':
    pass

    # First Node: no consideration on edge constraint
    # candidate_nodes = {}
    # if vid == 0:
    #     for n in candidate_nodes:
    #         candidate_nodes_path[n] = []
    #     return candidate_nodes_path
    # edge constraint
    # sub graph
    # bw_req = self.vn.graph.edges[vid-1, vid]['bw']
    # edges_data_bw_free = pn.graph.edges.data('bw_free')
    # available_egdes = [(u, v) for (u, v, bw_free) in edges_data_bw_free 
    #                         if bw_free >= bw_req]
    # temp_graph = nx.Graph()
    # temp_graph.add_edges_from(available_egdes)
    
    # candidate_nodes_path = {}
    # # find candidate node for curr_node
    # for candi_pid in candidate_nodes:
    #     try:
    #         # finded shortest path
    #         path = nx.dijkstra_path(temp_graph, curr_node.pid, candi_pid)
    #         candidate_nodes_path[candi_pid] = path
    #     except:
    #         candidate_nodes.remove(candi_pid)
    # t2 = time.time()
    # print('find candicate node: ', t2-t1)
    # return candidate_nodes_path