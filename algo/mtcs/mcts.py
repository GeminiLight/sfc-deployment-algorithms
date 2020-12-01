import copy
import math
import random
import numpy as np
import networkx as nx


class TreeNode(object):
    def __init__(self, pid, vid, parent=None, path=[], state=None):
        """
        """
        # state
        self.vid = vid  # the id of virtual network node
        self.pid = pid  # the id of physical network node
        self.state = state

        # node
        self.parent = parent
        self.children = []  # chirdren nodes
        self.path = path

        # value
        self.value = 0  # win value
        self.visit = 0  # visit visit
        self.selected = False

    def set_parent(self, parent):
        self.parent = parent
    
    def add_children(self, chid_node):
        self.children.append(chid_node)
    
    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.children == []


class MCTS(object):
    """[summary]

    Args:
        object ([type]): [description]
    """
    def __init__(self, d, pn, vn):
        self.d = d  # exploration constant
        self.pn = pn
        self.vn = vn
        self.root = TreeNode(None, None, 0)
        self.selected_nodes = []

    def is_terminate(self):
        if self.vn.node_num:
            return True

    def select(self, pn, vn, node, strategy='greedy'):
        """select a children node from all candicate nodes according to the strategy
        
        UTD = (child_value/ child_visit) + D * sqrt(In(parent_visit) / child_visit)

        strategy [greedy | random]:
            greedy: 
            random: 
        """
        if node.chirdren == []:
            return False

        utd = 0
        parent_visit = node.visit
        next_node = node.chirdren[0]

        for n in node.chirdren:
            if n.selected == False:
                curr_utd = np.inf
            else:
                # calculate utd
                child_value = n.value
                child_visit = n.visit
                curr_utd = (child_value/ child_visit) + self.d * math.sqrt((math.log(parent_visit) / child_visit))
            # update selected node
            if curr_utd > utd:
                next_node = n
        # pid  = next_node.pid
        # vid  = next_node.vid
        # path = next_node.path

        # update PN
        # cpu_req = vn.graph.node[vid]['cpu']
        # ram_req = vn.graph.node[vid]['cpu']
        # rom_req = vn.graph.node[vid]['cpu']
        # bw_req = vn.graph.edges[vid, vid+1]['bw']
        # pn.update_node(vid, 'cpu', -cpu_req)
        # pn.update_node(vid, 'ram', -ram_req)
        # pn.update_node(vid, 'rom', -rom_req)
        # pn.update_bw_with_path(path, -bw_req)
        return next_node

    def expand(self, curr_node):
        """[summary]
        """
        vid = curr_node.vid
        # node constraint
        cpu_req = vn.graph.nodes[vid]['cpu']
        ram_req = vn.graph.nodes[vid]['cpu']
        rom_req = vn.graph.nodes[vid]['cpu']
        
        candicate_nodes = self.pn.find_candicate_nodes(cpu_req, ram_req, rom_req, 
                                                    filter=self.selected_nodes)
        # edge constraint
        # sub graph
        bw_req = vn.graph.edges[vid, vid+1]['bw']
        edges_data_bw_free = pn.graph.edges.data('bw_free')
        available_egdes = [(u, v) for (u, v, bw_free) in edges_data_bw_free 
                                if bw_free >= bw_req]
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(available_egdes)
        
        # find chirdren curr_node
        for candi_pid in candicate_nodes:
            try:
                # finded shortest path
                path = nx.dijkstra_path(temp_graph, curr_node.pid, candi_pid)
                curr_node.append(TreeNode(candi_pid, vid+1, curr_node, path=path))
            except:
                pass
        return True

    def simulate(self, curr_node):
        """return reward
        """
        # The purpose of simulating:
        # find the best children node for curr node (UCT)
        if not curr_node.children:
            self.expand(curr_node)
        # FAILURE: false
        if not curr_node.children:
            return False
        # select node: calculate uct value
        next_node = self.select(pn, vn, curr_node)
        # update PN
        next_state = self.update(curr_node, next_node)
        # if next_state:
        #     # complete
        #     if next_node.state != None:
        #         return True
        #     else:
        #         next_node.state = next_state
        #         next_node.is_select = True
        #         next_node.value += 1
        #         next_node.visit += 1
        #         return True
        # children node never be selected
        if next_node.state is None:
            next_node.state = next_state
            next_node.is_select = True
            reward = simulate_end(next_node)
        # children node ever be selected
        # UCT -> next_node
        else:
            reward = simulate(next_node)
        next_node.value += reward
        next_node.visit_times += 1
        return reward

    def simulate_end(self, curr_node):
        """
        模拟从现在开始走到最后, 并且计算出最终的奖励函数。
        :param curr_node:
        :return:
        """
        # 获得当前的策略π,也就是说每个VNF的映射情况。我应该怎么获得呢。我可以遍历在当前情况下的VNF

        # curr_node = copy.deepcopy(curr_node)
        while curr_node.vid < self.vn.node_num:
            # 从当前可以为当前vnf进行映射的服务器中随机选择一个来作为映射的结果。
            self.expand(curr_node)
            # FAILURE
            if curr_node.children == []:
                return False
            nid = random.choice(len)
            curr_node = curr_node.children[nid]
        return True

    def decision(self, expand_num, curr_node):
        """find candicate node for current node"""
        # simulate * expand_num
        while(expand_num > 0):
            # simulate
            reward = self.simulate(curr_node)
            # SUCCESS -> reward
            if reward:
                curr_node.value += 1
                curr_node.visit += 1
            # FAILURE -> terminate
            elif reward == False:
                return False
            expand_num -= 1
        # select a node from current node's children nodes
        next_node = self.select(curr_node)

    def run(self, expand_num):
        """place the current vn"""
        # state
        pn = copy.deepcopy(self.pn)
        vn = copy.deepcopy(self.vn)
        # run
        curr_node = self.root
        terminate = False
        while not terminate:
            # place VNF one by one
            # state: {vn[vid], pn}
            next_node = self.decision(curr_node, expand_num)
            if next_node == False:
                # FAILURE
                terminate = True
            else:
                self.vn.slots[curr_node.vid] = next_node.pid
                if curr_node.vid >= vn.node_num:
                    # TERMINATE
                    terminate = True
                else:
                    curr_node = next_node
        # FAILURE
        if len(vn.slots) < vn.node_num:
            return False
        # SUCCESS
        elif len(vn.slots) == vn.node_num:
            for vid in range(self.vn.node_num):
                pid = self.vn.slots[vid]
                cpu_req = vn.graph.node[vid]['cpu']
                ram_req = vn.graph.node[vid]['cpu']
                rom_req = vn.graph.node[vid]['cpu']
                pn.update_node(vid, 'cpu', -cpu_req)
                pn.update_node(vid, 'ram', -ram_req)
                pn.update_node(vid, 'rom', -rom_req)
                if vid == 0:
                    continue
                path = self.vn.paths[(vid, vid+1)]
                bw_req = vn.graph.edges[vid, vid+1]['bw']
                flag = pn.update_bw_with_path(path, -bw_req)
                # FAILURE
                if flag == False:
                    return False


    def update(self, curr_node, chird_node_id):
        """[summary]

        Args:
            curr_node ([type]): [description]
            chird_node_id ([type]): [description]

        Returns:
            state [type]: [description]
            reward
        """
        # 
        vid = curr_node.vid
        chird_node = curr_node.chirdren[chird_node_id]
        pid = chird_node.pid
        path = chird_node.path
        
        # update PN
        cpu_req = vn.graph.node[vid]['cpu']
        ram_req = vn.graph.node[vid]['cpu']
        rom_req = vn.graph.node[vid]['cpu']
        bw_req = vn.graph.edges[vid, vid+1]['bw']
        pn.update_node(vid, 'cpu', -cpu_req)
        pn.update_node(vid, 'ram', -ram_req)
        pn.update_node(vid, 'rom', -rom_req)
        pn.update_bw_with_path(path, -bw_req)

        # return: state, reward
        if chird_node.vid < self.vn.node_num:
            # continue
            return pn, 0
        else:
            # finished
            return pn, 1





if __name__ == '__main__':
    pass