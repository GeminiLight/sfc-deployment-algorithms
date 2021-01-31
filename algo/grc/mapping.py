import os
import sys
import time
import numpy as np
import networkx as nx

file_path_dir = os.path.abspath('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(file_path_dir)

from data.virtual_network import VirtualNetwork

def grc_rank(vn, sigma=0.0001, d=0.95, rtype='array'):
    """Caculate the grc vector to rank node

    Args:
        sigma [float]: the pre-set small positive threshold
        d     [float]: weight the nodes attrs and edges attrs
        edge_attr [str]: the attr of edges considered by M

    Returns:
        r [type]: the grc rank vector
    """
    c = vn.calc_grc_c()
    M = vn.calc_grc_M()
    r = c
    delta = np.inf
    while(delta >= sigma):
        new_r = (1 - d) * c + d * M * r
        delta = np.linalg.norm(new_r - r)
        r = new_r
    if rtype == 'dict':
        dict_r = {}
        for i, v in enumerate(r):
            dict_r[i] = v
        return dict_r
    return r


def node_mapping(vn, pn, d=0.95):
    """
    Return: vn_pn_slots
    """
    vn_grc = grc_rank(vn, d=d, rtype='dict')
    pn_grc = grc_rank(pn, d=d, rtype='dict')
    vn_grc_sort = sorted(vn_grc.items(), reverse=True, key=lambda x: x[1])
    pn_grc_sort = sorted(pn_grc.items(), reverse=True, key=lambda x: x[1])
    vn_nodes = [v[0] for v in vn_grc_sort]
    pn_nodes = [p[0] for p in pn_grc_sort]
    vn_pn_slots = {}
    for vid in vn_nodes:
        for pid in pn_nodes:
            cpu_req = vn.graph.nodes[vid]['cpu']
            ram_req = vn.graph.nodes[vid]['ram']
            rom_req = vn.graph.nodes[vid]['rom']
            cpu_free = pn.graph.nodes[pid]['cpu_free']
            ram_free = pn.graph.nodes[pid]['ram_free']
            rom_free = pn.graph.nodes[pid]['rom_free']
            if pn.is_candidate_node(pid, cpu_req, ram_req, rom_req):
            # if (cpu_free >= cpu_req) and (ram_free >= ram_req) and (rom_free >= rom_req):
                vn_pn_slots[vid] = pid
                pn.update_node(pid, 'cpu_free', -cpu_req)
                pn.update_node(pid, 'ram_free', -ram_req)
                pn.update_node(pid, 'rom_free', -rom_req)
                pn_nodes.remove(pid)
                break
    # FAILURE
    if len(vn_pn_slots) < len(vn_nodes):
        vn.slots = {}
        vn.paths = {}
        return False
    # SUCCESS
    return vn_pn_slots


def link_mapping(vn, pn, vn_pn_slots):
    vn_pn_paths = {}
    for i in range(vn.edges_num):
        if i < (vn.edges_num-1) and vn_pn_slots[i] == vn_pn_slots[i+1]:
            continue
        # request
        bw_req = vn.graph.edges[i, i+1]['bw']
        # sub_gragh
        edges_data_bw_free = pn.graph.edges.data('bw_free')
        available_egdes = [(u, v) for (u, v, bw_free)
                           in edges_data_bw_free if bw_free >= bw_req]
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(available_egdes)
        try:
            # find shortest path
            path = nx.dijkstra_path(
                temp_graph, vn_pn_slots[i], vn_pn_slots[i+1])
            # update link resource
            pn.update_bw_with_path(path, -bw_req)
            vn_pn_paths[(i, i+1)] = path
        except:
            # FAILURE
            vn.slots = {}
            vn.paths = {}
            return False
    # SUCCESS
    vn.slots = vn_pn_slots
    vn.paths = vn_pn_paths
    return True


if __name__ == '__main__':
    '''test grc rank'''
    vn = VirtualNetwork(4)
    print(vn.bw_data)
    vn.bw_data = np.array([7, 18, 8])
    vn.add_edge_attrs(random=False)
    print(vn.bw_data)
    vn.bw_data = [68, 136, 136, 68]
    print(vn.calc_grc_M().todense())

    c = np.array([0.210526315789474, 0.315789473684211, 0.105263157894737, 0.368421052631579])
    # 0.109928647824913, 0.373627706891929, 0.385389979417696, 0.131053665865462
    r = grc_rank(vn, sigma=0.0001, d=0.95, rtype='array')
    print(r)