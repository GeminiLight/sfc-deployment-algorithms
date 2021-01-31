import copy
import numpy as np
import networkx as nx

def node_mapping(vn, pn, d=0.95):
    """
    Return: vn_pn_slots
    """
    vn_nodes = [i for i in range(vn.nodes_num)]
    vn_pn_slots = {}
    selected_nodes = []
    for vid in vn_nodes:
        cpu_req = vn.graph.nodes[vid]['cpu']
        ram_req = vn.graph.nodes[vid]['ram']
        rom_req = vn.graph.nodes[vid]['rom']
        candidate_nodes = pn.find_candidate_nodes(cpu_req, ram_req, rom_req, rtype='id')
        candidate_nodes = np.setdiff1d(candidate_nodes, vn_pn_slots.values())
        if len(candidate_nodes) == 0:
            vn.slots = {}
            vn.paths = {}
            return False
        pid = np.random.choice(candidate_nodes)
        vn_pn_slots[vid] = pid
        pn.update_node(pid, 'cpu_free', -cpu_req)
        pn.update_node(pid, 'ram_free', -ram_req)
        pn.update_node(pid, 'rom_free', -rom_req)
    return vn_pn_slots


def link_mapping(vn, pn, vn_pn_slots):
    vn_pn_paths = {}
    # vn_pn_slots_edges = [(vn_pn_slots[pid], vn_pn_slots[pid+1])for pid in range(vn.edges_num)]
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
    pass
