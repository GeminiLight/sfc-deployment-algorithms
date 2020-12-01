import math
import random
import copy

random.seed(10)


# 该函数用于选择可以将当前VNF进行映射的服务器的集合
def select_candidate_vm(vnf_kind, vnf_computation, chain_bandwidth, server):
    """
    该函数用于选择可以将当前VNF进行映射的服务器的集合
    1、要满足在该服务器上具有可以映射的虚拟机。种类约束
        1.1判断计算能力是否满足约束
        1.2判断与该服务器相连接的链路的中存在一条链路的剩余带宽大于服务链的带宽。
    :param chain_bandwidth:  传递当前服务链的带宽
    :param vnf_computation:  传递当前VNF的计算能力
    :param vnf_kind:  传递当前VNF的种类别
    :param server:    所有的服务器的集合
    :return:          返回符合要求的服务器集合
    """
    candidate = []
    # 遍历每个服务器得到服务器上面的虚拟机的类型
    for server_num in range(len(server)):
        # 得到当前服务器上的虚拟机的类型
        VM_type = server[server_num]["type"]
        # 得到当前服务器上的虚拟机的计算能力。
        VM_computation = server[server_num]["VM"]
        # 得到与当前服务器相连的边的集合。以及他们的带宽。只需要和服务器相连的边的带宽的最大的带宽大于需求就认为他可以映射。
        VM_E = list(server[server_num]["E"].values())
        # print(VM_E)
        # 遍历当前服务器上所有虚拟机的类型
        for VM_num in range(len(VM_type)):
            # 判断是否存在与该VNF类型相同的的虚拟机或者空的虚拟机。 # 判断该虚拟机的计算能力是否大于等于该VNF所需要的能力
            if VM_type[VM_num] == vnf_kind:
                # 然后再判断与该节点相连接的链路的剩余带宽是否存在大于等于该服务链所需要的带宽
                if VM_computation[VM_num] >= vnf_computation and \
                        max(VM_E) >= chain_bandwidth:  # and max(VM_E) >= chain_bandwidth
                    candidate.append(server_num)
                    break
    # 如果返回的candidate为空的话就说明不存在可以映射的服务器
    # print(candidate)
    return candidate


class TreeNode(object):
    """
      存储树节点的各个信息
    """

    def __init__(self, parent, state, num, server_num):
        """
        :param parent:  父节点
        :param state:   状态
        :param num:     节点的编号
        :param server_num: 服务器的编号
        """
        self.num = num  # 当前节点的编号
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 孩子节点的集合,里面存储的是一个一个的节点！
        self.value = 0  # 值
        self.visit_times = 0  # 访问次数
        self.state = state  # 若节点还有加入到树中,将其状态设置为None也就是不存在
        self.is_select = False  # 判断当前节点是否被扩展过
        self.server_num = server_num  # 服务器的编号

    # 为当前状态下需要映射的VNF创建候选孩子节点,候选孩子节点是还没有被扩展到树中的节点
    def create_candidate_children(self, vnI):
        # 获得当前VNF可以映射的候选服务器集合
        result = select_candidate_vm(self.state["current_chain"]["vnf_kind"][0],
                                     self.state["current_chain"]["vnf_computation"][0],
                                     self.state["current_chain"]["bandwidth"],
                                     self.state["SN_state"])
        # 根据候选服务器集合创建孩子节点,每个孩子节点中服务器的编号,就是服务器的编号
        # 如果result为None以下代码将不会执行
        for i in range(len(result)):
            self.children.append(TreeNode(self.num, None, i, result[i]))

    # 改变当前节点的父节点状态,因为根节点会发生变化。
    def set_parent(self, parent):
        self.parent = parent


def simulate(tree_node):
    """
    通过仿真来返回奖励值
    :return: reward 返回奖励
    """
    # 可下点的选择不是随机的，而是根据UCB的信息上界索引值进行选择。如果可下点没有被访问，则可下点的上限信心索引值为正无穷大；
    # 如果可下点被访问过，则可下点的上限信心索引值可以根据UCB算法给出的上限信心索引值确定。
    # 在实际应用中，我们采用UCB算法给出的上限信心索引值确定上限信心索引值。当我们需要在众多的可下点中选取一个时，我们要选择上限信心索引值最大的一个
    # 生成当前节点的候选孩子节点。
    if not tree_node.children:
        tree_node.create_candidate_children(vnI=0)
    if not tree_node.children:
        return float("-inf")
    children_UCT = {}
    # D是探索系数
    D = 0.5
    # 依次遍历每个根节点的候选孩子节点
    for i in range(len(tree_node.children)):
        if tree_node.children[i].is_select is False:
            # 也就是说都还没有被访问过,没有被访问过的节点的UCT的值都为无穷大
            children_UCT[str(i)] = float("inf")
        else:
            # 否则就使用UCT公式来计算
            left = tree_node.children[i].value / tree_node.children[i].visit_times
            right = D * ((math.log(tree_node.visit_times) / tree_node.children[i].visit_times) ** 0.5)
            children_UCT[str(i)] = (left + right)
    # 将当前节点UCT值最大的孩子节点的下标返回
    # print(children_UCT)
    snI = int(list(children_UCT.keys())[int(list(children_UCT.values()).index(max(children_UCT.values())))])
    # 输出下标
    # print(snI)
    # 获取下一个状态,也就是会更新一下底层网络的状态。
    (next_state, reward) = sample_next_state(tree_node, int(snI))
    # 如果下一个状态是终端状态,返回奖励值,就说最后一个都映射完了,后续没有还需要映射的VNF了
    if not next_state["current_chain"]["vnf_kind"]:  # next_state
        # next_state后面已经没有需要再扩展或者映射的节点了,我将状态赋值给当前节点,
        if tree_node.children[snI].state is not None:
            return float("inf")
        else:
            tree_node.children[snI].state = next_state
            tree_node.children[snI].is_select = True
            tree_node.children[snI].value += reward
            tree_node.children[snI].visit_times += 1
            return reward
    # 如果所选择的孩子节点的状态为None也就是说该孩子节点还没有加入到树中
    if tree_node.children[snI].state is None:
        # 如果当前选中的孩子节点的状态为空,我要对其进行仿真模拟,从选择一直到结束,计算出最后的reward
        tree_node.children[snI].state = next_state
        tree_node.children[snI].is_select = True
        # 对其进行模拟,计算出最终的奖励,已经计算出来了最终奖励
        reward = simulate_end(tree_node.children[snI])
    # 首先找出当前tree_node的候选孩子节点中的值最大的那个,
    # 如果是初始情况,或者说是小于扩展次数的情况下,还需要扩展新的节点
    else:
        reward = simulate(tree_node.children[snI])
    tree_node.children[snI].value += reward
    tree_node.children[snI].visit_times += 1
    return reward


def simulate_end(tree_node):
    """
    模拟从现在开始走到最后,并且计算出最终的奖励函数。
    :param tree_node:
    :return:
    """
    # 获得当前的策略π,也就是说每个VNF的映射情况。我应该怎么获得呢。我可以遍历在当前情况下的VNF
    current_node = copy.deepcopy(tree_node)
    while current_node.state["current_chain"]["vnf_kind"]:
        current_vnf = {"vnf_kind": current_node.state["current_chain"]["vnf_kind"][0],
                       "vnf_computation": current_node.state["current_chain"]["vnf_computation"][0]
                       }
        # 从当前可以为当前vnf进行映射的服务器中随机选择一个来作为映射的结果。
        candidate_server = select_candidate_vm(current_vnf["vnf_kind"],
                                               current_vnf["vnf_computation"],
                                               current_node.state["current_chain"]["bandwidth"],
                                               current_node.state["SN_state"])
        # 如果当前的VNF的候选服务器集为空就返回奖励值为﹣无穷大
        if not candidate_server:
            return float("-inf")
        # 在这里使用启发式的算法来进行选择 计算每个符合要求的候选服务器的GRC来进行映射。
        # 使用LRC的值来进行排序。
        server_num = random.choice(candidate_server)

        vm_num = None
        # 首先我要判断在当前的服务器中是否存在和当前VNF类型相同的虚拟机,并且返回该虚拟机的下标
        for i in range(len(current_node.state["SN_state"][server_num]["VM"])):
            if (current_vnf["vnf_kind"] == current_node.state["SN_state"][server_num]["type"][i]) and (
                    current_vnf["vnf_computation"] <= current_node.state["SN_state"][server_num]["VM"][i]):
                vm_num = i
                break
        # 否则使用新的虚拟机,返回服务器中第一次出现的虚拟机的类型为0的下标
        if vm_num is None:
            vm_num = current_node.state["SN_state"][server_num]["type"].index(0)
        del current_node.state["current_chain"]["vnf_computation"][0]
        del current_node.state["current_chain"]["vnf_kind"][0]
        insert_result = {"server_num": server_num,
                         "vm_num": vm_num}
        current_node.state["current_chain"]["attached"].append(insert_result)
        current_node.state["SN_state"][server_num]["VM"][vm_num] -= current_vnf["vnf_computation"]
        current_node.state["SN_state"][server_num]["type"][vm_num] = current_vnf["vnf_kind"]
    policy = []
    for i in range(len(current_node.state["current_chain"]["attached"])):
        policy.append(current_node.state["current_chain"]["attached"][i])
    # 通过将当前网络状态以及嵌入的策略传入来进行链路映射。
    current_node.state["SN_state"], reward, link_map = link_embedding(copy.deepcopy(current_node.state["SN_state"]),
                                                                      current_node.state["current_chain"]["bandwidth"],
                                                                      policy)
    if reward == float("-inf"):
        reward = -3
    return reward


# 取下一个节点的状态
def sample_next_state(tree_node, snI):
    # 根据传递过来的树的节点,以及孩子节点下标。
    current_vnf = {"vnf_kind": tree_node.state["current_chain"]["vnf_kind"][0],
                   "vnf_computation": tree_node.state["current_chain"]["vnf_computation"][0]
                   }
    current_state = copy.deepcopy(tree_node.state)  # 深度复制,复制的是值,改变current_state的值不会改变tree_node的值
    child = tree_node.children[snI]
    # 得到要映射的服务器的编号
    server_num = child.server_num
    # 得到映射前的状态
    # 根据服务器编号对其进行映射 我首先判断在服务器的VM的类型是否有和我一样的类型,有的话判断CPU是否满足 满足的话就可以映射。
    # 否则,选择一台新的虚拟机来进行映射。
    vm_num = None
    # 首先我要判断在当前的服务器中是否存在和当前VNF类型相同的虚拟机,并且返回该虚拟机的下标
    for i in range(len(current_state["SN_state"][server_num]["VM"])):
        if (current_vnf["vnf_kind"] == current_state["SN_state"][server_num]["type"][i]) and (
                current_vnf["vnf_computation"] <= current_state["SN_state"][server_num]["VM"][i]):
            vm_num = i
            break
    # 否则使用新的虚拟机,返回服务器中第一次出现的虚拟机的类型为0的下标
    if vm_num is None:
        vm_num = current_state["SN_state"][server_num]["type"].index(0)
    # 现在我已经知道了我要映射到哪个服务的那个虚拟机上面,进行状态更新
    # 将server_num上的server_num的类型更改为我当前要处理的VNF的类型,同时也要更新计算能力。
    # 将服务链的第一个VNF去除,因为后续不再需要映射这个了,同时将vnf_computation也去除掉。
    del current_state["current_chain"]["vnf_computation"][0]
    del current_state["current_chain"]["vnf_kind"][0]
    insert_result = {"server_num": server_num,
                     "vm_num": vm_num}
    current_state["current_chain"]["attached"].append(insert_result)
    current_state["SN_state"][server_num]["VM"][vm_num] -= current_vnf["vnf_computation"]
    current_state["SN_state"][server_num]["type"][vm_num] = current_vnf["vnf_kind"]
    if current_state["current_chain"]["vnf_kind"]:
        # 还没有结束,还要继续映射。
        return current_state, 0
    # 所有VNF都已经完成了映射,要进行最终的映射计算奖励值
    else:
        # 遍历获得当前的VNF映射策略π。使用最短路径的算法解决链路映射问题,并且计算出reward
        # 从根节点获得所有的映射策略,然后根据映射策略来进行链路映射,计算出reward
        current_state["SN_state"], reward, link_map = link_embedding(copy.deepcopy(current_state["SN_state"]),
                                                                     current_state["current_chain"]["bandwidth"],
                                                                     current_state["current_chain"]["attached"])
        if reward == float("-inf"):
            reward = -3
        return current_state, reward


def decision(tree_node, expand_num, vnI):
    """
    实现蒙特卡洛搜索树决策的方法
    :param vnI: vnf的下标
    :param tree_node:
    :param expand_num:
    :return:
    """
    # 为当前的根节点创建候选子节点,候选子节点就是符合当前根节点要进行映射的vnf的服务器集合
    # 如果当前根节点的子节点中不存在候选映射的节点。
    # if not tree_node.children:
    #     tree_node.create_candidate_children(vnI)
    # # 如果当前的根节点后面不存在孩子节点就返回错误
    # if not tree_node.children:
    #     return "*", "*"
    # 否则判断是否还在还在计算预算里面要执行的操作
    # expand_num = 5 * len(tree_node.children)
    while expand_num > 0:
        # 进行仿真测试并且返回reward
        reward = simulate(tree_node)
        # 如果奖励对应的是拒绝嵌入该服务链的则返回一个终止的行动
        if reward == float("-inf"):  # 失败的时候奖励应该是-inf
            return "*", "*"
        if reward == float("inf"):
            tree_node.value += reward
            tree_node.visit_times += 1
            break
        # 否则更新当前节点的奖励和访问次数
        tree_node.value += reward
        tree_node.visit_times += 1
        expand_num -= 1
    dit = {}
    # 返回当前节点的孩子节点中收益/访问次数比值最大的那个节点的下标。
    # 如果孩子节点被选择过,也就是is_select是true就说明他被加入到
    # 了蒙特卡洛搜索树中, 成为了真实的根节点的孩子节点。
    for i in range(len(tree_node.children)):
        if tree_node.children[i].is_select:
            # 值就等于孩子节点的值/孩子节点的访问次数
            # value = tree_node.children[i].value / tree_node.children[i].visit_times
            value = tree_node.children[i].value
            # str(i)是记录值所对应的孩子节点的下标
            dit[str(i)] = value
    max_index = list(dit.keys())[list(dit.values()).index(max(list(dit.values())))]
    # 返回孩子节点中对应的服务器的编号,以及孩子节点
    return tree_node.children[int(max_index)].server_num, tree_node.children[int(max_index)].num


def mcts(current_chain, SN, expand_num):
    """
    :param expand_num:
    :param current_chain: 当前要进行处理的服务链
    :param SN: 当前服务器的状态
    :return: 若返回success 表示成功嵌入, 若返回reject表示无法成功要拒绝
    """
    # 设置初始状态,初始状态为当前需要进行处理的服务链,和当前的底层网络状态
    state = {"current_chain": copy.deepcopy(current_chain),
             "SN_state": copy.deepcopy(SN)}
    # 创建树的根节点,初始状态为传入到函数中的初始状态,父节点为None,节点标号为0开始,服务的编号一开始是没有的所以也为None
    root = TreeNode(None, state, 0, None)
    # 当前服务链的映射情况,每个VNF分别要映射到哪些服务器上面。
    nodesMap = []
    terminate = False
    vnI = 0


    # 只要不是终端状态就要继续执行
    while not terminate:
        # 调用蒙特卡洛树搜索算法来进行决策.依次为服务链上的VNFs决定映射到哪个服务器上。
        # 在实际映射的时候我会尽量选择重用,在不能重用的情况下我就选择开启新的虚拟机。
        # decision方法返回的结果是服务器的编号。所以我的nodesmap里面也是服务器的编号。
        server_num, child_num = decision(root, expand_num, vnI)
        # 如果返回的结果不是*则说明存在映射的服务器,可以进行映射
        if server_num != "*":
            nodesMap.append(server_num)
            # 如果当前是最后一个节点的话,则结束服务器的映射,进行链路的映射。
            if not (root.children[int(child_num)].state["current_chain"]["vnf_kind"]):
                terminate = True
            else:
                # 否则把当前选择的节点作为根节点进行接下来的映射。
                root = root.children[int(child_num)]
                # 此时要进行第二个vnf的映射
                vnI += 1
        # 否则也就是说当前没有服务器可以来进行映射,我提前结束
        else:
            terminate = True


    # 如果节点映射集合的长度等于服务链的长度,也就是说每个VNF都有对应的服务器可以来进行映射
    if len(nodesMap) == len(state["current_chain"]["vnf_kind"]):
        # 使用得到的nodesMap来进行虚拟机的选择并且进行链路映射
        # 但是这样会存在一个问题,就是我这样选择的结果和我模拟时候选择的情况会有点不一样吧。
        # 都分配好以后就进行链路的映射,使用最短路径的方法来进行映射。
        # 最后需要更新一下状态,最后返回告诉前一个模块,该服务链映射成功了
        temp_chain = copy.deepcopy(current_chain)
        policy = []
        temp_nodesMap = copy.deepcopy(nodesMap)
        while temp_chain["vnf_kind"]:
            current_vnf = {"vnf_kind": temp_chain["vnf_kind"][0],
                           "vnf_computation": temp_chain["vnf_computation"][0]
                           }
            server_num = temp_nodesMap[0]
            vm_num = state["SN_state"][server_num]["type"].index(current_vnf["vnf_kind"])
            del temp_chain["vnf_computation"][0]
            del temp_chain["vnf_kind"][0]
            insert_result = {"server_num": server_num,
                             "vm_num": vm_num}
            current_chain["attached"].append(insert_result)
            state["SN_state"][server_num]["VM"][vm_num] -= current_vnf["vnf_computation"]
            del temp_nodesMap[0]
        for i in range(len(current_chain["attached"])):
            policy.append(current_chain["attached"][i])
        print(policy)
        state["SN_state"], reward, link_map = link_embedding(copy.deepcopy(state["SN_state"]),
                                                             current_chain["bandwidth"], policy)
        if reward != float("-inf"):
            print("映射成功")
            print(reward)
            current_chain["success"] = True
            current_chain["link_map"] = link_map
            return copy.deepcopy(state["SN_state"]), reward
        else:
            print("该服务链映射失败了")
            return copy.deepcopy(SN), None
    else:
        # 拒绝该服务链的映射请求。
        print("该服务链嵌入失败了")
        return copy.deepcopy(SN), None
        # 返回状态,并提示嵌入失败了
