#-*- coding: utf-8 -*-
import argparse


parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')

### Environment ###
env_arg = add_argument_group('Environment')
env_arg.add_argument('--num_cpus', type=int, default=10, help='number of CPUs')
env_arg.add_argument('--num_vnfd', type=int, default=8, help='VNF dictionary size')
env_arg.add_argument('--env_profile', type=str, default="small_default", help='environment profile')

### Network ###
net_arg = add_argument_group('Network')
net_arg.add_argument('--embedding_dim', type=int, default=64, help='agent embedding dim')

net_arg.add_argument('--hidden_dim', type=int, default=32, help='agent GRU num_neurons')
net_arg.add_argument('--num_layers', type=int, default=1, help='agent GRU num_stacks')
net_arg.add_argument('--enc_units', type=int, default=64, help='agent encoder GRU units')
net_arg.add_argument('--dec_units', type=int, default=64, help='agent decoder GRU units')
net_arg.add_argument('--gcn_units', type=int, default=64, help='agent decoder GCN units')

net_arg.add_argument('---drl_gamma', type=float, default=0.95, help='drl_gamma')
net_arg.add_argument('---dropout_rate', type=float, default=0.2, help='dropout_rate')
net_arg.add_argument('---l2reg_rate', type=float, default=2.5e-4, help='l2reg_rate')

### Data ###
data_arg = add_argument_group('Data')
# service function chain
data_arg.add_argument('--vns_num', type=int, default=2000, help='sfcs_num')
data_arg.add_argument('--batch_size', type=int, default=16, help='batch size')

data_arg.add_argument('--vn_min_length', type=int, default=2, help='service chain min length')
data_arg.add_argument('--vn_max_length', type=int, default=15, help='service chain max length')
data_arg.add_argument('--vn_min_request', type=int, default=2, help='virsual network the maxium')
data_arg.add_argument('--vn_max_request', type=int, default=30, help='virsual network the maxium')
data_arg.add_argument('--vn_aver_lifetime', type=int, default=500, help='the average lifetime')
data_arg.add_argument('--vn_arrival_rate', type=int, default=20, help='virsual network the maxium')

# pgysical network
data_arg.add_argument('--pn_nodes_num', type=int, default=100)
data_arg.add_argument('--pn_edges_num', type=int, default=500)
data_arg.add_argument('--pn_wm_alpha', type=float, default=0.2)
data_arg.add_argument('--pn_wm_beta', type=float, default=0.5)
data_arg.add_argument('--pn_min_resoure', type=int, default=50)
data_arg.add_argument('--pn_max_resoure', type=int, default=100)


### Training ###
train_arg = add_argument_group('Training')
train_arg.add_argument('--run_mode', type=str, default='test', help='number of epochs')
train_arg.add_argument('--random_seed', type=int, default=1024, help='random seed')
train_arg.add_argument('--epoch_num', type=int, default=100, help='number of epochs')
train_arg.add_argument('--actor_lr', type=float, default=0.00025, help='agent learning rate')
train_arg.add_argument('--critic_lr', type=float, default=0.0005, help='agent learning rate')

#train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='lr1 decay step')
#train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='lr1 decay rate')

### Performance ###
# perf_arg = add_argument_group('Training')
# perf_arg.add_argument('--enable_performance', type=str2bool, default=False, help='compare performance against solver')

### Misc ###
misc_arg = add_argument_group('User options')
misc_arg.add_argument('--save_model', type=str2bool, default=False, help='save model')
misc_arg.add_argument('--load_model', type=str2bool, default=False, help='load model')

misc_arg.add_argument('--save_to', type=str, default='save/model', help='saver sub directory')
misc_arg.add_argument('--load_from', type=str, default='save/model', help='loader sub directory')
misc_arg.add_argument('--log_dir', type=str, default='summary/repo', help='summary writer log directory')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

if __name__ == "__main__":
    
    config, _ = get_config()
