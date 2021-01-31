
import os
import sys
import copy
import time
import numpy as np
import pandas as pd

import spektral as sk
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.layers import Layer, Embedding, Dense, GRU, Flatten, Softmax, Dropout, BatchNormalization, Conv1D
from spektral.layers import GraphConv


# import tf_geometric as tfg
# from tf_geometric.layers import GCN

# file_path_dir = os.path.dirname(os.path.dirname(__file__))
# path = os.path.join(file_path_dir, 'generator')
# path = os.path.join(file_path_dir, 'RL')
# sys.path.append(file_path_dir)

from RL.agent import Agent
from RL.env import PhysicalNetworkEnv, PNBatchEnv
from generator.sfc_batch_generator import SFCBatchGenerator
from generator.sfc_requests_simulator import SFCRequestsSimulator
from generator.physical_network_generator import PhysicalNetworkGenerator
from generator.physical_network_loader import PhysicalNetwork

from config import get_config
args, _ = get_config()

tf.random.set_seed(args.random_seed)


def generate_physical_network(auto_save=True, auto_draw=False):
    G = PhysicalNetworkGenerator(nodes_num=args.pn_nodes_num,
                    wm_alpha=args.pn_wm_alpha, wm_beta=args.pn_wm_beta, 
                    min_resoure=args.pn_min_resoure, max_resoure=args.pn_max_resoure,
                    auto_save=auto_save, auto_draw=auto_draw, random_seed=args.random_seed)


def main():
    ### Initialiaztion ###
    batch_env = PNBatchEnv(args.batch_size, args.pn_nodes_num, args.vn_max_length)
    cpu_benchmark = batch_env.batch_env[0].pn.max_cpu_resource
    bw_benchmark = batch_env.batch_env[0].pn.max_sum_bw_resource
    sfc_simulator = SFCRequestsSimulator(args.sfcs_num, args.batch_size, 
                                        args.vn_min_length, args.vn_max_length, 
                                        min_request=args.vn_min_request, max_request=args.vn_max_request, 
                                        cpu_benchmark=cpu_benchmark, bw_benchmark=bw_benchmark, random_seed=args.random_seed)
    ### Learn Mode ###
    log_dir = './tensorboard'
    save_dir = './save'
    agent = Agent(args.pn_nodes_num, args.vn_max_request+1, args.batch_size,
                    args.embedding_dim, args.enc_units, args.gcn_units, args.dec_units, 
                    dropout_rate=args.dropout_rate, l2reg_rate=args.l2reg_rate)
    checkpoint = tf.train.Checkpoint(agentActorModel=agent.actor, agentCriticModel=agent.critic, 
                                agentActorOpt=agent.a_opt, agentCriticOpt=agent.c_opt)
    summary_writer = tf.summary.create_file_writer(log_dir)
    # Train Mode
    if args.learn_mode == 'train':
        agent.drl_epsilon = 0
        # print('----------------load the model--------------')
        # checkpoint.restore(tf.train.latest_checkpoint('./save'))  # 从文件恢复模型参数
        tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
    # Test Mode
    else:
        # if os.path.exists(checkpoint_save_path+'.index'):
        #     print('----------------load the model--------------')
        # model.load_weights(checkpoint_save_path)
        print('----------------load the model--------------')
        checkpoint.restore(tf.train.latest_checkpoint(save_dir))  # 从文件恢复模型参数

    ### Start to train/test ###
    epoch = 20
    all_actor_loss  = []
    all_critic_loss = []
    all_reward = []
    step_id = 0
    batch_pn_state = batch_env.reset()
    for epoch_id in range(epoch):
        # Generate SFCs and Events
        sfcs = sfc_simulator.renew_sfcs()
        events = sfc_simulator.renew_events()
        events_num = len(events)
        placement_records = []
        for i in range(events_num):
            # event
            event = events.loc[i]
            sfc_id = event['sfc_id']
            event_type = event['type']

            # release event
            if event_type==0:
                for rid in range(len(placement_records)-1, -1, -1):
                    if placement_records[rid]['sfc_id'] == sfc_id:
                        plc_rst = placement_records[rid]
                        batch_env.release_resources(plc_rst)
                        print(f'epoch: {epoch_id: 5d}, type: {int(event_type): 2d}, sfc: {int(sfc_id): 4d}')
                        break
                continue
            
            # placement event
            step_id += 1
            # vn state
            batch_vn_info = sfcs[int(sfc_id)]

            # train/test
            actor_loss, critic_loss, placement_record = agent.learn(batch_env, batch_pn_state, batch_vn_info, learn_mode=args.learn_mode)

            batch_env.ready()
            placement_record['sfc_id'] = sfc_id
            placement_records.append(placement_record)
            
            print(f'epoch: {epoch_id: 5d}, type: {int(event_type): 2d}, sfc: {int(sfc_id): 4d}, search: {placement_record["search_stratery"]}'
                + '\n' + f'--- actor_loss: {actor_loss: 2.10f}, critic_loss: {critic_loss: 2.10f}')

            # record
            if args.learn_mode == 'train':
                with summary_writer.as_default():                                   # 指定记录器
                    tf.summary.scalar("train_actor_loss", actor_loss, step=step_id)      # 将当前损失函数的值写入记录器
                    tf.summary.scalar("train_critic_loss", critic_loss, step=step_id)    # 将当前损失函数的值写入记录器
                if step_id % 100 == 0:
                    path = checkpoint.save('./save/model.ckpt')
                    # path = checkpoint.save('./save/model.ckpt')
                    print("model saved to %s" % path)
                if step_id % 500 == 0:
                    agent.drl_gamma += 0.1
            if args.learn_mode == 'test':
                with summary_writer.as_default():                                   # 指定记录器
                    tf.summary.scalar("test_actor_loss", actor_loss, step=step_id)      # 将当前损失函数的值写入记录器
                    tf.summary.scalar("test_critic_loss", critic_loss, step=step_id)    # 将当前损失函数的值写入记录器

            
        # save info of record 
        pd_prs = pd.DataFrame.from_dict(placement_records)
        path = f'data/sfc/{epoch_id}_placement_results.csv'
        pd_prs.to_csv(path)
        sfc_simulator.save(sfcs=True, events=True, id=epoch_id)
        batch_env.reset()
    
    if args.learn_mode == 'train':
        with summary_writer.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)    # 保存Trace信息到文件（可选）

    
if __name__ == '__main__':
    args.learn_mode = 'test'
    main()
