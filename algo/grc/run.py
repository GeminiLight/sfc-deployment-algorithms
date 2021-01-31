import os
import sys
import time
import numpy as np
import pandas as pd

file_path_dir = os.path.abspath('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(file_path_dir)

from data.physical_network import PhysicalNetwork
from data.virtual_network import VirtualNetwork, VirtualNetworksSimulator
from algo.grc.env import GRCEnv

def main():
    ### Initialization ###
    grc_d = 0.1
    env = GRCEnv(grc_d=grc_d)

    # [2000, True, 1749, 0.8745, 422.675, 726, 912, 0.7615484473096039, 44]
    vns_num = 2000
    min_length = 2
    max_length = 15
    min_request = 2
    max_request = 30
    arrival_rate = 14
    aver_lifetime = 400
    vns_simulator = VirtualNetworksSimulator(vns_num, min_length=min_length, max_length=max_length, min_request=min_request, max_request=max_request, arrival_rate=arrival_rate, aver_lifetime=aver_lifetime)
    
    ### Data Source ###
    # generate random date
    # vns_simulator.renew_vns()
    # events = vns_simulator.renew_events()

    # load from dataset
    vns_simulator.load_from_csv(vns_data_file=f'vns_data_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}.csv', 
                        events_data_file=f'events_data_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}.csv')
    events = vns_simulator.events

    # record
    place_id = 0
    place_records = []
    time_records = []
    ### Start Running ###
    for e_id in range(len(events)):
        # Event Info
        vn_id, _, type = events.loc[e_id]
        vn = vns_simulator.vns[int(vn_id)]

        # leave event
        if type==0:
            env.release_resources(vn)
            print("release resources: ", vn_id)
        # enter event
        else:
            place_id += 1
            t1 = time.time()
            result = env.step(vn)
            t2 = time.time()
            running_time = t1 - t2
            # SUCCESS
            if result:
                revenue_to_cost = env.total_revenue / env.total_cost
                time_record = [place_id, vn.nodes_num, running_time]
            aver_revenue = env.total_revenue / place_id
            acceptance_ratio = env.success / place_id
            place_record = [place_id, result, env.success, acceptance_ratio, aver_revenue, vn.revenue, vn.cost, revenue_to_cost, env.inservice]
            place_records.append(place_record)
            time_records.append(time_record)
            print(place_record)
    
    ### Save Record ###
    # place record
    place_columns = ['place_id', 'result', 'success', 'acceptance_ratio', 'aver_revenue', 'revenue', 'cost', 'revenue_to_cost', 'inservice']
    pr = pd.DataFrame(data=place_records, columns=place_columns)
    pr.to_csv(f'record/grc/grc_place_record_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}_{grc_d}.csv')
    # running time record
    time_columns = ['place_id', 'nodes_size', 'running_time']
    tr = pd.DataFrame(data=time_records, columns=time_columns)
    tr.to_csv(f'record/grc/grc_time_record_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}_{grc_d}.csv')
    
    ### reset ###
    env.reset()
    print('Finished')

if __name__ == '__main__':
    main()
