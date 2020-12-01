import os
import sys
import numpy as np
import pandas as pd

file_path_dir = os.path.abspath('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(file_path_dir)

from generator.physical_network import PhysicalNetwork
from generator.virtual_network import VirtualNetwork, VirtualNetworksSimulator
from algo.random.env import RandomEnv


def main():
    ### Initialization ###
    env = RandomEnv()
    
    vns_num = 2000
    min_length = 2
    max_length = 15
    min_request = 2
    max_request = 30
    arrival_rate = 20
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
    records = []

    ### Start Running ###
    for e_id in range(len(events)):
        # Event Info
        vn_id, time, type = events.loc[e_id]
        vn = vns_simulator.vns[int(vn_id)]

        # leave event
        if type==0:
            env.release_resources(vn)
            print("release resources: ", vn_id)
        # enter event
        else:
            place_id += 1
            result = env.step(vn)
            # SUCCESS
            if result:
                revenue_to_cost = env.total_revenue / env.total_cost
            aver_revenue = env.total_revenue / place_id
            acceptance_ratio = env.success / place_id
            record = [place_id, result, env.success, acceptance_ratio, aver_revenue, vn.revenue, vn.cost, revenue_to_cost, env.inservice]
            records.append(record)
            print(record)

    ### Save Record ###
    columns = ['place_id', 'result', 'success', 'acceptance_ratio', 'aver_revenue', 'revenue', 'cost', 'revenue_to_cost', 'inservice']
    r = pd.DataFrame(records, columns=columns)
    r.to_csv(f'record/random/rdm_record_{vns_num}_{min_length}-{max_length}_{min_request}-{max_request}_{arrival_rate}_{aver_lifetime}.csv')

    # reset
    env.reset()

if __name__ == '__main__':
    main()
    
