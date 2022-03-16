import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import torch
import sys
from utils import *
from scipy import sparse
from TFE_LSTM_model import task_LSTM
import random

sample_num = 1
time_length = 2
time_interval = 100
time_period = time_length * time_interval
tasksize_per_interval = 50
tasknum_upperbound = 255
tasknum_lowerbound = 30
null_tolerance = 2
state_usage_recorded = 20
samples_per_file = 1
resource_height = 1000


# task_type
# start_time
# end_time
# plan_cpu
# plan_mem

# Index(['task_type', 'start_time', 'end_time', 'machine_id', 'plan_cpu',
#        'plan_mem'],
#       dtype='object')
# Index(['cpu_util_percent', 'mem_util_percent', 'disk_io_percent',
#        'machine_id', 'mem_gps', 'mkpi', 'net_in', 'net_out'],
#       dtype='object')

file_num = 0
expert_trajs = []
machine_meta_path = 'machine_meta.tar.gz'
machine_meta = pd.read_csv(machine_meta_path, compression='gzip', header=None,
        names=['machine_id', 'time_stamp', 'failure_domain_1', 'failure_domain_2', 'cpu_num', 'mem_size', 'status'])
machine_used_array = []
task_usage_model = pickle.load(open('task_usage_model_0723.p', 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(2)
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# filter the available machines
for i in range(1, 4034):
    machine_name = 'm_' + str(i)
    machine_usage_path = 'machine_usage_sorted/' + machine_name + \
                         '_usage_sorted.csv'
    usage_df = pd.read_csv(machine_usage_path)
    if len(usage_df)-time_length > time_length:
        machine_used_array.append(i)



while file_num < sample_num:
    i = random.choice(machine_used_array)
    machine_name = 'm_' + str(i)
    print(machine_name)

    # read the csv and drop the unused
    machine_info = machine_meta[machine_meta.machine_id == machine_name]
    machine_info.drop(['machine_id', 'time_stamp', 'status', 'failure_domain_1', 'failure_domain_2'], axis=1, inplace=True)
    machine_info.reset_index(drop=True, inplace=True)
    task_path = 'machine_tasks_with_resource/' + machine_name + \
                '_task_with_resource.csv'
    machine_usage_path = 'machine_usage_sorted/' + machine_name + \
                '_usage_sorted.csv'
    task_df = pd.read_csv(task_path)
    usage_df = pd.read_csv(machine_usage_path)
    usage_df.drop(['machine_id', 'mem_gps', 'mkpi', 'net_in', 'net_out', 'disk_io_percent'], axis=1, inplace=True)
    task_df.drop(['instance_name', 'task_name', 'job_name', 'machine_id'], axis=1, inplace=True)
    # print(task_df.head())

    # extract the data in each time period random pick
    j = random.randint(state_usage_recorded+2, len(usage_df)-time_length)
    # for j in tqdm(range(1, len(usage_df)-time_length, sample_interval)):
        # set the sample time range
    time_started = j * time_interval
    time_ended = (j+time_length-1) * time_interval
    # query the tasks in range
    filtered_tasks = task_df[(task_df.start_time >= time_started) & (task_df.start_time <= time_ended) |
                  (task_df.end_time >= time_started) & (task_df.end_time <= time_ended)]
    filtered_tasks.reset_index(drop=True, inplace=True)
    # delete the data with too few tasks
    if len(filtered_tasks.index) <= 10:
        continue
    # query the usages in range
    filtered_usage = usage_df[(usage_df.time_stamp >= time_started-state_usage_recorded*time_interval) & (usage_df.time_stamp <= time_ended)]
    # fill the samples with missing usage
    if len(filtered_usage.index) < time_length:
        continue
        # for time in range(time_started, time_ended+time_interval, time_interval):
        #     temp = filtered_usage[filtered_usage.time_stamp == time]
        #     if temp.empty:
        #         fulfiller = pd.Series({'time_stamp': time, 'cpu_util_percent': -1,
        #                                'mem_util_percent': -1})
        #         filtered_usage = filtered_usage.append(fulfiller, ignore_index=True)
    # reset the time with time_started = 0
    with pd.option_context('mode.chained_assignment', None):
        filtered_tasks.start_time = filtered_tasks.start_time.apply(lambda x: x - time_started)
        filtered_tasks.end_time = filtered_tasks.end_time.apply(lambda x: x - time_started)
        filtered_tasks.plan_cpu = filtered_tasks.plan_cpu.apply(lambda x: x/100)
        filtered_usage.time_stamp = filtered_usage.time_stamp.apply(lambda x: x - time_started)
        filtered_usage.cpu_util_percent = filtered_usage.cpu_util_percent.apply(lambda x: x/100)
        filtered_usage.mem_util_percent = filtered_usage.mem_util_percent.apply(lambda x: x/100)
    # clean and drop the rebundant data
    filtered_usage_sorted = filtered_usage.sort_values(by=['time_stamp'])
    for index, rows in filtered_usage_sorted.iterrows():
        print(rows)
    filtered_usage_sorted.drop(['time_stamp'], axis=1, inplace=True)
    filtered_usage_sorted.reset_index(drop=True, inplace=True)
    filtered_tasks = filtered_tasks.loc[:, ~filtered_tasks.columns.str.contains("^Unnamed")]
    filtered_tasks.dropna(inplace=True)

    # state sequence generation
    state_seq = []

    # append the machine_state to the state
    with pd.option_context('mode.chained_assignment', None):
        machine_state = machine_info.loc[0]/100
    # if too many tasks running in a interval, del that sample
    tasknum_excess_flag = True
    running_tasks_seq = []
    usage_memo_seq = []
    for k in range(0, time_length):
        start_time = (k-1) * 10
        end_time = k * 10
        # append running tasks to each state
        running_tasks = filtered_tasks[(filtered_tasks.start_time >= start_time) &
                                       (filtered_tasks.start_time <= end_time) |
                  (filtered_tasks.end_time >= start_time) & (filtered_tasks.end_time <= end_time)]
        running_tasks.reset_index(drop=True, inplace=True)
        running_tasks_list = []
        for index, rows in running_tasks.iterrows():
            print(rows)
            # feature the task resource
            running_tasks_list.append(rows['plan_cpu'])
            running_tasks_list.append(rows['plan_mem'])
            running_tasks_list.append(rows['task_type']/12)
            running_tasks_list.append(math.log10(math.fabs(rows['end_time']-rows['start_time']+1)))
        if len(running_tasks_list) < 4 * tasksize_per_interval:
            running_tasks_list = np.pad(np.array(running_tasks_list),  (0, 4 * tasksize_per_interval - len(running_tasks_list)),
                               'constant', constant_values=0)
        else:
            tasknum_excess_flag = False
        running_tasks_seq.append(np.array(running_tasks_list))
    usage_memo_seq = filtered_usage_sorted.values
    running_tasks_seq = np.array(running_tasks_seq)
    # print(running_tasks_seq.shape)
    usage_memo_seq = np.array(usage_memo_seq)
    if usage_memo_seq.shape[0] != time_length+state_usage_recorded:
        continue
    usage_memo_seq = np.reshape(np.array(usage_memo_seq), (time_length+state_usage_recorded, -1))
    # print(usage_memo_seq.shape)
    # append usage_memo to each state
    if not tasknum_excess_flag:
        continue

    # truncate new state with task usage and machine usage
    for k in range(0, time_length):
        # generate the task and usage state of the formal state-usage-recorded info
        kth_task_state = running_tasks_seq[:k]
        kth_usage_state = usage_memo_seq[:k+state_usage_recorded]
        if k < state_usage_recorded:
            kth_task_state = np.reshape(kth_task_state, (-1, 4 * tasksize_per_interval))
            kth_usage_state = np.reshape(kth_usage_state, (-1, 2))
            for i_iter in range(k, state_usage_recorded):
                kth_task_state = np.append(kth_task_state, np.zeros((1, 4 * tasksize_per_interval)), axis=0)
                kth_usage_state = np.append(kth_usage_state, np.zeros((1, 2)), axis=0)
        else:
            kth_task_state = kth_task_state[-state_usage_recorded:]
            kth_usage_state = kth_usage_state[-state_usage_recorded:]
        kth_state = []
        task_states = []
        for task_state, usage_state in zip(kth_task_state, kth_usage_state):
            task_states.append(task_state)

            # take the machine state or not
            # temp = np.append(machine_state, usage_state)
            temp = usage_state

            kth_state.append(temp)

        # forward pass the task states
        task_states = np.array(task_states)
        with torch.no_grad():
            x = torch.from_numpy(task_states).type('torch.FloatTensor')
            x = x.view(-1, state_usage_recorded, 200)
            _, task_info = task_usage_model.forward(x.to(device))
        task_info = task_info.view(-1)
        # print(task_info.shape)
        kth_state = np.array(kth_state)
        kth_state = np.reshape(kth_state, (-1))
        kth_state = np.append(kth_state, task_info.cpu())
        kth_state = standardization(kth_state)
        state_seq.append(kth_state)
    state_seq = np.array(state_seq)
    state_seq = np.reshape(state_seq, (time_length, -1))

    # 100 * (2*20 + 128)
    # action sequence generate

    action_seq = []
    for index, rows in filtered_usage_sorted.iterrows():
        # take cpu and mem
        temp = np.array([rows.cpu_util_percent, rows.mem_util_percent])
        action_seq.append(temp)
    action_seq = np.array(action_seq)

    # expert traj generate and append
    expert_traj = []
    for state, action in zip(state_seq, action_seq):
        expert_traj.append(np.append(state, action))
        # total_state = time_recorded * (machine_state_length, task_state_length, machine_usage_length)
        # total_traj = total_state + action(2)
    expert_traj = np.array(expert_traj)
    traj_path = 'data/traj_500_test/' + str(file_num) + '_pkl'
    with open(traj_path, 'wb') as f:
        pickle.dump(expert_traj, f)
    print(file_num)

    expert_traj = []
    file_num += 1




