import pandas as pd
import pickle
from tqdm import tqdm

## Readin parameters
# PATHS = ['batch_instance', 'batch_task', 'machine_meta', 'machine_usage']
chunkSize = 2 ** 21
original_path = 'IRL_TWP/batch_task_query.csv'
MACHINE_NUM = 4034

## Read original files
query_df = pd.read_csv(original_path, engine='c')
query_df.set_index(['job_name', 'task_name'], inplace=True, drop=False)
print(query_df.head())
for i in range(1, 4034):
    ## Machine extraction setting
    machine_name = 'm_' + str(i)
    print(machine_name)
    task_path = 'IRL_TWP/machine_tasks/' + machine_name + '_task.csv'
    df = pd.read_csv(task_path, header=None, names=['instance_name', 'task_name', 'job_name', 'task_type', 'start_time', 'end_time', 'machine_id'])
    df.sort_values(by=['start_time', 'end_time'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['plan_cpu'] = '-1'
    df['plan_mem'] = '-1'
    for x in tqdm(range(0, len(df.index))):
        job_name = df['job_name'].iloc[x]
        task_name = df['task_name'].iloc[x]
        temp = query_df.loc[job_name, task_name]
        plan_cpu = temp['plan_cpu']
        plan_mem = temp['plan_mem']
        df.loc[x, 'plan_cpu'] = plan_cpu
        df.loc[x, 'plan_mem'] = plan_mem
    print(df.head(1))
    print(df.tail(1))
    save_path = 'IRL_TWP/machine_tasks_with_resource/' + machine_name + \
                '_task_with_resource.csv'
    df.to_csv(save_path)



