from IRL_TWP_env import IRL_TWP_Env
import pickle
import argparse
import numpy as np
import torch
from torch import tensor
from torch import nn
from models.mlp_policy_concat import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
import matplotlib.pyplot as plt
from tqdm import tqdm
env_parser = argparse.ArgumentParser(description='Resource prediction parameters')

env_parser.add_argument('--time-length', type=int, default=20, metavar='G',
                    help='the time length of the policy network, the length of state recorded for a prediction')
env_parser.add_argument('--machine-state-dim', type=int, default=0, metavar='G',
                    help='the dimension of machine static information')
env_parser.add_argument('--task-state-dim', type=int, default=128, metavar='G',
                    help='the dimension of each running task')
env_parser.add_argument('--machine-usage-dim', type=int, default=1, metavar='G',
                    help='the dimension of machine usage per time')
env_parser.add_argument('--machine-usage-bias', type=int, default=0, metavar='G',
                    help='machine usage bias 0 for cpu 1 for memory')
env_parser.add_argument('--file-num', type=int, default=20000, metavar='G',
                    help='the number of files of trajectories')
env_parser.add_argument('--data-per-file', type=int, default=1, metavar='G',
                    help='num of data per file')
env_args = env_parser.parse_args()

parser = argparse.ArgumentParser(description='PyTorch GAIL parameters')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--env-name', default="ResourcePred", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
# parser.add_argument('--model-path',
#                     default="assets/learned_models/ResourcePrediction_gail.p",
#                     metavar='G',
#                     help='path of the expert trajectories')
parser.add_argument('--model-path',
                    metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--lambda_gp', type=float, default=1, metavar='G',
                    help='gradient penalty lambda (default: 1)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=2, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=20000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=1000, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=1, metavar='N')
args = parser.parse_args()


env = IRL_TWP_Env(env_args.time_length, env_args.machine_state_dim, env_args.task_state_dim
                      , env_args.machine_usage_dim, env_args.machine_usage_bias,
                      env_args.file_num, env_args.data_per_file,
                      '/27T/TE_TWP/traj_500_test/', None, cold_start=False)

filename = '/27T/TE_TWP/GAIL/gail/assets/learned_models/gail_experiment_0726-2_0.765.p'
with open(filename, 'rb') as f:
    policy_net, value_net, discrim_net = pickle.load(f)
mse_array = []
mse_episode = 0
error_array = []
total_error_array = []
for i in tqdm(range(100)):
    state = env.reset(selected_index=i)
    done = False
    reward_episode = 0
    if len(error_array) > 0:
        total_error_array.append(error_array)
    error_array = []
    mse_array.append(mse_episode)
    mse_episode = 0

    while not done:
        state_var = tensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy_net.select_action(state_var)[0].numpy()
            action.astype(np.float64)
            next_state, reward, done, mse = env.step(action)
            error_array.append(mse)
            reward_episode += reward
            mse_episode += mse
total_error_array.append(error_array)
mse_array.append(mse_episode)


total_error_array = np.array(total_error_array)
print(np.mean(total_error_array))
print(np.shape(total_error_array))
total_error_array=np.sum(total_error_array, axis=0)
with open('error_pickle_100_20000_0.765-9','wb') as f:
    pickle.dump(total_error_array, f)

print(np.shape(total_error_array))

