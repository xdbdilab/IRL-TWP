import argparse
import os
import pickle
import sys
import time
import tracemalloc
import gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy_concat import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from GAIL_TWP_env import GAIL_TWP_Env
from core.agent import Agent
from tensorboardX import SummaryWriter
import torch


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
env_parser.add_argument('--file-num', type=int, default=70000, metavar='G',
                    help='the number of files of trajectories')
env_parser.add_argument('--data-per-file', type=int, default=1, metavar='G',
                    help='num of data per file')
env_args = env_parser.parse_args()

parser = argparse.ArgumentParser(description='PyTorch GAIL parameters')
parser.add_argument('--env-name', default="ResourcePred", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--experiment-id', default="gail_experiment_0726", metavar='G',
                    help='the id of this experiment')
parser.add_argument('--traj-path', default="IRL_TWP/traj_100_train/", metavar='G',
                    help='the path of the traj loaded')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--traj-load-upperbound', type=int, default=100000, metavar='G',
                    help='the loading upper bound of the traj')
parser.add_argument('--traj-load-lowerbound',type=int, default=70000, metavar='G',
                    help='the loading lower bound of the traj')
parser.add_argument('--traj-load-num',type=int, default=3000, metavar='G',
                    help='the loading number of the traj')
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
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--lambda_gp', type=float, default=1, metavar='G',
                    help='gradient penalty lambda (default: 1)')
parser.add_argument('--tau', type=float, default=0.99, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--value-learning-rate', type=float, default=1e-5, metavar='G',
                    help='learning rate for value network (default: 3e-4)')
parser.add_argument('--policy-learning-rate',type=float, default=3e-5, metavar='G',
                    help='learning rate for policy network')
parser.add_argument('--discrim-learning-rate', type=float, default=3e-4,metavar='G',
                    help='learning rate for discrim network')
parser.add_argument('--clip-epsilon', type=float, default=0.1, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=20000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=500, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--error-memo-length', type=int, default=10, metavar='G',
                    help='length of error to be memoized')
parser.add_argument('--stop-error', type=float, default=2.0, metavar='G',
                    help='avg error to stop and save')
parser.add_argument('--gpu-index', type=int, default=2, metavar='N')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""Set the Writter"""
writer = SummaryWriter('runs/'+ args.experiment_id)
"""environment"""
env = GAIL_TWP_Env(env_args.time_length, env_args.machine_state_dim, env_args.task_state_dim
                      , env_args.machine_usage_dim, env_args.machine_usage_bias,
                      env_args.file_num, env_args.data_per_file,
                      args.traj_path, None)
state_dim = (env_args.machine_state_dim + env_args.task_state_dim + env_args.machine_usage_dim * env_args.time_length)
readin_state_dim = (env_args.machine_state_dim + env_args.task_state_dim + (env_args.machine_usage_dim+1) * env_args.time_length)
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
traj_load_range = [args.traj_load_lowerbound, args.traj_load_upperbound]
traj_load_num = args.traj_load_num
# running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(env_args.time_length*env_args.machine_usage_dim, env_args.task_state_dim, env.action_space.shape[0],
                            device=device,usage_hidden_size=(64, 64, 64), task_hidden_size=(256, 256, 256),
               output_hidden_size=128, activation='relu')
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)
discrim_net = Discriminator(state_dim + action_dim,hidden_size=(256,256,256),activation='relu')
discrim_criterion = nn.BCELoss()
to_device(device, policy_net, value_net, discrim_net, discrim_criterion)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.policy_learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.value_learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.discrim_learning_rate)

# optimization epoch number and batch size for TRPO
optim_epochs = 10
optim_batch_size = 64

# load trajectory
def load_random_traj(load_num,load_range):
    expert_trajs = []
    traj_path = args.traj_path
    indexes = np.random.randint(load_range[0], load_range[1], size=load_num)
    for i in indexes:
        data_path = traj_path + str(i) + '_pkl'
        with open(data_path, 'rb') as f:
            expert_trajs.append(pickle.load(f))
    expert_trajs = np.stack(expert_trajs)
    # index = np.append(np.arange(0, 54), 54 + 0)
    target_action_seq = expert_trajs[:, :, readin_state_dim + env_args.machine_usage_bias] * 100
    expert_traj = []
    for states, actions in zip(expert_trajs, target_action_seq):
        state_seq_tmp = np.array(states[:, :readin_state_dim])
        usage_seq_tmp = state_seq_tmp[:, :(env_args.machine_usage_dim+1) * env_args.time_length]
        usage_seq_tmp = np.reshape(usage_seq_tmp, (-1, env_args.time_length, (env_args.machine_usage_dim+1)))
        usage_seq_tmp = usage_seq_tmp[:, :, env_args.machine_usage_bias]
        # usage_seq_tmp = np.reshape(usage_seq_tmp, (-1))
        task_tmp = state_seq_tmp[:, -env_args.task_state_dim:]
        state_seq_tmp = []
        for task, usage in zip(task_tmp, usage_seq_tmp):
            state_seq_tmp.append(np.append(task, usage))
        state_seq_tmp = np.array(state_seq_tmp)
        tmp = []
        for state, action in zip(state_seq_tmp, actions):
            tmp.append(np.append(state, action / 100))
        expert_traj.append(np.array(tmp))
    expert_traj = np.array(expert_traj)
    expert_traj = np.reshape(np.array(expert_traj), (-1, state_dim + action_dim))
    return expert_traj

# define the summary writer
writer.add_text('Env args', str(env_args))
writer.add_text('Training args', str(args))

def expert_reward(state, action):
    state_action = tensor(np.hstack([state, action]), dtype=dtype)
    # print(discrim_net(state_action)[0].item())
    with torch.no_grad():
        return -math.log(np.clip(discrim_net(state_action)[0].item(),0.0001,1000000))


"""create agent"""
agent = Agent(env, policy_net, device, custom_reward=expert_reward,
              running_state=None, render=args.render, num_threads=args.num_threads)


def update_params(batch, expert_traj, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)
    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(decive)
        d_interpolates = D(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).requires_grad_(False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    """update discriminator"""
    for _ in range(1):
        expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
        generated_state_actions = torch.cat([states, actions], 1).to(dtype).to(device)
        g_o = discrim_net(generated_state_actions)
        e_o = discrim_net(expert_state_actions)
        # Gradient penalty
        # gradient_penalty = compute_gradient_penalty(discrim_net, expert_state_actions.data, generated_state_actions.data)
        # Adversarial loss

        optimizer_discrim.zero_grad()
        # discrim_loss = -torch.mean(e_o) + torch.mean(g_o) + env_args.lambda_gp * gradient_penalty
        discrim_loss = -torch.mean(e_o) + torch.mean(g_o)
        discrim_loss.backward()
        optimizer_discrim.step()
        for p in discrim_net.parameters():
            p.data.clamp_(-1,1)
    """perform PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)
            # trpo_step(policy_net, value_net, states_b, actions_b, returns_b, advantages_b, args.max_kl, args.damping, args.l2_reg,
            #           use_fim=True)

    

def main_loop():

    expert_traj = load_random_traj(traj_load_num, traj_load_range)
    error_array = []
    for i_iter in range(args.max_iter_num):
        discrim_net.to(torch.device('cpu'))
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        discrim_net.to(device)

        t0 = time.time()
        update_params(batch, expert_traj, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}\tmse_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_c_reward'], log['avg_reward'], log['avg_mse']))
            writer.add_scalar('mse_loss', log['avg_mse'], i_iter)
            writer.add_scalar('R_avg', log['avg_reward'], i_iter)
            writer.add_scalar('expert_R_avg', log['avg_c_reward'], i_iter)
            writer.add_scalar('T_sample', log['sample_time'], i_iter)
            writer.add_scalar('T_update', t1-t0, i_iter)
            error_array.append(log['avg_mse'])
            if len(error_array) > args.error_memo_length:
                recent_error = error_array[-args.error_memo_length:]
                if log['avg_mse']<2.4:
                    to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
                    filename = 'IRL_TWP/GAIL/gail/assets/learned_models/' + args.experiment_id + \
                               '_{}.p'.format(log['avg_mse'], i_iter + 1)
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, 'wb') as f:
                        pickle.dump((policy_net, value_net, discrim_net), f)
                    to_device(device, policy_net, value_net, discrim_net)
        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            filename = 'GAIL/gail/assets/learned_models/' + args.experiment_id + \
                       '_{}.p'.format(log['avg_mse'], i_iter + 1)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump((policy_net, value_net, discrim_net), f)
            to_device(device, policy_net, value_net, discrim_net)
            expert_traj = load_random_traj(traj_load_num, traj_load_range)
        # if log['R_avg'] <0:
        #     break
        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
writer.close()
