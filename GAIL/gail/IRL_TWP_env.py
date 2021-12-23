import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import pickle
from utils import *
from torch import nn
import torch

class IRL_TWP_Env(gym.Env):

    def __init__(self, time_length, machine_state_dim, task_state_dim, machine_usage_dim,
                 machine_usage_bias, file_num, data_per_file, pickle_path, model_path,cold_start=False):

        # set the data parameter
        self.time_length = time_length
        self.machine_state_dim = machine_state_dim
        self.task_state_dim = task_state_dim
        self.machine_usage_dim = machine_usage_dim
        self.machine_usage_bias = machine_usage_bias
        self.state_len = (self.machine_state_dim + self.machine_usage_dim * self.time_length + self.task_state_dim)
        # readin state
        self.readin_state_len = self.state_len + 1 * self.time_length
        # set the action and observation space
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float)
        self.observation_space = \
            spaces.Box(low=-100, high=100, shape=(self.time_length, self.state_len), dtype=np.float)
        # set the data readin parameter
        self.file_num = file_num
        self.data_per_file = data_per_file
        self.traj_num = self.file_num * self.data_per_file
        self.pickle_path = pickle_path
        self.model_path = model_path
        self.cold_start = cold_start
        # set the embedding layer
        # set the state of different part
        self.machine_state = np.zeros(self.machine_state_dim)
        self.task_state = np.zeros(self.task_state_dim)
        self.machine_usage_state = np.zeros((self.machine_usage_dim, self.time_length))
        # set the RL sequence
        self.target_action = []
        self.state_seq = []
        self.action_seq = [[0]]
        self.task_seq = [] 
        self.usage_record = []
        self.machine_state_seq = []
        self.done = False
        self.step_num = 0
        self.step_len = 0
        self.expert_traj = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward_(self, target, action):
        #if action > 1 or action < 0:
        #    return -1
        abs_error = np.absolute(target - action)
        # print(abs_error)
        # print(np.clip(-abs_error+1, -1, 1))
        return np.clip(-abs_error+1, -1, 1).mean()
    # def state_extraction(self):

    def step(self, action):
        # print(self.target_action[self.step_num])
        reward = self.get_reward_(self.target_action[self.step_num], action)
        self.action_seq.append(action)
        # print(self.step_num)
        if self.step_num == self.step_len-1:
            self.done = True
        state = self._get_obs()
        mse = np.square(self.target_action[self.step_num]-action).mean()
        self.step_num=self.step_num+1
        return state, reward, self.done, mse

    def _get_obs(self):
        # finish the usage state truncate
        #print(self.machine_state_seq)
        usage_state = np.append(self.usage_record[:,self.machine_usage_bias],np.array(self.action_seq))
        # machine_state = self.machine_state_seq[:self.step_num+1]
        task_state = self.task_seq[self.step_num]
        usage_state = np.array(usage_state[-self.time_length:])
            # machine_state = np.array(machine_state[-self.time_length:])
        # process the task state to encoding
        # add
        #a concat the variable together as the observation
        # observation = np.append(usage_state/100, np.reshape(machine_state/100, (-1)))
        observation = np.append(usage_state, task_state)
        # return np.reshape(usage_state, (-1))
        return observation

    def reset(self, selected_index=None):
        # read in a random trajectory
        self.expert_traj = None
        if selected_index:
            index = selected_index
        else:
            index = np.random.randint(0, self.traj_num)
        data_path = self.pickle_path + str(index) + '_pkl'

        with open(data_path, 'rb') as f:
            self.expert_traj = pickle.load(f)
        # print(self.expert_traj.shape)
        self.step_len = self.expert_traj.shape[0]
        self.state_seq = []
        self.action_seq = []
        self.task_seq = []
        self.machine_state_seq = []
        self.target_action = []
        for pairs in self.expert_traj:
            self.state_seq.append(pairs[:self.readin_state_len])
            self.target_action.append(pairs[self.readin_state_len+self.machine_usage_bias])
        self.state_seq = np.array(self.state_seq)
        self.task_seq = self.state_seq[:, -self.task_state_dim:]
        # self.machine_state_seq = self.state_seq[:, -1, -self.machine_state_dim-2:-2]
        self.target_action = np.array(self.target_action)
        if self.cold_start:
            self.usage_record = np.zeros((self.time_length, 2))
        else:
            self.usage_record = self.expert_traj[0][:self.time_length *2]
            self.usage_record = np.reshape(self.usage_record, (self.time_length,2))
        self.done = False
        self.step_num = 0
        return self._get_obs()
