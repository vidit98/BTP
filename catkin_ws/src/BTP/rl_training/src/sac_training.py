#!/usr/bin/env python
import math
import rospy
import time
import random
import gym
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
#import mathplotlib.pyplot as plt

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)

		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done


class NormalizedActions(gym.ActionWrapper):
	def action(self, action):
		low = self.action_space.low
		high = self.action_space.high
		action = low + (action + 1.0) * 0.5 * (high - low)
		action = np.clip(action, low, high)
		return action

	def normalized_action(action, low, high): 
		action = low + (action + 1.0) * 0.5 * (high - low)
		action = np.clip(action, low, high)
		return action



class Soft_Q_Network(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size=[256,256], init_w=3e-3):
		super(Soft_Q_Network, self).__init__()
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
		self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
		self.linear3 = nn.Linear(hidden_size[1], 1)
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x


class PolicyNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size=[256,256], init_w=3e-3, log_std_min=-20, log_std_max=2, epsilon=1e-6):
		super(PolicyNetwork, self).__init__()

		self.epsilon = epsilon
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		self.linear1 = nn.Linear(num_inputs, hidden_size[0])
		self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])

		self.mean_linear = nn.Linear(hidden_size[1], num_actions)
		self.mean_linear.weight.data.uniform_(-init_w, init_w)
		self.mean_linear.bias.data.uniform_(-init_w, init_w)

		self.log_std_linear = nn.Linear(hidden_size[1], num_actions)
		self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		self.log_std_linear.bias.data.uniform_(-init_w, init_w)

	def forward(self, state, deterministic=False):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))

		mean = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

		std = torch.exp(log_std)
		log_prob = None

		if deterministic:
			action = torch.tanh(mean)
		else:
			normal = Normal(0, 1)
			z = mean + std * normal.sample().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
			action = torch.tanh(z)
			log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon)
			log_prob = log_prob.sum(dim=1, keepdim=True)

		return action, mean, log_std, log_prob, std

	def get_action(self, state, deterministic=False):
		action,_,_,_,_ = self.forward(state, deterministic)
		act = action.cpu()[0]
		return act

class SAC(object):
	def __init__(self, env, replay_buffer, hidden_dim=[256, 256], steps_per_epoch=200, epochs=1000, discount=0.99, tau=1e-2, policy_lr=1e-3, qf_lr=1e-3, auto_alpha_tuning=True, batch_size=100):

		self.env = env
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.shape[0]
		self.hidden_dim = hidden_dim

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.auto_alpha = auto_alpha_tuning

		self.soft_q_net1 = Soft_Q_Network(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
		self.soft_q_net2 = Soft_Q_Network(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

		self.target_soft_q_net1 = Soft_Q_Network(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
		self.target_soft_q_net2 = Soft_Q_Network(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

		for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
			target_param.data.copy_(param.data)

		for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
			target_param.data.copy_(param.data)

		self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

		self.soft_qf_criterion = nn.MSELoss()
		self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr = qf_lr)
		self.soft_q_optimizer2 = optim.Adam(self.soft_q_net1.parameters(), lr = qf_lr)
		self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

		if self.auto_alpha:
			self.target_entropy = -np.prod(env.action_space.shape).item()
			self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)

		self.replay_buffer = replay_buffer
		self.discount = discount
		self.batch_size = batch_size
		self.tau = tau

	def get_action(self, state, deterministic=False, explore=False):
		state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		if explore:
			return self.env.action_space.sample()
		else:
			action = self.policy_net.get_action(state, deterministic).detach()
			return action.numpy()

	def update(self, iterations):
		for _ in range(iterations):

			state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
			#print reward

			state = torch.FloatTensor(state).to(self.device)
			next_state = torch.FloatTensor(next_state).to(self.device)
			action = torch.FloatTensor(action).to(self.device)
			reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
			done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

			new_actions, policy_mean, policy_log_std, log_pi, _ = self.policy_net(state)

			if self.auto_alpha:
				alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
				self.alpha_optimizer.zero_grad()
				alpha_loss.backward()
				self.alpha_optimizer.step()
				alpha = self.log_alpha.exp()
			else:
				alpha_loss = 0
				alpha = 0.2

			q_new_actions = torch.min(self.soft_q_net1(state, new_actions), self.soft_q_net2(state, new_actions))
			policy_loss = (alpha * log_pi - q_new_actions).mean()

			q1_pred = self.soft_q_net1(state, action)
			q2_pred = self.soft_q_net2(state, action)

			new_next_actions, _, _, new_log_pi, _ = self.policy_net(next_state)

			#print q1_pred.shape

			target_q_values = torch.min(self.target_soft_q_net1(next_state, new_next_actions), self.target_soft_q_net2(next_state, new_next_actions),) - alpha * new_log_pi

			#print target_q_values.shape
			
			q_target = reward + (1 - done) * self.discount * target_q_values
			q1_loss = self.soft_qf_criterion(q1_pred, q_target.detach())
			q2_loss = self.soft_qf_criterion(q2_pred, q_target.detach())

			self.soft_q_optimizer1.zero_grad()
			q1_loss.backward()
			self.soft_q_optimizer1.step()

			self.soft_q_optimizer2.zero_grad()
			q2_loss.backward()
			self.soft_q_optimizer2.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			self.policy_optimizer.step()

			for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

			for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

def train(agent, steps_per_epoch=1000, epochs=1000, start_steps=1000, max_ep_len=500):

	start_time = time.time()
	total_rewards = []
	avg_reward = None

	o,r,d,ep_reward,ep_len,ep_num = env.reset(), 0, False, 0, 0, 1

	total_steps = steps_per_epoch * epochs

	for t in range(1, total_steps):

		explore = t<start_steps
		a = agent.get_action(o, explore=False)

		o2,r,d, _ = env.step(a)
		ep_reward += r
		ep_len += 1

		d = False if ep_len == max_ep_len else d

		replay_buffer.push(o,a,r,o2,d)

		o=o2

		if d or (ep_len == max_ep_len):

			if not explore:
				agent.update(ep_len)

			total_rewards.append(ep_reward)
			avg_reward = np.mean(total_rewards[-100:])

			print("Steps:{} Episode:{} Reward:{} Avg Reward:{}".format(t,ep_num,ep_reward,avg_reward))

			o,r,d,ep_reward,ep_len = env.reset(), 0, False, 0, 0
			ep_num += 1

		if t>0 and t%steps_per_epoch == 0:
			epoch = t

rospy.init_node('test_RL_PANDA', anonymous=True, log_level=rospy.WARN)
replay_buffer = ReplayBuffer(int(1e6))
env = NormalizedActions(StartOpenAI_ROS_Environment('MyPandaTrainingEnv-v0'))
agent = SAC(env, replay_buffer, hidden_dim=[256, 256])
train(agent)