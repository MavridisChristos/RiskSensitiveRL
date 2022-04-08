# Test the REINFORCE algorithm with OpenAI Gym's cartpole environment.
# For details see:
# https://www.datahubbs.com/policy-gradients-with-reinforce/

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from time import time

from networks import policy_estimator, value_estimator
from reinforce import *
from reinforce_with_baseline import *
from reinforce_risk import *


# #tf.reset_default_graph()
# #sess = tf.Session()
#
# #pe = policy_estimator(sess, env)
# #ve = value_estimator(sess, env)
#
# #Initialize variables
# #init = tf.global_variables_initializer()
# #sess.run(init)
#
# #print("\nBegin Training...")
#
# #start_time = time()
# #pe_rewards = reinforce(env, pe)
# #end_time = time()
# #print("Training Time(REINFORCE): {:.2f}".format(end_time - start_time))
#
# #start_time = time()
# #pe_baseline_rewards = reinforce_baseline(env, pe, ve)
# #end_time = time()
# #print("Training Time(REINFORCE with baseline): {:.2f}".format(end_time - start_time))
#
# #start_time = time()
# #pe_rewards_exp = risk_reinforce(env, pe, ve)
# #end_time = time()
# #print("Training Time(Risk-sensitive REINFORCE): {:.2f}".format(end_time - start_time))
#
#
# # Smooth rewards
# #smoothed_pe_rewards = [np.mean(pe_rewards[max(0,i-10):i+1]) for i in
# # range(len(pe_rewards))]
#
# #smoothed_pe_baseline_rewards = [np.mean(pe_baseline_rewards[max(0,i-10):i+1]) for i in
# # range(len(pe_baseline_rewards))]
#
# #smoothed_pe_rewards_exp = [np.mean(pe_rewards_exp[max(0,i-10):i+1]) for i in
# # range(len(pe_baseline_rewards))]
#
# # Plot results
# #plt.figure()
# #plt.plot(smoothed_pe_rewards, label='REINFORCE')
# #plt.plot(smoothed_pe_baseline_rewards, label='REINFORCE with Baseline')
# #plt.plot(smoothed_pe_baseline_rewards, label='Risk-sensitive REINFORCE')
# #plt.legend(loc='best')
# #plt.title("Comparison of REINFORCE Algorithms for Cart-Pole")
# #plt.xlabel("Episode")
# #plt.ylabel("Rewards")
# #plt.show()
#
#
# #env = gym.make('CartPole-v0')
# env = gym.make('Acrobot-v1')
# env = gym.make('Pendulum-v0')
#env = gym.envs.make("MountainCar-v0")
# env._max_episode_steps = 1000

# N = 50  # Number of training runs
# num_episodes = 5000
#
# pe_rewards_exp = np.zeros(num_episodes)
# pe_rewards = np.zeros(num_episodes)
# pe_baseline_rewards = np.zeros(num_episodes)
#
# for n in range(N):
#  beta = 0.01
#  tf.reset_default_graph()
#  sess = tf.Session()
#
#  pe = policy_estimator(sess, env)
#
#  # Initialize variables
#  init = tf.global_variables_initializer()
#  sess.run(init)
#
#  # Train model
#  rewards_exp = risk_reinforce(env, pe, num_episodes, beta)
#  pe_rewards_exp += rewards_exp
#
# for n in range(N):
#  tf.reset_default_graph()
#  sess = tf.Session()
#
#  pe = policy_estimator(sess, env)
#
#  # Initialize variables
#  init = tf.global_variables_initializer()
#  sess.run(init)
#
#  # Train model
#  rewards = reinforce(env, pe, num_episodes)
#  pe_rewards += rewards
#
# for n in range(N):
#  tf.reset_default_graph()
#  sess = tf.Session()
#
#  pe = policy_estimator(sess, env)
#  ve = value_estimator(sess, env)
#
#  # Initialize variables
#  init = tf.global_variables_initializer()
#  sess.run(init)
#
#  # Train model
#  baseline_rewards = reinforce_baseline(env, pe, ve, num_episodes)
#  pe_baseline_rewards += baseline_rewards
#
# pe_rewards_exp /= N
# pe_rewards /= N
# pe_baseline_rewards /= N
#
# plt.figure(figsize=(12, 8))
# plt.plot(pe_rewards, label='Vanila REINFORCE')
# plt.plot(pe_baseline_rewards, label='REINFORCE with Baseline')
# plt.plot(pe_rewards_exp, label='Risk-sensitive REINFORCE with exponential criteria')
# plt.legend(loc='best')
# plt.title('Comparison of REINFORCE Algorithms for Acrobot')
# plt.show()
#plt.savefig('MountainCar.png', bbox_inches='tight')
#env = gym.envs.make("MountainCar-v0")
#env._max_episode_steps = 4000

# #################################
# beta graphs
# plt.figure(figsize=(12, 8))
# for i, beta in enumerate([-0.001, -0.01, -0.1, -1, -10]):
#  env = gym.make('CartPole-v0')
#  tf.reset_default_graph()
#  sess = tf.Session()
#
#  N = 50  # Number of training runs
#  num_episodes = 2000
#
#  pe_rewards_exp = np.zeros(num_episodes)
#
#  for n in range(N):
#   tf.reset_default_graph()
#   sess = tf.Session()
#
#   pe = policy_estimator(sess, env)
#
#   # Initialize variables
#   init = tf.global_variables_initializer()
#   sess.run(init)
#
#   # Train model
#   rewards_exp = risk_reinforce(env, pe, num_episodes, beta)
#   pe_rewards_exp += rewards_exp
#
#  pe_rewards_exp /= N
#  plt.plot(pe_rewards_exp, label='beta = %s' % (beta))
#
# plt.legend(loc='best')
# plt.title('Comparison of different risk-sensitive parameter for Cart-pole')
# plt.show()

# beta for negative reward structure
plt.figure(figsize=(12, 8))
for i, beta in enumerate([0.001, 0.01, 0.1, 1, 10]):
 env = gym.make('Acrobot-v1')
 tf.reset_default_graph()
 sess = tf.Session()

 N = 50  # Number of training runs
 num_episodes = 2000

 pe_rewards_exp = np.zeros(num_episodes)

 for n in range(N):
  tf.reset_default_graph()
  sess = tf.Session()

  pe = policy_estimator(sess, env)

  # Initialize variables
  init = tf.global_variables_initializer()
  sess.run(init)

  # Train model
  rewards_exp = risk_reinforce(env, pe, num_episodes, beta)
  pe_rewards_exp += rewards_exp

 pe_rewards_exp /= N
 plt.plot(pe_rewards_exp, label='beta = %s' % (beta))

plt.legend(loc='best')
plt.title('Comparison of different risk-sensitive parameter for Acrobot')
plt.show()