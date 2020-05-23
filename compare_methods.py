import gym
import numpy as np
import sys
import matplotlib.pyplot as plt
import random

np.set_printoptions(threshold=sys.maxsize)

env = gym.make('Taxi-v3')
# print(env.action_space)
# print(env.observation_space)

discount_factor = 0.99
learning_rate = 1
num_training_episodes = 5000
num_training_steps = 30


Q_values = np.zeros((500, 6))
rewards_random = np.zeros(num_training_episodes)
# Random policy evaluation
for episode in range(num_training_episodes):
    observation = env.reset()
    for step in range(num_training_steps):
        initial_observation = observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        Q_values[initial_observation][action] = Q_values[initial_observation][action] + learning_rate * (reward + discount_factor * np.max(Q_values[observation]) - Q_values[initial_observation][action])
        rewards_random[episode] += reward
        if done:
            break


Q_values = np.zeros((500, 6))
rewards = np.zeros(num_training_episodes)
# # Simple Q-learning algorithm
for episode in range(num_training_episodes):
    observation = env.reset()
    for step in range(num_training_steps):
        initial_observation = observation
        action = np.argmax(Q_values[initial_observation])
        observation, reward, done, info = env.step(action)
        Q_values[initial_observation][action] = Q_values[initial_observation][action] + learning_rate * (reward + discount_factor * np.max(Q_values[observation]) - Q_values[initial_observation][action])
        rewards[episode] += reward
        if done:
            break

Q_values = np.zeros((500, 6))
rewards_exploration = np.zeros(num_training_episodes)
exploration_rate = 0.05
for episode in range(num_training_episodes):
    observation = env.reset()
    for step in range(num_training_steps):
        initial_observation = observation
        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_values[initial_observation])
        observation, reward, done, info = env.step(action)
        Q_values[initial_observation][action] = Q_values[initial_observation][action] + learning_rate * (reward + discount_factor * np.max(Q_values[observation]) - Q_values[initial_observation][action])
        rewards_exploration[episode] += reward
        if done:
            break

Q_values_A = np.zeros((500, 6))
Q_values_B = np.zeros((500, 6))
rewards_double_Q_learning = np.zeros(num_training_episodes)
for episode in range(num_training_episodes):
    observation = env.reset()
    for step in range(num_training_steps):
        initial_observation = observation
        action = np.argmax(Q_values_A[initial_observation] + Q_values_B[initial_observation])
        observation, reward, done, info = env.step(action)
        if random.random() < 0.5:
            action_star = np.argmax(Q_values_A[observation])
            Q_values_A[initial_observation][action] = Q_values_A[initial_observation][action] + learning_rate * (reward + discount_factor * Q_values_B[observation, action_star] - Q_values_A[initial_observation][action])
        else:
            action_star = np.argmax(Q_values_B[observation])
            Q_values_B[initial_observation][action] = Q_values_B[initial_observation][action] + learning_rate * (reward + discount_factor * Q_values_A[observation, action_star] - Q_values_B[initial_observation][action])
        rewards_double_Q_learning[episode] += reward
        if done:
            break


fig = plt.figure(figsize=(15, 7))
plt.plot(np.arange(0, num_training_episodes)[0::50], rewards_random[0::50], label='Random Policy')
plt.plot(np.arange(0, num_training_episodes)[0::50], rewards[0::50], label='Q-learning')
plt.plot(np.arange(0, num_training_episodes)[0::50], rewards_exploration[0::50], label='Q-learning with Exploration')
plt.plot(np.arange(0, num_training_episodes)[0::50], rewards_double_Q_learning[0::50], label='Double Q-learning')
plt.legend(loc='best')
plt.xlabel('Episode of Tranining')
plt.ylabel('Reward per Episode')
plt.show()
