import gym
import numpy as np

# Load the environment
# env = gym.make('FrozenLake-v0')
env = gym.make('CartPole-v0')
# Implement Q-Table learning algorithm
# Initialize table with all zeros
# Q = np.zeros([env.observation_space.n, env.action_space.n])
# # Set learning parameters
# lr = .85
# y = .99
# num_episodes = 2000
# # create lists to contain total rewards and steps per episode
# # jList = []
# rList = []
# for i in range(num_episodes):
#     # Reset environment and get first new observation
#     s = env.reset()
#     rAll = 0
#     d = False
#     j = 0
#     # The Q-Table learning algorithm
#     while j < 99:
#         j += 1
#         # Choose an action by greedily (with noise) picking from Q table
#         a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
#         # Get new state and reward from environment
#         s1, r, d, _ = env.step(a)
#         # Update Q-Table with new knowledge
#         Q[s, a] += lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
#         rAll += r
#         s = s1
#         if d:
#             break
#     # jList.append(j)
#     rList.append(rAll)
# print("Score over time: " + str(sum(rList) / num_episodes))
# print("Final Q-Table Values")
# print(Q)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

