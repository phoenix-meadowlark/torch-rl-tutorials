# Vanilla Policy Gradient Optimization in PyTorch
## An implementation focused tutorial

I decided to make this tutorial because, while there are a plethora of detailed guides on the mathematics behind reinforcement learning, and a number of different implementations, I couldn't find any guides that really focused on the details of implementing them. So I created my own implementation of a policy gradient agent while trying to make my code as easy to understand as I possibly could.

Before reading the tutorial, or perhaps concurrently to, I would recommend reading over these three articles from [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html):
- [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

The most important section to understand is `CategoricalPolicyAgent.update_policy_weights`, which spans lines 154 to 212 in `vpg_torch_tutorial.py`. This is where the policy gradient is calculated, and I provide some direction for understanding how it is calculated in the way that it is.

I decided to split the code up into two files:

`vpg_torch_tutorial.py`: This code is commented a bit excessively for normal purposes, but is intended to be an easy introduction to policy gradient optimization in a ML framework.

`vpg_torch_tutorial_sparse.py`: This version of the code is closer to how I would comment it were it not a for a tutorial. It's more readable, and still explains anything which may appear hacky or arbitrary, but assumes more familarity with torch and RL.

## The class APIs

I decided to represent the algorithm by two classes: a `PolicyGradientAgent`, and a `CategoricalPolicyNetwork`. To use the algorithm, one only needs to interact with the `PolicyGradientAgent` class as the `CategoricalPolicyNetwork` is handled by the agent.

The `PolicyGradientAgent` is initialized with with the environment state and action sizes, along with optional hyperparameters. Learning is performed in a loop with the environment by asking the agent to act at each step via `agent.act(state)`, and storing the results of that action via `agent.remember(action, state, reward, done)`. After training, (or before if you like), you can have the agent act without learning by simply calling `agent.act` without calling `agent.remember`.

A simple training loop will look something like
```Python
for ep in episodes:
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(action, state, reward, done)
        state = next_state
```

While testing the agent at a fixed state would look something like
```Python
for ep in episodes:
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
```

## Example Training Loop with CartPole-v* and LunarLander-v2

Note that RL methods are pretty susceptible to initial conditions and divergence.

A Colab hosted notebook containing this implementation and code for training it on a few gym environments can be found [here](https://colab.research.google.com/drive/1-o9W05S8a3atS97clhEcu_lmhmLMNpVf)! Just open it and run in playground mode.

Imports:
```Python
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# if you're using Google Colab
# !pip install box2d-py
# !pip install gym[Box_2D]

plt.style.use('ggplot')

# Useful env for seeing more asymptotic behaviour
cap = 2000
threshold = 1000
gym.envs.register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': cap},
    reward_threshold=threshold,
)
```

Training Loop:
```Python
def train(env, agent, episodes, window):
    reward_buffer = []
    for episode in range(episodes):
        state = env.reset()
        done = False       
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(action, state, reward, done)
            state = next_state
        
        episode_reward = int(agent.reward_history[-1])
        
        if len(reward_buffer) > window:
            del reward_buffer[0]
        reward_buffer.append(episode_reward)
        
        moving_ave = np.average(reward_buffer)

        sys.stdout.write('\r' + 'Episode {:4d} Last Reward: {:5d} Moving Average: {:7.2f}'
                                .format(episode, episode_reward, moving_ave))
        sys.stdout.flush()

        if env.spec.reward_threshold is not None and moving_ave > env.spec.reward_threshold:
            print("\nSolved! Moving average is now {:.2f}.".format(moving_ave, episode_reward))
            break
```

Choose env and run training:
```Python
envs = ['CartPole-v0', 'CartPole-v1', 'CartPole-v2', 'LunarLander-v2']
env_ind = 1

env = gym.make(envs[env_ind])
env_obs_dim = env.observation_space.shape[0]
env_act_dim = env.action_space.n

max_episodes = 2000
moving_average_window = 50

print('Training PGA on {}'.format(envs[env_ind]))
agent = PolicyGradientAgent(env_obs_dim, env_act_dim)
train(env, agent, max_episodes, moving_average_window)
```

Plot Results
```Python
window = moving_average_window
fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[12,12])

rolling_mean = pd.Series(agent.reward_history).rolling(window).mean()
std          = pd.Series(agent.reward_history).rolling(window).std()

ax1.plot(rolling_mean)
ax1.fill_between(range(len(agent.reward_history)),rolling_mean-std, rolling_mean+std, alpha=0.2)

ax1.set_title('Episode Rewards Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Reward')

ax2.scatter(np.arange(len(agent.reward_history)), agent.reward_history, alpha=0.5)

ax2.set_title('Episode Rewards')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Episode Reward')

# Useful for some envs, which are learned exponentially with number of episodes
log_scale = False
if log_scale:
    ax2.set_yscale('log')

fig.tight_layout()
plt.show()
```

## Credits

This code was initially inspired by [Tim Sullivan](ts1829.github.io)'s medium tutorial, which can be found [here](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf).

Other resources which I've found helpful, or think may be helpful for the reader are sprinkled throughout. 

Author: Phoenix Meadowlark