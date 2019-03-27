# Vanilla Policy Gradient Optimization in PyTorch
## An implementation focused tutorial

I decided to make this tutorial because, while there are a plethora of detailed guides on the mathematics behind reinforcement learning, and a number of different implementations, I couldn't find any guides that really focused on the details of implementing them. So I created my own implementation of a policy gradient agent while trying to make my code as easy to understand as I possibly could. 

What I was most concerned about elucidating is the the process of calculating the policy gradient and using it to update a neural net's weights. This process is incredibly transparent PyTorch with its eager execution and the ease of building modular, classed code with it. 

As such, the most important section to understand is `CategoricalPolicyAgent.update_policy_weights`, which spans lines 154 to 212 in `vpg_torch_tutorial.py`. This is where the policy gradient is calculated, and I provide some direction for understanding how it is calculated in the way that it is.

Currently, the code file `vpg_torch_tutorial.py` is the tutorial itself. It is commented with that in mind, and contains some links to other resources which explain things more broadly. Before reading the code, or perhaps concurrently to reading it, I would recommend reading over these three articles from [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html):
- [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

I'll likely pull out the most relevant bits into a medium article, or make a youtube video walking through the code in the future, but I hope that some find if useful as it is now! I'll give a brief overview of the classes and how to use them in a training loop below.

A Colab hosted notebook containing this implementation and code for training it on a few gym environments can be found [here](https://colab.research.google.com/drive/1-o9W05S8a3atS97clhEcu_lmhmLMNpVf)! Just open it and run in playground mode. It's good to play with a few hyperparameters and run the algorithm a few times to understand how it behaves.

## The class APIs

I decided to represent the algorithm by two classes: a `PolicyGradientAgent`, and a `CategoricalPolicyNetwork`. To use the algorithm, one only needs to interact with the `PolicyGradientAgent` class as the `CategoricalPolicyNetwork` is handled by the agent. Understanding both is necessary for understanding how policy gradient optimization is implemented however.

The `PolicyGradientAgent` is initialized with with the environment state and action sizes, along with optional hyperparameters. Learning is performed in a loop with the environment by asking the agent to act at each step via `agent.act(state)`, and storing the results of that action via `agent.remember(action, state, reward, done)`. After training, you can have the agent act without learning by calling `agent.act` with the learn parameter explicitly set to `False` and without calling `agent.remember`.

A simple training loop will look something like
```Python
for ep in episodes:
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        agent.remember(reward, done)
```

While testing the agent at a fixed state would look something like
```Python
for ep in episodes:
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state, learn=False)
        state, reward, done, info = env.step(action)
```



## Example Training Loop with CartPole-v* and LunarLander-v2

Imports:
```Python
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# If using google colab
# !pip install box2d-py
# !pip install gym[Box_2D]

plt.style.use('ggplot')

# Useful env for seeing more asymptotic behaviour
cap = 2000
threshold = int(0.95 * cap)
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
    # Keeps track of the `window` most recent rewards
    reward_buffer = []
    for episode in range(episodes):
        # Resets environment, which also provides the initial state
        state = env.reset()
        
        done = False       
        while not done:
            # Have our agent choose and action
            action = agent.act(state)
            
            # Have the environment react to that action
            # and give us a reward and the next state
            # 
            # If the game is over, then it will set 
            # `done` to `True`
            state, reward, done, _ = env.step(action)
            
            # Have our agent remember the reward the action
            # gave, and implicitly update it's policy
            # if it has collected `batch_size` trajectories
            agent.remember(reward, done)
        
        episode_reward = int(agent.reward_history[-1])
        
        if len(reward_buffer) > window:
            del reward_buffer[0]
        reward_buffer.append(episode_reward)
        
        moving_ave = np.average(reward_buffer)

        sys.stdout.write('\r' + 'Episode {:4d} Last Reward: {:5d} Moving Average: {:7.2f}'
                                .format(episode, episode_reward, moving_ave))
        sys.stdout.flush()

        if env.spec.reward_threshold is not None and moving_ave > env.spec.reward_threshold:
            print("\nThe environment was solved, with a moving average reward of {:7.2f}!."
                  .format(moving_ave, episode_reward))
            break
```

Choose env and run training:
```Python
# A few envs to choose from
envs = ['CartPole-v0', 'CartPole-v1', 'CartPole-v2', 'LunarLander-v2']
env_ind = 2

print('Training PGA on {}'.format(envs[env_ind]))

# Create Env and get the dimensions of the action and observation spaces
env = gym.make(envs[env_ind])
env_obs_dim = env.observation_space.shape[0]
env_act_dim = env.action_space.n

# Initialize our agent to the default hyperparameters
agent = PolicyGradientAgent(
    obs_dim = env_obs_dim,
    act_dim = env_act_dim,
    hid_sizes  = [128],
    batch_size = 1,
    gamma      = 0.99, 
    lr         = 1e-2,
    dropout    = 0.5,
    l2_weight  = 0,
)

# Controls the limit for how long our agent has to solve the environment
# and how confident we are that it performs that well
max_episodes = 2000
moving_average_window = 50

# Trains our agent with all of the above settings
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

This code was initially inspired by [Tim Sullivan](https://ts1829.github.io)'s medium tutorial, which can be found [here](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf).

Other resources which I've found helpful, or think may be helpful for the reader are sprinkled throughout. 

Author: Phoenix Meadowlark