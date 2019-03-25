# Vanilla Policy Gradient Optimization in PyTorch
## An implementation focused tutorial

I decided to split the code up into two files:

`vpg_torch_tutorial.py`: This code is commented a bit excessively for normal purposes, but is intended to be an easy introduction to policy gradient optimization in a ML framework.

`vpg_torch_tutorial_sparse.py`: This version of the code is closer to how I would comment it were it not a for a tutorial. It's more readable, and still explains anything which may appear hacky or arbitrary, but assumes more familarity with torch and RL.

This code was initially inspired by [Tim Sullivan](ts1829.github.io)'s medium tutorial, which can be found [here](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf).

Other resources which I've found helpful, or think may be helpful for the reader are sprinkled throughout. 

The most notable resource however, which deserves mentioning up front, is [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html), which contains a through introduction to reinforcement learning concepts and its mathematical objcts. They also provide performant tensorflow implementations of several reinforcement learning algorithms, however as the code is written in `tensorflow version 1`, it is not the most interpretable for beginners. Which is why I created this tutorial.

Author: [Phoenix Meadowlark](https://phoenix-meadowlark.github.io)

## The class APIs

I decided to represent the algorithm by two classes: a `PolicyGradientAgent`, and a `CategoricalPolicyNetwork`. To use the algorithm, one only needs to interact with the `PolicyGradientAgent` class as the `CategoricalPolicyNetwork` is handled by the agent.

The `PolicyGradientAgent` is initalized with with the enviroment state and action sizes, along with optional hyperparameters. Learning is performed in a loop with the enviroment by asking the agent to act at each step via `agent.act(state)`, and storing the results of that action via `agent.remember(action, state, reward, done)`. After training, (or before if you like), you can have the agent act without learning by simply calling `agent.act` without calling `agent.remember`.

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
