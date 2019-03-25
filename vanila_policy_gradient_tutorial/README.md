# Vanilla Policy Gradient Optimization in PyTorch
### An implementation focused tutorial

I decided to split the code up into two files:

`vpg_torch_tutorial.py`: This code is commented a bit excessively for normal purposes, but is intended to be an easy introduction to policy gradient optimization in a ML framework.

`vpg_torch_tutorial_sparse.py`: This version of the code is closer to how I would comment it were it not a for a tutorial. It's more readable, and still explains anything which may appear hacky or arbitrary, but assumes more familarity with torch and RL.

This code was initially inspired by [Tim Sullivan](ts1829.github.io)'s medium tutorial, which can be found [here](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf).

Other resources which I've found helpful, or think may be helpful for the reader are sprinkled throughout. 

The most notable resource however, which deserves mentioning up front, is [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html), which contains a through introduction to reinforcement learning concepts and its mathematical objcts. They also provide performant tensorflow implementations of several reinforcement learning algorithms, however as the code is written in `tensorflow version 1`, it is not the most interpretable for beginners. Which is why I created this tutorial.

```
Author:  Phoenix Meadowlark
Website: phoenix-meadowlark.github.io
```
