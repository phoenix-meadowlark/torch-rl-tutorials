from typing import List

import numpy as np
import numpy # For typing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torch.distributions import Categorical

# This version of the code is closer to how I would comment it were it not a for a tutorial.
# It's more readable, and still explains anything which may appear hacky or arbitrary, but assumes more 
# familarity with torch and RL.

# This code was initially inspired by Tim Sullivan's (ts1829.github.io) medium tutorial, which
# can be found here (https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf)

# Other resources which I've found helpful, or think may be helpful for the reader are sprinkled throughout. 

# The most notable resource however, which deserves mentioning up front, is 
# (https://spinningup.openai.com/en/latest/index.html), which contains a through introduction to reinforcement
# learning concepts and its mathematical objcts. They also provide performant tensorflow implementations of several
# reinforcement learning algorithms, however as the code is written in tensorflow 1.*, it is not the most interpretable
# for beginners. Which is why I created this tutorial.

# Author:  Phoenix Meadowlark
# Website: phoenix-meadowlark.github.io

class CategoricalPolicyNetwork(nn.Module):
    """
    A torch neural network which represents a policy gradient policy pi(action | state).
    The distribution of action probabilities pi(* | state) can be calculated via PolicyNetwork(observation).
    The network can be updated with the update_policy_weights function, which updates
    the network's weight based on the values of one or more trajectories in the environment.

    Parameters
    ----------
    obs_dim : int, required
        The size of the observation space of the environment. This network only handles vector observations.
    act_dim : int, required
        The size of the action space of the environment. This network is designed for use with categorical
        action spaces.
    hid_sizes : List[int], required
        The sizes of the hidden layers of the network. If an empty list is passed, logistic regression will
        be performed between the observation and action spaces.
    gamma : float, required
        The discount rate for our infinite horizon sum of rewards.
    lr    : float, required
        The learning rate of our network.
    dropout : float, required
        The dropout rate between the network's hidden layers.
    """
    def __init__(self, obs_dim:   int,
                       act_dim:   int,
                       hid_sizes: List[int],
                       gamma:     float,
                       lr:        float,
                       dropout:   float) -> None:
        super(CategoricalPolicyNetwork, self).__init__()
        # Env Info
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Hyperparameters
        self.hid_sizes = hid_sizes
        self.gamma    = gamma
        self.lr       = lr
        self.dropout  = dropout
        
        # Util
        self.eps = 1e-7 # some small value to prevent div by 0

        # Network Parameters
        self.layers = self._build_policy_network()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def _build_policy_network(self) -> torch.nn.ModuleList:
        """
        Internal method which builds a policy network with the specified hidden sizes and hyperparameters.

        Returns
        _______
        layers : torch.nn.ModuleList
            A list of all of the layers in our network.
        """
        layers = nn.ModuleList()

        if len(self.hid_sizes) == 0:
            # Create a logistic computation graph
            layers.append(nn.Sequential(
                nn.Linear(self.obs_dim, self.act_dim, bias=False),
                nn.Softmax(dim=-1)
            ))
        else:
            # Create a multilayer perceptron
            layers.append(nn.Sequential(
                nn.Linear(self.obs_dim, self.hid_sizes[0], bias=False),
                nn.Dropout(p=self.dropout),
                nn.ReLU()
            ))

            for i in range(1, len(self.hid_sizes) - 1):
                layers.append(nn.Sequential(
                    nn.Linear(self.hid_sizes[i-1], self.hid_sizes[i], bias=False),
                    nn.Dropout(p=self.dropout),
                    nn.ReLU()
                ))

            layers.append(nn.Sequential(
                nn.Linear(self.hid_sizes[-1], self.act_dim, bias=False),
                nn.Softmax(dim=-1)
            ))

        return layers

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass through the policy network, taking an observation from the environment
        and transforming it into some probability distribution in the shape of the action space.

        Parameters
        __________
        state : torch.Tensor, required
            The observation of the environment that our env gives to us turned into a float tensor
            (e.g. via torch.from_numpy(obs).to(torch.float32))

        Returns
        _______
        action_probs : torch.Tensor
            The learned probability distribution over actions for the inputted state.
        """
        action_probs = state
        for layer in self.layers:
            action_probs = layer(action_probs)

            if True in torch.isnan(action_probs):
                raise ValueError('Divergent weight update detected. Turn down learning rate or adjust other hyperparams.')
                    
        return action_probs

    def update_policy_weights(self, batch_action_log_probs: torch.Tensor,
                                    batch_trajectory_rewards:          List[List[float]]) -> float:
        """
        Approximates the gradient of our expected return for our policy network per the reward to go version of
        (https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html).

        Parameters
        __________
        batch_action_log_probs : torch.Tensor, required
            A (len(trajectory_0) + len(trajectory_1) + ... + len(trajectory_(batch_size-1)), ) torch tensor 
            containing all of action probability distributions that our policy network assigned to the states we saw. 
            This is a 1D tensor of the form torch.cat([trajectory_0_action_probs, trajectory_1_action_probs,...]).
        batch_trajectory_rewards : List[List[float]], required
            A (batch_size, len(trajectory_n)) list of lists. The first index iterates over the trajectories, and the
            second index iterates over rewards at each step, which are variable depending on how long our agent
            survived.
        
        Returns
        _______
        loss : float
            The 'loss' of this policy gradient update. It's not really a loss in the supervised learning sense, since
            the data generating distribution (our policy) is always changing.
        """
        rewards_to_go = self._discount_rewards_and_normalize(batch_trajectory_rewards, batch_action_log_probs.shape[0])

        # Compute log(pi(action | state)) * reward_to_go(action, state, next_state) for each step in the environment
        pi_r = torch.mul(batch_action_log_probs, rewards_to_go)
        loss = -torch.sum(pi_r, dim=-1) # Create a loss who's minimization maximizes our expected return.
        loss /= len(batch_trajectory_rewards) # Normalize by the number of trajectories
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.item()
    
    def _discount_rewards_and_normalize(self, batch_trajectory_rewards: List[List[float]], 
                                              num_rewards:        int) -> torch.Tensor:
        """
        Internal method for computing and normalizing the discounted sum of rewards to go. 

        Parameters
        __________
        batch_trajectory_rewards : List[List[float]], required
            A (batch_size, len(trajectory_n)) list of lists. The first index iterates over the trajectories, and the
            second index iterates over rewards at each step, which are variable depending on how long our agent
            survived.
        num_rewards : int, required
            The number of steps taken (and thus rewards) in the environment over all trajectories in this batch.
        Returns
        _______
        rewards : torch.Tensor
            The discounted and normalized reward to go for each trajectory concatinated together.
        """
        rewards = torch.Tensor(num_rewards)
        ind = num_rewards - 1

        for trajectory in batch_trajectory_rewards[::-1]:
            rewards[ind] = trajectory[-1]
            ind -= 1
            for step_reward in trajectory[-2::-1]:
                rewards[ind] = step_reward + self.gamma * rewards[ind + 1]
                ind -= 1

        # Normalize rewards and add some smallest float value in case std = 0.
        return (rewards - rewards.mean()) / (rewards.std() + self.eps)

class PolicyGradientAgent():
    """
    A class representing an agent which acts in environment, and learns from the environments reward signal
    via gradient descent on the gradient of its expected return. An easy to follow mathematical derivation
    can be found here (https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) with a useful overview 
    of terms here (https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).

    To make the agent learn in an environment, progressively step through the environment by having the agent choose an
    action via `agent.act(state)`, and the have the agent remember that action, state and reward via 
    `agent.remember(action, state, reward, done)`. It is assumed that the calls to remember are done in sequential 
    steps as an agent progresses through an environment until the end of the episode (where done = True).

    An example barebones training loop using a gym env would be:

    for ep in episodes:
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(action, state, reward, done)
            state = next_state

    To make the agent act without learning, simply call `agent.act(state)` without later calling `agent.remember()`.

    Parameters
    ----------
    obs_dim : int, required
        The size of the observation space of the environment. This network only handles vector observations.
    act_dim : int, required
        The size of the action space of the environment. This network is designed for use with categorical
        action spaces.
    hid_sizes : List[int], optional
        The sizes of the hidden layers of the network. If an empty list is passed, logistic regression will
        be performed between the observation and action spaces.
    batch_size : int, optional
        The number of trajectories to collect before estimating the policy gradient
    gamma : float, optional
        The discount rate for our infinite horizon sum of rewards.
    lr : float, optional
        The learning rate of our network.
    dropout : float, optional
        The dropout rate between the network's hidden layers.
    """
    def __init__(self, obs_dim:    int,
                       act_dim:    int,
                       hid_sizes:  List[int] = [128],
                       batch_size: int   = 1,
                       gamma:      float = 0.99,
                       lr:         float = 5e-3,
                       dropout:    float = 0.6) -> None:
        # Env Info
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Hyperparameters
        self.hid_sizes   = hid_sizes
        self.batch_size  = batch_size
        self.gamma   = gamma
        self.lr      = lr
        self.dropout = dropout
        
        # Network
        self.policy_network = CategoricalPolicyNetwork(obs_dim, act_dim, hid_sizes, gamma, lr, dropout)
        
        # Current Training Information
        self.new_trajectory = True
        self.trajectory_count    = 0
        self.batch_action_log_probs    = None
        self.batch_trajectory_rewards  = []
        
        # Agent Learning History
        self.reward_history = []
        self.loss_history   = []

    def act(self, state: numpy.array) -> int:
        """
        Observes the inputted state and stochastically samples an action from our policy function pi(* | state).

        Parameters
        __________
        state: numpy.array, required
            The state that the environment is in during this step.

        Returns
        _______
        action: int
            The action that our agent chose to take in this situation.
        """
        state = torch.from_numpy(state).to(torch.float32)

        # Compute the action probs without having torch's autograd save gradient information.
        with torch.no_grad():
            action_probs = self.policy_network(state)
        
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

        return action.data.item()
    
    def remember(self, action: int,
                       state:  numpy.array,
                       reward: float,
                       done:   bool) -> None:
        """
        Stores the information from this step in the environment, and periodically updates the agent's policy
        network based every batch_size episodes.

        Assumes that the action-state-reward tuples provided are sequential steps in some environment between
        each time `agent.remember()` is called with `done = True`.

        Parameters
        __________
        action : int, required
            The action that the agent took at this step
        state : numpy.array, required
            The state that the environment was in before the agent acted
        reward : float, required
            The reward that the environment provided for the action
        done : bool, required
            A flag saying whether or not this was the last action in a trajectory
            This is automatically provided by gym env's `env.step()` function.
        """
        state = torch.from_numpy(state).to(torch.float32)

        # Recompute the action probabilities, this time storing grad information
        action_probs = self.policy_network(state)
        action_log_prob = torch.log(action_probs[action]).reshape(1)
        # reshape moves from a scalar to a 1D tensor so we can concat it with the pervious action_log_probs

        if self.batch_action_log_probs is None:
            self.batch_action_log_probs = action_log_prob
        else:
            self.batch_action_log_probs = torch.cat([self.batch_action_log_probs, action_log_prob]) 
    
        if self.new_trajectory:
            # append a new array to the batch_trajectory_rewards to keep track of the next trajectory's rewards
            self.batch_trajectory_rewards.append([])
            self.new_trajectory = False
            
        # Append step reward to current trajectory's rewards
        self.batch_trajectory_rewards[-1].append(reward)
        
        if done:
            self.trajectory_count += 1
            self.new_trajectory = True
            self.reward_history.append(np.sum(self.batch_trajectory_rewards[-1]))

            # Perform a gradient update on our policy network if we've collected batch_size trajectories
            if self.trajectory_count == self.batch_size:
                self._update_policy()
    
    def _update_policy(self) -> None:
        """
        Internal method for telling the policy network to update its weights, and resetting some internal variables.
        """
        loss = self.policy_network.update_policy_weights(self.batch_action_log_probs, self.batch_trajectory_rewards)

        self.loss_history.append(loss)

        # Clear variables for the next batch
        self.batch_action_log_probs = None
        self.batch_trajectory_rewards = []
        self.trajectory_count = 0