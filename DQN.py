# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).


# Complete the four methods, 'create_greedy_policy', 'epsilon_greedy', 'compute_target_values', and 'train_episode'
# A neural network has been initialized for you in the '__init__' function
# and a function, 'update_target_model', for copying it to the target network has been provided.
# You must create a memory of a size given by the parameter '-m' which is also known as a replay buffer.
# At each time step you should take an action given by the epsilon greedy policy (same as Q-learning),
# record the transition in memory, and update the online/active neural network.
# Apply a single update per time step with a minibatch of size '-b', selected uniformly from the memory.
# Refresh the target neural network at an interval equal to the parameter '-N'.

import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from debugpy.common.timestamp import current
from torch.optim import AdamW
from Abstract_Solver import AbstractSolver, Statistics
#from lib import plotting


class QFunction(nn.Module):
    """
    Q-network definition.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
    ):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class DQN(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # Create Q-network
        self.model = QFunction(
            env.observation_space.shape[0],
            env.action_space.n,
            self.options.layers,
        )
        # Create target Q-network
        self.target_model = deepcopy(self.model)
        # Set up the optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.options.alpha, amsgrad=True
        )
        # Define the loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

        # Number of training steps so far
        self.n_steps = 0

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def epsilon_greedy(self, state):
        """
        Apply an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            The probabilities (as a Numpy array) associated with each action for 'state'.

        Use:
            self.env.action_space.n: Number of available actions
            self.torch.as_tensor(state): Convert Numpy array ('state') to a tensor
            self.model(state): Returns the predicted Q values at a 
                'state' as a tensor. One value per action.
            torch.argmax(values): Returns the index corresponding to the highest value in
                'values' (a tensor)
        """
        # Don't forget to convert the states to torch tensors to pass them through the network.
        ################################

        n_actions = self.env.action_space.n
        epsilon = self.options.epsilon
        prob_vector = np.ones(n_actions) * (epsilon / n_actions)
        state_tensor = torch.as_tensor(state)
        q_values = self.model(state_tensor)
        best_action = torch.argmax(q_values)
        prob_vector[best_action] += (1 - epsilon)
        prob_vector = prob_vector/sum(prob_vector)
        return prob_vector

        ################################


    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.

        Returns:
            The target q value (as a tensor) of shape [len(next_states)]
        """
        ################################
        next_q_values = self.target_model(next_states)
        max_q_values = torch.max(next_q_values, dim=1)[0]
        # Compute the target Q-values
        dones = dones.float()
        target_q = rewards + self.options.gamma * max_q_values * (1 - dones)
        return target_q
        ################################


    def replay(self):
        """
        TD learning for q values on past transitions.

        Use:
            self.target_model(state): predicted q values as an array with entry
                per action
        """
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Current Q-values
            current_q = self.model(states)
            # Q-values for actions in the replay memory
            current_q = torch.gather(
                current_q, dim=1, index=actions.unsqueeze(1).long()
            ).squeeze(-1)

            with torch.no_grad():
                target_q = self.compute_target_values(next_states, rewards, dones)
            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation. Finds the optimal greedy policy
        while following an epsilon-greedy policy.

        Use:
            self.epsilon_greedy(state): return probabilities of actions.
            np.random.choice(array, p=prob): sample an element from 'array' based on their corresponding
                probabilites 'prob'.
            self.memorize(state, action, reward, next_state, done): store the transition in the replay buffer
            self.update_target_model(): copy weights from model to target_model
            self.replay(): TD learning for q values on past transitions
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps (HINT: to be done across episodes)
        """

        # Reset the environment
        state, _ = self.env.reset()
        episode_steps = 0
        total_reward = 0
        for _ in range(self.options.steps):
            ################################
            self.n_steps += 1
            episode_steps += 1

            # Select action
            probabilities = self.epsilon_greedy(state)
            action = np.random.choice(self.env.action_space.n, p=probabilities)

            # Execute action
            next_state, reward, done, _ = self.step(action)

            total_reward += reward

            # Store transition
            self.memorize(state, action, reward, next_state, done)

            # Learning step
            self.replay()

            # Update target network periodically
            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()
            state = next_state
            if done:
                self.statistics[Statistics.Rewards.value] = total_reward
                self.statistics[Statistics.Steps.value] = episode_steps
                break
            ################################


    def __str__(self):
        return "DQN"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a greedy
            action
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).detach().numpy()

        return policy_fn
