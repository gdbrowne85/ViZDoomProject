import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import numpy as np

from torch.optim import AdamW
from Abstract_Solver import AbstractSolver, Statistics


class QFunction(nn.Module):
    def __init__(self, len_game_variables, num_actions):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 30 * 40 + len_game_variables, 128),  # Adjust input size
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, screen_buffer, game_variables):
        # Pass the screen buffer through the CNN
        screen_features = self.conv_layers(screen_buffer)
        screen_features = torch.flatten(screen_features, start_dim=1)  # Flatten CNN output

        # Concatenate screen features with game variables
        combined_features = torch.cat([screen_features, game_variables], dim=1)

        # Pass through fully connected layers
        return self.fc_layers(combined_features)


class CDQN(AbstractSolver):
    def __init__(self, env, eval_env, options, len_game_variables, num_actions):
        assert str(env.action_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)

        self.model = QFunction(len_game_variables, num_actions)
        self.target_model = deepcopy(self.model)

        self.optimizer = AdamW(
            self.model.parameters(), lr=self.options.alpha, amsgrad=True
        )
        self.loss_fn = nn.SmoothL1Loss()

        for p in self.target_model.parameters():
            p.requires_grad = False

        self.replay_memory = deque(maxlen=self.options.replay_memory_size)
        self.n_steps = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def epsilon_greedy(self, state):
        screen_buffer, game_variables = state
        screen_buffer_tensor = torch.as_tensor(screen_buffer, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        game_variables_tensor = torch.as_tensor(game_variables, dtype=torch.float32).unsqueeze(0)

        q_values = self.model(screen_buffer_tensor, game_variables_tensor)
        n_actions = self.env.action_space.n
        epsilon = self.options.epsilon
        prob_vector = np.ones(n_actions) * (epsilon / n_actions)
        best_action = torch.argmax(q_values).item()
        prob_vector[best_action] += (1 - epsilon)
        return prob_vector / prob_vector.sum()

    def compute_target_values(self, next_states, rewards, dones):
        next_screen_buffers, next_game_variables = zip(*next_states)
        next_screen_buffers = torch.as_tensor(np.stack(next_screen_buffers), dtype=torch.float32)
        next_game_variables = torch.as_tensor(np.stack(next_game_variables), dtype=torch.float32)

        # Add channel dimension if necessary
        if len(next_screen_buffers.shape) == 3:  # [batch_size, H, W]
            next_screen_buffers = next_screen_buffers.unsqueeze(1)

        next_q_values = self.target_model(next_screen_buffers, next_game_variables)
        max_q_values = torch.max(next_q_values, dim=1)[0]
        dones = dones.float()
        target_q = rewards + self.options.gamma * max_q_values * (1 - dones)
        return target_q

    def replay(self):
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            # Convert to NumPy arrays first, then to PyTorch tensors
            screen_buffers = torch.as_tensor(np.stack([s[0] for s in states]), dtype=torch.float32)
            game_variables = torch.as_tensor(np.stack([s[1] for s in states]), dtype=torch.float32)

            next_screen_buffers, next_game_variables = zip(*next_states)
            next_screen_buffers = torch.as_tensor(np.stack(next_screen_buffers), dtype=torch.float32)
            next_game_variables = torch.as_tensor(np.stack(next_game_variables), dtype=torch.float32)

            actions = torch.as_tensor(actions, dtype=torch.long)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Add channel dimension if necessary
            if len(screen_buffers.shape) == 3:  # [batch_size, H, W]
                screen_buffers = screen_buffers.unsqueeze(1)
            if len(next_screen_buffers.shape) == 3:  # [batch_size, H, W]
                next_screen_buffers = next_screen_buffers.unsqueeze(1)

            # Compute current Q-values
            current_q = self.model(screen_buffers, game_variables)
            current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values
            with torch.no_grad():
                target_q = self.compute_target_values(
                    list(zip(next_screen_buffers, next_game_variables)),
                    rewards,
                    dones,
                )

            # Compute loss and update the model
            loss_q = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        episode_steps = 0

        for _ in range(self.options.steps):
            self.n_steps += 1
            probabilities = self.epsilon_greedy(state)
            action = np.random.choice(self.env.action_space.n, p=probabilities)

            next_state, reward, done, _ = self.env.step(action)

            self.memorize(state, action, reward, next_state, done)
            self.replay()

            total_reward += reward
            episode_steps += 1
            state = next_state

            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()

            if done:
                self.statistics[Statistics.Rewards.value] = total_reward
                self.statistics[Statistics.Steps.value] = episode_steps
                break

    def __str__(self):
        return "DQN"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def create_greedy_policy(self):
        def policy_fn(state):
            screen_buffer, game_variables = state
            screen_buffer_tensor = torch.as_tensor(screen_buffer, dtype=torch.float32).unsqueeze(0)
            game_variables_tensor = torch.as_tensor(game_variables, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(screen_buffer_tensor, game_variables_tensor)
            return torch.argmax(q_values).item()

        return policy_fn
