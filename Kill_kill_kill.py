import vizdoom as vzd
import time
from DQN import DQN
from gym.spaces import Box
import numpy as np


class Options:
    def __init__(self, replay_memory_size, layers, alpha, gamma, epsilon, batch_size, update_target_estimator_every):
        self.replay_memory_size = replay_memory_size
        self.layers = layers
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.update_target_estimator_every = update_target_estimator_every

options = Options(replay_memory_size=1000, layers=[128, 128], alpha=0.01, gamma=0.9, epsilon=0.4, batch_size=32, update_target_estimator_every=100)

from gym.spaces import Discrete

class DoomEnv:
    def __init__(self, game):
        self.game = game
        # Define the action space as discrete, with one action per combination
        self.actions = [
            [True, False, False, False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False, False, False],
            [False, False, True, False, False, False, False, False, False, False],
            [False, False, False, True, False, False, False, False, False, False],
            [False, False, False, False, True, False, False, False, False, False],
            [False, False, False, False, False, True, False, False, False, False],
            [False, False, False, False, False, False, True, False, False, False],
            [False, False, False, False, False, False, False, True, False, False],
            [False, False, False, False, False, False, False, False, True, False],
            [False, False, False, False, False, False, False, False, False, True]
        ]
        self.action_space = Discrete(len(self.actions))
        self.observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(len(game.get_available_game_variables()),),
            dtype=np.float32
        )
        # Store the names of the game variables
        self.game_variable_names = [
            str(var) for var in game.get_available_game_variables()
        ]
        # To track game variable changes
        self.previous_game_variables = None

    def reset(self):
        """Resets the game to start a new episode and returns the initial state."""
        self.game.new_episode()
        state = self.game.get_state()
        game_variables = state.game_variables  # Extract game variables
        # Initialize previous game variables
        self.previous_game_variables = np.array(game_variables, dtype=np.float32)
        return np.array(game_variables, dtype=np.float32), {}

    def step(self, action_index):
        """Takes an action, advances the game, and returns the result."""
        # Map the discrete action index to the corresponding action combination
        action = self.actions[action_index]
        self.game.make_action(action)
        done = self.game.is_episode_finished()
        reward = 0

        if not done:
            state = self.game.get_state()
            game_variables = np.array(state.game_variables, dtype=np.float32)

            # Compute changes in game variables
            variable_changes = game_variables - self.previous_game_variables

            # Custom reward logic based on changes:
            # Reward for collecting ammo, penalize for losing health, and encourage movement
            ammo_change_reward = variable_changes[0]  # Change in AMMO2
            health_change_penalty = -variable_changes[1]  # Negative if health decreases
            newkills = variable_changes[2]  # Change in POSITION_X encourages movement

            # Aggregate reward
            reward = -1 + newkills*1000 # rewarded for killing

            # Update previous game variables
            self.previous_game_variables = game_variables

            # Print game variable names, values, and changes
            print("Game Variables:")
            for name, value, change in zip(self.game_variable_names, game_variables, variable_changes):
                print(f"  {name}: {value} (Change: {change})")

            return game_variables, reward, done, {}, {}
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, done, {}, {}

    def close(self):
        """Closes the game."""
        self.game.close()




if __name__ == "__main__":
    # Initialize DoomGame
    game = vzd.DoomGame()
    level_path = "simpler_basic.cfg"  # Update this to the actual configuration file path
    game.load_config(level_path)

    # Overwrite available actions
    game.clear_available_buttons()
    buttons = [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.MOVE_FORWARD, vzd.Button.USE, vzd.Button.ATTACK,
               vzd.Button.JUMP, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.LOOK_UP, vzd.Button.LOOK_DOWN]
    for button in buttons:
        game.add_available_button(button)

    # Overwrite available game variables
    game.clear_available_game_variables()
    game_variables = [vzd.GameVariable.AMMO2, vzd.GameVariable.HEALTH, vzd.GameVariable.KILLCOUNT]
    for game_variable in game_variables:
        game.add_available_game_variable(game_variable)
    # Configure the game
    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_mode(vzd.Mode.PLAYER)

    # Initialize the game
    game.init()

    # Wrap DoomGame in a custom environment
    env = DoomEnv(game)
    eval_env = env

    print("Env: ", env)
    print("Env.observation_space: ", env.observation_space)

    # Initialize DQN with the environment
    dqn_agent = DQN(env=env, eval_env=eval_env, options=options)

    # Run episodes
    num_episodes = 100
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")
        state, _ = env.reset()
        done = False
        total_reward = 0

        # Run episodes
        for episode in range(num_episodes):
            print(f"Starting episode {episode + 1}")
            state, _ = env.reset()
            done = False
            total_reward = 0
            episode_steps = 0

            while not done:
                print('Episode Step: ', episode_steps)
                print('DQN_N_steps: ',dqn_agent.n_steps)
                time.sleep(0.0001)

                # Select an action using epsilon-greedy policy
                probabilities = dqn_agent.epsilon_greedy(state)
                print('Action Probabilities: ', probabilities)
                action = np.random.choice(env.action_space.n, p=probabilities)

                # Execute the action
                next_state, reward, done, _, _ = env.step(action)

                # Store the transition in replay memory
                dqn_agent.memorize(state, action, reward, next_state, done)

                # Perform a learning step
                dqn_agent.replay()

                # Update total reward and state
                total_reward += reward
                state = next_state
                episode_steps += 1
                dqn_agent.n_steps += 1
                print(f"State: {next_state}, Action: {action}, Reward: {reward}")

                # Update target network periodically
                if dqn_agent.n_steps % dqn_agent.options.update_target_estimator_every == 0:
                    dqn_agent.update_target_model()
                    print('UPDATED TARGET MODEL!!!!!!!!!!')
                if episode_steps == 1000: # end episode after this many steps
                    break

            print(f"Episode {episode + 1} finished with total reward: {total_reward}, steps: {episode_steps}")



        env.close()
