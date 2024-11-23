import vizdoom as vzd
import time
from CDQN import CDQN
from gym.spaces import Box
import numpy as np
from PIL import Image  # Import Pillow


def process_screen_buffer(screen_buffer):
    """
    Process the screen buffer to convert it to grayscale and downsample.

    Args:
        screen_buffer (np.ndarray): The original screen buffer (RGB format).

    Returns:
        np.ndarray: Processed screen buffer as a 2D array.
    """
    # Convert the screen buffer (NumPy array) to a Pillow Image
    image = Image.fromarray(screen_buffer.astype('uint8'), mode='RGB')

    # Convert to grayscale
    image = image.convert("L")

    # Resize to a smaller resolution (e.g., 80x60)
    image = image.resize((80, 60), Image.Resampling.LANCZOS)

    # Convert back to a NumPy array
    return np.array(image)


class Options:
    def __init__(self, replay_memory_size, layers, alpha, gamma, epsilon, batch_size, update_target_estimator_every):
        self.replay_memory_size = replay_memory_size
        self.layers = layers
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.update_target_estimator_every = update_target_estimator_every


options = Options(replay_memory_size=1000, layers=[128, 128], alpha=0.01, gamma=0.9, epsilon=0.2, batch_size=32, update_target_estimator_every=100)

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
            [False, False, False, False, False, False, False, False, False, True],
        ]
        self.action_space = Discrete(len(self.actions))

        # Observation space
        screen_shape = (1, 60, 80)  # Grayscale screen with channel
        game_var_shape = (len(game.get_available_game_variables()),)
        self.observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(np.prod(screen_shape) + np.prod(game_var_shape),),
            dtype=np.float32,
        )

        self.game_variable_names = [str(var) for var in game.get_available_game_variables()]
        self.previous_game_variables = None

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state()
        screen_buffer = np.array(state.screen_buffer, dtype=np.float32)
        processed_screen_buffer = process_screen_buffer(screen_buffer)

        # Add channel dimension for CNN input
        processed_screen_buffer = np.expand_dims(processed_screen_buffer, axis=0)  # Add channel dimension

        game_variables = np.array(state.game_variables, dtype=np.float32)
        self.previous_game_variables = game_variables

        return (processed_screen_buffer, game_variables), {}

    def step(self, action_index):
        action = self.actions[action_index]
        self.game.make_action(action)
        done = self.game.is_episode_finished()
        reward = 0

        if not done:
            state = self.game.get_state()
            screen_buffer = np.array(state.screen_buffer, dtype=np.float32)
            processed_screen_buffer = process_screen_buffer(screen_buffer)
            processed_screen_buffer = np.expand_dims(processed_screen_buffer, axis=0)  # Add channel dimension

            game_variables = np.array(state.game_variables, dtype=np.float32)
            variable_changes = game_variables - self.previous_game_variables

            # Reward logic
            ammo_change_reward = variable_changes[0]
            health_change_penalty = variable_changes[1]
            new_kills = variable_changes[2]
            new_hits = variable_changes[3]
            reward = -1 + 50*ammo_change_reward+ new_hits * 100 + new_kills * 1000 + health_change_penalty*50

            self.previous_game_variables = game_variables

            return (processed_screen_buffer, game_variables), reward, done, {}, {}
        else:
            return (
                (
                    np.zeros((1, 60, 80), dtype=np.float32),
                    np.zeros(len(self.previous_game_variables), dtype=np.float32),
                ),
                reward,
                done,
                {},
                {},
            )

    def close(self):
        self.game.close()


if __name__ == "__main__":
    game = vzd.DoomGame()
    level_path = "defend_the_center.cfg"
    game.load_config(level_path)

    game.clear_available_buttons()
    buttons = [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.MOVE_FORWARD, vzd.Button.USE, vzd.Button.ATTACK,
               vzd.Button.JUMP, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.LOOK_UP, vzd.Button.LOOK_DOWN]
    for button in buttons:
        game.add_available_button(button)

    game.clear_available_game_variables()
    game_variables = [vzd.GameVariable.AMMO2, vzd.GameVariable.HEALTH, vzd.GameVariable.KILLCOUNT, vzd.GameVariable.HITCOUNT]
    for game_variable in game_variables:
        game.add_available_game_variable(game_variable)

    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()

    env = DoomEnv(game)
    eval_env = env

    dqn_agent = CDQN(env=env, eval_env=eval_env, options=options, len_game_variables=len(game_variables), num_actions=len(buttons))

    num_episodes = 1000
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0

        while not done:
            probabilities = dqn_agent.epsilon_greedy(state)
            action = np.random.choice(env.action_space.n, p=probabilities)

            next_state, reward, done, _, _ = env.step(action)
            dqn_agent.memorize(state, action, reward, next_state, done)
            dqn_agent.replay()

            total_reward += reward
            state = next_state
            episode_steps += 1
            dqn_agent.n_steps += 1

            if dqn_agent.n_steps % dqn_agent.options.update_target_estimator_every == 0:
                dqn_agent.update_target_model()
                print('UPDATING TARGET MODEL!')
            if episode_steps == 1000: # end episode after this many steps
                break
        print(f"Episode {episode + 1} finished with total reward: {total_reward}, steps: {episode_steps}")

    env.close()