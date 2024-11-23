import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import numpy as np
from random import choice
import os


class ViZDoomEnv(gym.Env):
    def __init__(self):
        super(ViZDoomEnv, self).__init__()

        # Initialize DoomGame instance
        self.game = vzd.DoomGame()

        # Set up ViZDoom configurations
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_sectors_info_enabled(True)

        # Define action space and observation space
        self.action_space = spaces.Discrete(3)  # Three actions: Move forward, turn left, turn right
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)

        # Additional buttons and variables
        self.game.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT])
        self.game.set_available_game_variables([vzd.GameVariable.POSITION_X])

        # Set episode timeout
        self.game.set_episode_timeout(2000)

        # Enable rendering
        self.game.set_window_visible(True)

        # Set mode to player
        self.game.set_mode(vzd.Mode.PLAYER)

        # Initialize the game
        self.game.init()

        # Define actions mapping
        self.actions = [
            [True, False, False],  # MOVE_FORWARD
            [False, True, False],  # TURN_LEFT
            [False, False, True]  # TURN_RIGHT
        ]

        # Initialize episode variables
        self.previous_x_position = 0

    def reset(self):
        # Start a new episode
        self.game.new_episode()
        self.previous_x_position = self.game.get_game_variable(vzd.GameVariable.POSITION_X)

        # Get initial state
        state = self.game.get_state()
        obs = state.screen_buffer  # RGB screen buffer

        return obs, {}

    def step(self, action):
        # Perform the action in ViZDoom and get the reward
        self.game.make_action(self.actions[action])
        current_x_position = self.game.get_game_variable(vzd.GameVariable.POSITION_X)

        # Calculate reward based on change in position
        reward = current_x_position - self.previous_x_position
        self.previous_x_position = current_x_position

        # Check if episode is done
        done = self.game.is_episode_finished()

        # Get the next observation or reset if done
        if not done:
            state = self.game.get_state()
            obs = state.screen_buffer
        else:
            obs = np.zeros((240, 320, 3), dtype=np.uint8)  # Return a blank screen when done

        return obs, reward, done, {}

    def render(self, mode="human"):
        # ViZDoom handles rendering automatically with `set_window_visible`
        pass

    def close(self):
        self.game.close()
