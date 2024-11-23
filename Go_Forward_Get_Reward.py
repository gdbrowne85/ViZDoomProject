from gymnasium_wrapper.base_gymnasium_env import VizdoomEnv
import time
from DQN import DQN

from gym import ObservationWrapper
from gym.spaces import Box

class GameVariablesEnv(ObservationWrapper):
    def __init__(self, env):
        super(GameVariablesEnv, self).__init__(env)
        # Set the observation space to just the 'gamevariables'
        self.observation_space = env.observation_space['gamevariables']

    def observation(self, obs):
        # Extract 'gamevariables' from the observation dictionary
        return obs['gamevariables']

class Options:
    def __init__(self, replay_memory_size, layers, alpha, gamma):
        self.replay_memory_size = replay_memory_size
        self.layers = layers
        self.alpha = alpha
        self.gamma = gamma
options = Options(replay_memory_size=1000, layers=[128, 128], alpha=0.01, gamma=0.9)

if __name__ == "__main__":

    # Path to the Doom config file, which specifies the level and game settings
    level_path = "deadly_corridor.cfg"  # Update this path to the actual configuration file location

    # Create an instance of the VizdoomEnv environment
    env = VizdoomEnv(level=level_path, render_mode="human")
    # Wrap the environment to use only 'gamevariables'
    env = GameVariablesEnv(env)
    env.reset()
    eval_env = env
    print('Env: ', env)
    print('Env.observation_space: ', env.observation_space)
    dqn_agent = DQN(env=env, eval_env=eval_env, options=options)

    # Define the number of episodes to run
    num_episodes = 5

    # Run through each episode
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")
        state, _ = env.reset()  # Reset the environment to start a new episode
        done = False
        total_reward = 0

        while not done:
            time.sleep(0.01)
            # Select a random action from the action space
            action = env.action_space.sample()

            # Take the action and get the new observation, reward, and done flag
            next_state, reward, terminated, truncated, info = env.step(action)

            # Update total reward for the current episode
            total_reward += reward
            done = terminated or truncated  # Stop the episode if terminated or truncated

            # Print the reward obtained from this step
            print(f"Action: {action}, Reward: {reward}")

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    # Close the environment once done
    env.close()