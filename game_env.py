import gymnasium as gym
import numpy as np

from game import AI_PLAYER_ID, STATE_END, STATE_ERROR, Player, GameState, PlayerDecision, game_is_over, get_game_score, initialize_game_state, play_game, RandomPlayer, play_game_until_decision_one_player, play_game_until_decision_one_player_that_is_not_a_shop_decision, set_decision

class GameEnv(gym.Env):

	def __init__(self, render_mode=None, size=5):
		self.game_state = initialize_game_state()

		state_len = len(self.game_state.to_state_array(AI_PLAYER_ID))
		# Observation space is a one dimensional array of size 21
		self.observation_space = gym.spaces.Box(low=0, high=30, shape=(state_len,), dtype=np.float32)

		# We have 4 actions, corresponding to "right", "up", "left", "down"
		self.action_space = gym.spaces.Discrete(9)

		# """
		# The following dictionary maps abstract actions from `self.action_space` to 
		# the direction we will walk in if that action is taken.
		# I.e. 0 corresponds to "right", 1 to "up" etc.
		# """
		# self._action_to_direction = {
		# 	0: np.array([1, 0]),
		# 	1: np.array([0, 1]),
		# 	2: np.array([-1, 0]),
		# 	3: np.array([0, -1]),
		# }

		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode

		"""
		If human-rendering is used, `self.window` will be a reference
		to the window that we draw to. `self.clock` will be a clock that is used
		to ensure that the environment is rendered at the correct framerate in
		human-mode. They will remain `None` until human-mode is used for the
		first time.
		"""
		self.window = None
		self.clock = None

	def get_obs(self):
		return np.array(self.game_state.to_state_array(AI_PLAYER_ID), dtype=np.float32)

	def get_reward(self):
		if self.game_state.state == STATE_ERROR:
			self.reward = -10000
			return self.reward

		return get_game_score(self.game_state) * 100 + self.game_state.turn_counter


	def reset(self, seed=None, options=None):
		# We need the following line to seed self.np_random
		super().reset(seed=seed)

		self.game_state = initialize_game_state()
		random_player = RandomPlayer()

		play_game_until_decision_one_player_that_is_not_a_shop_decision(self.game_state, random_player)

		observation = self.get_obs()

		if self.render_mode == "human":
			self._render_frame()

		return observation, dict()

	def step(self, action: np.ndarray):
		play_game_until_decision_one_player_that_is_not_a_shop_decision(self.game_state, RandomPlayer())

		decision = PlayerDecision.from_state_array(action)

		if decision:
			set_decision(self.game_state, decision, AI_PLAYER_ID)
		else:
			assert False, "Invalid action"

		play_game_until_decision_one_player_that_is_not_a_shop_decision(self.game_state, RandomPlayer())

		terminated = self.game_state.end_game or self.game_state.state == STATE_ERROR or self.game_state.state == STATE_END

		reward = self.get_reward()

		observation = self.get_obs()

		if self.render_mode == "human":
			self._render_frame()


		return observation, reward, terminated, False, dict()