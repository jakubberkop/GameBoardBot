import gymnasium as gym
import numpy as np
import numpy.typing as npt

from game import AI_PLAYER_ID, GameStep, PlayerDecision, game_is_over, get_game_score, get_legal_moves, initialize_game_state,  RandomPlayer, play_game_until_decision_one_player_that_is_not_a_shop_decision, print_game_state, set_decision

class GameEnv(gym.Env):

	metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

	def __init__(self):
		self.game_state = initialize_game_state()

		state_len = len(self.game_state.to_state_array(AI_PLAYER_ID))

		self.observation_space = gym.spaces.Box(low=0, high=30, shape=(state_len,), dtype=np.float32)
		self.action_space = gym.spaces.Discrete(len(PlayerDecision(0).to_state_array()))

		# assert render_mode is None or render_mode in self.metadata["render_modes"]
		# self.render_mode = render_mode

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
		if self.game_state.state == GameStep.STATE_ERROR:
			assert False, "Invalid action"

		if game_is_over(self.game_state):
			if get_game_score(self.game_state) > 0:
				return 1
			else:
				return -1

		return get_game_score(self.game_state) / 100

	def reset(self, seed=None, options=None):
		# We need the following line to seed self.np_random
		super().reset(seed=seed)

		self.game_state = initialize_game_state()
		random_player = RandomPlayer()

		play_game_until_decision_one_player_that_is_not_a_shop_decision(self.game_state, random_player)

		observation = self.get_obs()

		if self.render_mode == "human":
			self._render_frame()

		return observation, {"legal_moves": get_legal_moves(self.game_state, AI_PLAYER_ID)}

	def _render_frame(self):
		print_game_state(self.game_state)

	def step(self, action: np.ndarray):
		play_game_until_decision_one_player_that_is_not_a_shop_decision(self.game_state, RandomPlayer())

		decision = PlayerDecision.from_state_array(action)

		if decision:
			set_decision(self.game_state, decision, AI_PLAYER_ID)
		else:
			assert False, "Invalid action"

		if self.render_mode == "human":
			print("Decision: ", decision)

		play_game_until_decision_one_player_that_is_not_a_shop_decision(self.game_state, RandomPlayer())

		terminated = self.game_state.end_game or self.game_state.state == GameStep.STATE_ERROR or self.game_state.state == GameStep.STATE_END or game_is_over(self.game_state) #TODO: This is messy

		reward = self.get_reward()

		observation = self.get_obs()

		if self.render_mode == "human":
			self._render_frame()

		return observation, reward, terminated, False, {"legal_moves": get_legal_moves(self.game_state, AI_PLAYER_ID)}

	

	def action_masks(self) -> npt.NDArray[np.float64]:
		"""
		Returns a boolean array of size 9, where each element is True if the action is valid, False otherwise.
		"""
		return get_legal_moves(self.game_state, AI_PLAYER_ID)
	
	