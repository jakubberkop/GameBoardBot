from typing import List

import torch
from transformers import DecisionTransformerConfig, DecisionTransformerModel
import numpy as np

from game import GameState, PlayerDecision, get_legal_moves, initialize_game_state, Player, ACTION_SIZE, STATE_SIZE

weights_path = "results_dt/decision_transformer0"
device = "cuda"
model: DecisionTransformerModel = DecisionTransformerModel.from_pretrained(weights_path).to(device)
TARGET_RETURN = 20

class TransformerPlayer(Player):

	def __init__(self) -> None:
		self.reset()

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		self.states.append(game_state.to_state_array(player_id))
		self.rewards.append(game_state.get_reward(player_id))

		# TODO: Supply the rest of the data

		BATCH_SIZE = 1
		state_count = len(self.states)

		states = torch.from_numpy(np.array(self.states, dtype=np.float32)).reshape(BATCH_SIZE, state_count, STATE_SIZE).to(device=device, dtype=torch.float32)
		actions = torch.zeros((BATCH_SIZE, 1, ACTION_SIZE), device=device, dtype=torch.float32)
		rewards = torch.zeros(BATCH_SIZE, 1, device=device, dtype=torch.float32)
		target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(BATCH_SIZE, 1).to()
		timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(BATCH_SIZE, 1)
		attention_mask = torch.zeros(BATCH_SIZE, 1, device=device, dtype=torch.float32)

		# forward pass	
		with torch.no_grad():
			state_preds, action_preds, return_preds = model(
				states=states,
				actions=actions,
				rewards=rewards,
				returns_to_go=target_return,
				timesteps=timesteps,
				attention_mask=attention_mask,
				return_dict=False,
			)
		
		model_output = action_preds[0][0].cpu()

		legal_moves = get_legal_moves(game_state, player_id)

		assert np.sum(legal_moves) > 0

		masked_output = model_output.masked_fill(~torch.tensor(legal_moves, dtype=torch.bool), float('-inf'))
		self.actions.append(masked_output)

		action = PlayerDecision.from_state_array(masked_output.argmax().item())

		return action

	def name(self) -> str:
		return f"Transformer Player"
	
	def reset(self):
		self.actions: List[float]       = []
		self.states:  List[List[float]] = []
		self.rewards: List[float]       = []
		self.dones:   List[float]       = []