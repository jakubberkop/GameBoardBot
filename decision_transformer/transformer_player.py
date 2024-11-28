from typing import List

import torch
from transformers import DecisionTransformerConfig, DecisionTransformerModel
import numpy as np

from game import GameState, PlayerDecision, get_legal_moves, initialize_game_state, Player, ACTION_SIZE, STATE_SIZE

weights_path = "results_dt/decision_transformer1"
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

		states  = torch.from_numpy(np.array(self.states)) .reshape(BATCH_SIZE, state_count, STATE_SIZE)
		actions = torch.from_numpy(np.array(self.actions)).reshape(BATCH_SIZE, state_count, PlayerDecision.state_space_size())
		rewards = torch.tensor(self.rewards).reshape(BATCH_SIZE, state_count)
		target_return = torch.tensor(TARGET_RETURN).reshape(BATCH_SIZE, 1)
		timesteps = torch.tensor([i for i in range(state_count)]).reshape(BATCH_SIZE, state_count)
		attention_mask = torch.ones(BATCH_SIZE, state_count)

		states = states                .to(device=device, dtype=torch.float32)
		actions = actions              .to(device=device, dtype=torch.float32)
		rewards = rewards              .to(device=device, dtype=torch.float32)
		target_return = target_return  .to(device=device, dtype=torch.float32)
		timesteps = timesteps          .to(device=device, dtype=torch.long)
		attention_mask = attention_mask.to(device=device, dtype=torch.float32)

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
		self.actions: List[float]       = [np.zeros(ACTION_SIZE)]
		self.states:  List[List[float]] = []
		self.rewards: List[float]       = []
		self.dones:   List[float]       = []