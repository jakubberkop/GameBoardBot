import random

from typing import List

import numpy as np

from game import Player, GameState, PlayerDecision, GameStep, get_legal_moves

class RandomPlayer(Player):

	def name(self) -> str:
		return "RandomPlayer"

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		if game_state.state in [GameStep.STATE_SHOP_0_DECISION, GameStep.STATE_SHOP_1_DECISION]:
			if game_state.state == GameStep.STATE_SHOP_0_DECISION:
				shop_id = 0
			else:
				shop_id = 1

			shop = game_state.shops[shop_id]
			return PlayerDecision(PlayerDecision.Type.SHOP_DECISION, item_id=random.randint(0, len(shop.items) - 1))

		possible_decision_types: List[int] = []

		if sum(game_state.player_states[player_id].deck.values()) > 0:
			possible_decision_types.append(PlayerDecision.Type.DRAW_CARD)

		if sum(game_state.player_states[player_id].hand.values()) > 0:
			possible_decision_types.append(PlayerDecision.Type.PLACE_CARD_IN_QUEUE)

		assert len(possible_decision_types) > 0, "Player cannot play"

		type = random.choice(possible_decision_types)

		if type == PlayerDecision.Type.DRAW_CARD:
			return PlayerDecision(PlayerDecision.Type.DRAW_CARD)
		else:
			player_state = game_state.player_states[player_id]
			card_type = random.choice(list(player_state.hand.keys()))
			count = random.randint(1, player_state.hand[card_type])
			# queue_id = random.randint(0, 1) # TODO: Limit to 1 queue for now
			queue_id = 0
			return PlayerDecision(PlayerDecision.Type.PLACE_CARD_IN_QUEUE, card_type, count, queue_id=queue_id)

class AlwaysFirstPlayer(Player):

	def name(self) -> str:
		return "AlwaysFirstPlayer"

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		legal_moves = get_legal_moves(game_state, player_id)
		first_legal = np.argwhere(legal_moves != 0)[-1][0]
		decision = PlayerDecision.from_state_array(first_legal)

		assert decision is not None
		return decision

class AlwaysLastPlayer(Player):

	def name(self) -> str:
		return "AlwaysLastPlayer"

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		legal_moves = get_legal_moves(game_state, player_id)
		last_legal = np.argwhere(legal_moves != 0)[-1][0]
		decision = PlayerDecision.from_state_array(last_legal)

		assert decision is not None
		return decision

