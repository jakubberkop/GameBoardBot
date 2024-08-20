import os
import argparse

from pathlib import Path
from typing import Sequence

import tqdm
import numpy as np
from sb3_contrib import MaskablePPO

from game import *

Scores = List[int]

def evaluate_player_pair(player0: Player, player1: Player, game_count: int) -> Scores:
	scores: List[int] = []

	for _ in tqdm.tqdm(range(game_count), leave=False):
		game = initialize_game_state()
		play_game(game, player0, player1)

		assert game.state != GameStep.STATE_ERROR

		assert game_is_over(game)

		scores.append(get_game_score(game))

		player0.reset()
		player1.reset()

	return scores

def evaluate_players(players: Sequence[Player], game_count: int = 100):

	scores: List[List[Scores]] = []

	with tqdm.tqdm(total=len(players)**2) as progress_bar:
		for a in players:
			scores.append([])

			for b in players:
				progress_bar.set_postfix_str(f"{a.name()} vs {b.name()}")
				scores[-1].append(evaluate_player_pair(a, b, game_count))
				progress_bar.update(1)

	for i, a in enumerate(players):
		for j, b in enumerate(players):
			tie_count          = np.sum(np.array(scores[i][j]) == 0)
			player_0_win_count = np.sum(np.array(scores[i][j]) > 0)
			player_1_win_count = np.sum(np.array(scores[i][j]) < 0)

			ratio_0 = player_0_win_count / len(scores[i][j])
			ratio_1 = player_1_win_count / len(scores[i][j])
			tie_ratio = tie_count / len(scores[i][j])

			print(f"{a.name():20} vs {b.name():20}:  {ratio_0 * 100:2.0f}%-{ratio_1 * 100:2.0f}% [{tie_ratio:2.0f}]%  {player_0_win_count}-{player_1_win_count}:[{tie_count}] ")


class SimplePlayer(RandomPlayer):

	def name(self) -> str:
		return "Simple Player"

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		decision = self._run_player_decision(game_state, player_id)
		action_is_legal = get_legal_moves(game_state, player_id)[decision.encode_action()] == 1.0
		assert action_is_legal, "Illegal action"
		return decision

	def _run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		if game_state.state in [GameStep.STATE_SHOP_0_DECISION, GameStep.STATE_SHOP_1_DECISION]:
			if game_state.state == GameStep.STATE_SHOP_0_DECISION:
				shop_id = 0
			else:
				shop_id = 1

			shop = game_state.shops[shop_id]
			return PlayerDecision(PlayerDecision.Type.SHOP_DECISION, item_id=random.randint(0, len(shop.items) - 1))

		deck = game_state.player_states[player_id].deck
		hand = game_state.player_states[player_id].hand

		can_draw  = sum(deck.values()) > 0
		can_place = sum(hand.values()) > 0

		assert can_draw or can_place, "Player cannot play"

		# Place a card in the queue
		# TODO: For now only first queue
		shop = game_state.shops[0]

		my_score = shop.get_player_score(player_id)
		opponent_score = shop.get_player_score(1 - player_id)
		score_diff_to_beat = opponent_score - my_score
		best_card_type  = max(hand.keys(), key=lambda x: x.value)
		worst_card_type = min(hand.keys(), key=lambda x: x.value)

		if can_draw ^ can_place:
			if can_draw:
				return PlayerDecision(PlayerDecision.Type.DRAW_CARD)

			if my_score > opponent_score:
				# If we are winning, play the card with the lowest cost

				if can_draw:
					return PlayerDecision(PlayerDecision.Type.DRAW_CARD)


				return PlayerDecision(PlayerDecision.Type.PLACE_CARD_IN_QUEUE, worst_card_type, 1, queue_id=0)

			else:
				# If we are losing, play the card with the highest cost

				if can_draw:
					return PlayerDecision(PlayerDecision.Type.DRAW_CARD)

				if score_diff_to_beat > best_card_type.value and can_draw:
					# We cannot win, so we draw
					return PlayerDecision(PlayerDecision.Type.DRAW_CARD)

				return PlayerDecision(PlayerDecision.Type.PLACE_CARD_IN_QUEUE, best_card_type, 1, queue_id=0)

		if my_score > opponent_score:
			# If we are winning, play the card with the lowest cost

			if can_draw:
				return PlayerDecision(PlayerDecision.Type.DRAW_CARD)

			assert False, "This should not happen"

		else:
			# If we are losing, play the card with the highest cost

			if score_diff_to_beat > best_card_type.value and can_draw:
				# We cannot win, so we draw
				return PlayerDecision(PlayerDecision.Type.DRAW_CARD)

			if can_draw:
				return PlayerDecision(PlayerDecision.Type.DRAW_CARD)

			return PlayerDecision(PlayerDecision.Type.PLACE_CARD_IN_QUEUE, best_card_type, 1, queue_id=0)


class HumanPlayer(Player):

	TEXT_TO_DECISION: List[str] = [
		"D",
		"1x1Q0",
		"1x2Q0",
		"1x3Q0",
		"1x4Q0",
		"1x5Q0",
		"1x6Q0",
		"1x7Q0",
		"2x1Q0",
		"2x2Q0",
		"2x3Q0",
		"3x1Q0",
		"LKx1Q0",
		"MDx1Q0",
		"MDx2Q0",
		"PSNx1Q0",
		"PSNx2Q0",
		"PSNx3Q0",
		"PSNx4Q0",
		"PSNx5Q0",
		"PTNx1Q0",
		"SYx1Q0",
		"1x1Q1",
		"1x2Q1",
		"1x3Q1",
		"1x4Q1",
		"1x5Q1",
		"1x6Q1",
		"1x7Q1",
		"2x1Q1",
		"2x2Q1",
		"2x3Q1",
		"3x1Q1",
		"LKx1Q1",
		"MDx1Q1",
		"MDx1Q1",
		"PSNx1Q1",
		"PSNx2Q1",
		"PSNx3Q1",
		"PSNx4Q1",
		"PSNx5Q1",
		"PTNx1Q1",
		"SYx1Q1",
		"I0",
		"I1",
		"S",
		"PTNr1Q0P0",
		"PTNr2Q0P0",
		"PTNr3Q0P0",
		"PTNrLKQ0P0",
		"PTNrMDQ0P0",
		"PTNrPSNQ0P0",
		"PTNrPTNQ0P0",
		"PTNrSYQ0P0",
		"PTNr1Q0P1",
		"PTNr2Q0P1",
		"PTNr3Q0P1",
		"PTNrLKQ0P1",
		"PTNrMDQ0P1",
		"PTNrPSNQ0P1",
		"PTNrPTNQ0P1",
		"PTNrSYQ0P1",
		"PTNr1Q1P0",
		"PTNr2Q1P0",
		"PTNr3Q1P0",
		"PTNrLKQ1P0",
		"PTNrMDQ1P0",
		"PTNrPSNQ1P0",
		"PTNrPTNQ1P0",
		"PTNrSYQ1P0",
		"PTNr1Q1P1",
		"PTNr2Q1P1",
		"PTNr3Q1P1",
		"PTNrLKQ1P1",
		"PTNrMDQ1P1",
		"PTNrPSNQ1P1",
		"PTNrPTNQ1P1",
		"PTNrSYQ1P1",
		"SYr1",
		"SYr2",
		"SYr3",
		"SYrLK",
		"SYrMD",
		"SYrPSN",
		"SYrPTN",
	]

	assert len(TEXT_TO_DECISION) == PlayerDecision.state_space_size(), f"{len(TEXT_TO_DECISION)} != {PlayerDecision.state_space_size()}"

	def decision_text_to_id(self, text: str) -> Optional[int]:
		try:
			return int(text)
		except ValueError:
			pass
		except IndexError:
			pass
		try:
			decision_text = text.upper()
			return HumanPlayer.TEXT_TO_DECISION.index(decision_text)
		except ValueError:
			pass
		except IndexError:
			pass
		
		return None

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:

		while True:
			print()
			print("Possible decisions:")
			legal_moves = get_legal_moves(game_state, player_id)

			for index, legal in enumerate(legal_moves):
				if legal == 1.0:
					print(f"{index:3}: {HumanPlayer.TEXT_TO_DECISION[index]:7} {PlayerDecision.from_encoded_action(index)}")

			decision_text = input("Decision number:")
			decision_text = decision_text.strip()

			decision_id = self.decision_text_to_id(decision_text)

			if decision_id is None:
				print("Invalid decision")
				continue
			
			if decision_id >= len(legal_moves):
				print("Invalid decision")
				continue

			if legal_moves[decision_id] != 1.0:
				print("Ilegal decision")
				continue

			break

		decision = PlayerDecision.from_encoded_action(decision_id)
		assert decision is not None
		return decision

	def name(self) -> str:
		return "Human Player"

class PPOPlayer(Player):

	@staticmethod
	def get_newest_model(model_folder: str = ".", model_mask: str = "ppo_mask") -> "PPOPlayer":
		models = [str(Path(model_folder) / Path(f)) for f in os.listdir(model_folder) if f.startswith(model_mask)]
		models.sort(key=lambda x: os.path.getmtime(x))
		return PPOPlayer.from_model_name(models[-1])

	@staticmethod
	def from_model_name(model_name: str) -> "PPOPlayer":
		return PPOPlayer(MaskablePPO.load(model_name), model_name)

	def __init__(self, model: MaskablePPO, model_name: str) -> None:
		self.model = model
		self.model_name = model_name

		assert self.model is not None
		assert self.model_name is not None

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		obs = np.array(game_state.to_state_array(player_id))
		action, _ = self.model.predict(obs, action_masks=get_legal_moves(game_state, player_id))
		return PlayerDecision.from_encoded_action(action)

	def name(self) -> str:
		return f"PPO: {self.model_name}"


def main(n: int):
	players = [
		RandomPlayer(),
		AlwaysFirstPlayer(),
		AlwaysLastPlayer(),
		# SimplePlayer(),
		# TransformerPlayer(),
		# PPOPlayer("ppo_mask_fixed_reward_500000"),
		# PPOPlayer("ppo_mask_5000"),
		PPOPlayer.get_newest_model(), # Newest model
		PPOPlayer.get_newest_model("ppo_elo", "m_") # Newest model
	]
	evaluate_players(players, n)

def human_game():
	game = initialize_game_state()
	human = HumanPlayer()
	computer = PPOPlayer.from_model_name("ppo_elo/m_2024-08-19_19-14-00.zip")

	play_game(game, computer, human, verbose=True, human_player=1)

	winner = "Human" if game.player_states[1].points > game.player_states[0].points else "Computer"
	print(f"Game over. {winner} won")

	print_game_state(game)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--train", type=bool, default=False, help="Train a model")
	parser.add_argument("-n", type=int, default=100, help="Number of games to play")

	# Adda flag that enables human vs computer
	parser.add_argument("--human", action="store_true", help="Play a game against the computer")

	args = parser.parse_args()

	if args.human:
		human_game()
	else:
		main(args.n)
