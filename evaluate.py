from typing import Sequence

from game import *

import tqdm
import numpy as np

Scores = List[int]

import pytracy
pytracy.set_tracing_mode(pytracy.TracingMode.MarkedFunctions)

def evaluate_player_pair(player0: Player, player1: Player, game_count: int) -> Scores:
	scores: List[int] = []

	for _ in tqdm.tqdm(range(game_count), leave=False):
		game = initialize_game_state()
		play_game(game, player0, player1, skip_shop_decisions=True)

		assert game_is_over(game)

		scores.append(get_game_score(game))

		player0.reset()
		player1.reset()

	return scores

def evaluate_players(players: Sequence[Player], game_count: int = 5000):

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

			print(f"{a.name():20} vs {b.name():20}:  {ratio_0 * 100:2.0f}%-{ratio_1 * 100:2.0f}% [{tie_count:2.0f}]%  {player_0_win_count}-{player_1_win_count}:[{tie_count}] ")


class SimplePlayer(RandomPlayer):

	def name(self) -> str:
		return "Simple Player"

	def run_player_decision(self, game_state: GameState, player_id: int, legal_moves: Optional[np.ndarray] = None) -> PlayerDecision:
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

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:

		while True:
			print()
			print("Possible decisions:")
			legal_moves = get_legal_moves(game_state, player_id)

			# Print legal moves
			for i, move in enumerate(legal_moves):
				if move:
					# TODO: This crashes
					print(f"{i}: {PlayerDecision.from_state_array(to_one_hot(i, ACTION_SIZE))}")

			decision = input("Decision number:")

			try:
				decision = int(decision)

				if legal_moves[decision]:
					break
			except ValueError:
				pass
			except IndexError:
				pass

			print("Invalid decision")

		player_decision = PlayerDecision.from_state_array(to_one_hot(int(decision), ACTION_SIZE))
		assert player_decision is not None
		return player_decision


	def name(self) -> str:
		return "Human Player"

from transformer_player import TransformerPlayer

@pytracy.mark_function
def main():
	players = [
		# HumanPlayer(),
		RandomPlayer(),
		SimplePlayer(),
		# TransformerPlayer(),
		# SimplePlayer(),
	]
	import time
	a = time.time()
	evaluate_players(players)
	b = time.time()
	print(b-a)


def human_game():
	game = initialize_game_state()
	computer = RandomPlayer()
	# human = HumanPlayer()
	human = SimplePlayer()
	play_game(game, computer, human, skip_shop_decisions=True, verbose=True)

def evaluate_t():
	game = initialize_game_state()
	tran = TransformerPlayer()
	random = RandomPlayer()
	play_game(game, tran, random, skip_shop_decisions=True, verbose=True)

if __name__ == "__main__":
	# main()
	human_game()
	evaluate_t()