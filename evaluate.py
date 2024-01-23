from typing import Sequence
from game import *

import tqdm
import numpy as np

Scores = List[int]

def evaluate_player_pair(player0: Player, player1: Player, game_count: int) -> Scores:
	scores: List[int] = []

	for _ in range(game_count):
		game = initialize_game_state()
		play_game(game, player0, player1)

		assert game_is_over(game)

		scores.append(get_game_score(game))

	return scores

def evaluate_players(players: Sequence[Player], game_count: int = 1000):

	scores: List[List[Scores]] = []

	with tqdm.tqdm(total=len(players)**2) as progress_bar:
		for a in players:
			scores.append([])

			for b in players:
				progress_bar.set_postfix_str(f"{a.name()} vs {b.name()}")
				scores[-1].append(evaluate_player_pair(a, b, game_count))
				progress_bar.update(1)

	for i, a in enumerate(players):
		print(a.name())
		for j, b in enumerate(players):
			player_0_win_count = np.sum(np.array(scores[i][j]) > 0)
			player_1_win_count = len(scores[i][j]) - player_0_win_count

			ratio = player_0_win_count / len(scores[i][j])

			print(f"  {b.name():20}: {ratio * 100:2.0f}%    {player_0_win_count}-{player_1_win_count}")


class RandomPlayerCopy(RandomPlayer):

	def name(self) -> str:
		return "Random Player Copy"


def main():
	players = [
		RandomPlayer(),
		RandomPlayerCopy(),
	]

	evaluate_players(players)


if __name__ == "__main__":
	main()