from game import *


def run_shop(game_runner: GameRunner, shop_id: int, player_id: int) -> None:
	shop = game_runner.game_state.shops[shop_id]

	score0 = shop.get_player_score(0)
	score1 = shop.get_player_score(1)

	if score0 == score1:
		return
	
	if score0 > score1:
		winner = 0
	else:
		winner = 1

	if player_id != winner:
		print("This should never happen YO")
		return

	if shop.get_item_count() == 2:
		pass
		# TODO: Fixup
		# winning_player = game_runner.players[winner]
		# shop_decision = winning_player.run_shop_item_decision(game_runner.game_state, winner, shop_id)
		# shop.place_item_in_limbo(shop_decision.item_id, winner)
	elif shop.get_item_count() == 1:
		# Get the item from the limbo
		assert shop.limbo_item

		limbo_item = shop.limbo_item
		player_states = game_runner.game_state.player_states

		player_states[limbo_item.player_id].points += limbo_item.item.value
		player_states[winner].points += shop.items[0].value

		game_runner.game_state.shops[shop_id] = ShopState()
		shop = game_runner.game_state.shops[shop_id]

		# If 2+ items left in deck, draw 2
		# if 1 item left in deck, draw 1
		# if 0 items left in deck, invalidate shop
		items_to_draw = sum(game_runner.game_state.items_deck.values())

		if items_to_draw == 0:
			game_runner.game_state.end_game = True
			shop.items = []
			return
		elif items_to_draw >= 2:
			shop.items = [draw_from_deck(game_runner.game_state.items_deck), draw_from_deck(game_runner.game_state.items_deck)]
		else:
			# items_to_draw == 1 is not possible, as we number of items in deck is always even
			assert False, "Invalid game state"
	else:
		assert False


def run_player_turn(game_runner: GameRunner, player_id: int):
	run_shop(game_runner, 0, player_id)
	if game_runner.game_state.end_game:
		return

	run_shop(game_runner, 1, player_id)
	if game_runner.game_state.end_game:
		return

	player = game_runner.players[player_id]
	player_state = game_runner.game_state.player_states[player_id]

	def execute_decision(player_decision: PlayerDecision):
		if player_decision.type == PlayerDecision.DRAW_CARD:
			draw_card = draw_from_deck(player_state.deck)
			player_state.hand[draw_card] += 1
		elif player_decision.type == PlayerDecision.PLACE_CARD_IN_QUEUE:
			assert player_decision.count
			assert player_decision.card_type

			player_state.hand[player_decision.card_type] -= player_decision.count
			if player_state.hand[player_decision.card_type] == 0:
				del player_state.hand[player_decision.card_type]
			game_runner.game_state.shops[player_id].queue.append(QueueItem(player_decision.card_type, player_decision.count, player_id))

		else:
			assert False, "Invalid player decision type"

	if not player_can_play(player_state):
		game_runner.game_state.end_game = True
		return

	player_decision = player.run_player_decision(game_runner.game_state, player_id)
	assert player_decision.isValid()
	assert decision_can_be_played(player_decision, player_state)
	execute_decision(player_decision)

	if not player_can_play(player_state):
		game_runner.game_state.end_game = True
		return

	player_decision = player.run_player_decision(game_runner.game_state, player_id)
	assert player_decision.isValid()
	assert decision_can_be_played(player_decision, player_state)
	execute_decision(player_decision)

def play_game(player0: Player, player1: Player) -> GameState:
	game_runner = GameRunner(game_state = initialize_game_state(), players = [player0, player1])

	while not game_is_over(game_runner.game_state):
		run_player_turn(game_runner, 0)
		if game_runner.game_state.end_game:
			break

		run_player_turn(game_runner, 1)
		if game_runner.game_state.end_game:
			break

	assert validate_game_state(game_runner.game_state)

	return game_runner.game_state