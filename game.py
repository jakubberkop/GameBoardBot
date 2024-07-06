from dataclasses import dataclass, field
from enum import IntEnum
import random
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple

# import torch
import tqdm
import numpy as np
import numpy.typing as npt

# from pytracy import *
# set_tracing_mode(TracingMode.All)

@dataclass
class CardType:
	name: str
	id: int

	def __hash__(self):
		return hash(self.name)
	
	def __eq__(self, other: Any):
		return self.name == other.name
	
	@property
	def value(self) -> int:
		if self.name == "1":
			return 1
		elif self.name == "2":
			return 2
		elif self.name == "3":
			return 3
		elif self.name == "LK":
			return 1
		elif self.name == "MD":
			return 2
		elif self.name == "PSN":
			return 1
		elif self.name == "PTN":
			return 1
		elif self.name == "SY":
			return 1
		else:
			assert False, "Invalid card type"

	@staticmethod
	def from_id(id: int) -> Optional["CardType"]:
		for card_type, _ in CARD_INFO:
			if card_type.id == id:
				return card_type
		return None

@dataclass
class Item:
	name: str
	value: int

	def __hash__(self):
		return hash(self.name)
	
	def __eq__(self, other: Any):
		return self.name == other.name

CARD_INFO: List[Tuple[CardType, int]] = [
	(CardType("1", 0),   7),
	(CardType("2", 1),   3),
	(CardType("3", 2),   1),
	(CardType("LK", 3),  1),
	(CardType("MD", 4),  2),
	(CardType("PSN", 5), 5),
	(CardType("PTN", 6), 1),
	(CardType("SY", 7),  1),
]

def get_default_card_deck() -> Dict[CardType, int]:
	return {card_type: count for card_type, count in CARD_INFO}

@dataclass
class PlayerState:
	hand: DefaultDict[CardType, int] = field(default_factory=lambda: DefaultDict(int))
	deck: Dict[CardType, int] = field(default_factory=dict)
	points: int = 0

	def to_state_array(self, private: bool) -> List[int]:
		state: List[int] = [
			self.points,
			sum(self.deck.values()),
		]

		for card_type, _ in CARD_INFO:
			state.append(self.hand.get(card_type, 0))

		return state

	STATE_SIZE = 2 + len(CARD_INFO)

	def to_state_array_fast(self, array, index, private: bool) -> None:
		array[index] = self.points
		array[index+1] = sum(self.deck.values())

		for i, (card_type, _) in enumerate(CARD_INFO):
			array[index+2+i] = self.hand.get(card_type, 0)
		

	def __str__(self):
		return f"Points: {self.points} Hand: {self.hand}"

@dataclass
class QueueItem:
	card_type: CardType
	count: int
	player_id: int

@dataclass
class LimboItem:
	item: Item
	player_id: int

@dataclass
class ShopState:
	items: List[Item] = field(default_factory=list)
	queue: List[QueueItem] = field(default_factory=list)

	limbo_item: Optional[LimboItem] = None
	STATE_SIZE: int = 0

	def __post_init__(self):
		self.STATE_SIZE = len(self.to_state_array(0))

	def get_player_score(self, player_id: int) -> int:
		# TODO: Implement other scoring rules
		score = sum(queue_item.card_type.value * queue_item.count for queue_item in self.queue if queue_item.player_id == player_id)

		if self.limbo_item and self.limbo_item.player_id == player_id:
			score -= self.limbo_item.item.value

		return score

	def get_item_count(self) -> int:
		return len(self.items)
	
	def place_item_in_limbo(self, item_id: int, player_id: int) -> None:
		self.limbo_item = LimboItem(self.items[item_id], player_id)
		self.items.pop(item_id)

	def to_state_array(self, player_id: int) -> List[int]:
		state: List[int] = []

		item_count = len(self.items)

		if item_count == 2:
			state.append(self.items[0].value)
			state.append(self.items[1].value)
			state.append(self.limbo_item.item.value if self.limbo_item else 0)
		elif item_count == 1:
			state.append(self.items[0].value)
			state.append(0)
			state.append(self.limbo_item.item.value if self.limbo_item else 0)
		else:
			state.append(0)
			state.append(0)
			state.append(self.limbo_item.item.value if self.limbo_item else 0)

		# player_id
		for card_type, _ in CARD_INFO:
			card_count = sum(queue_item.count for queue_item in self.queue if queue_item.card_type == card_type and queue_item.player_id == player_id)
			state.append(card_count)

		# opponent_id
		for card_type, _ in CARD_INFO:
			card_count = sum(queue_item.count for queue_item in self.queue if queue_item.card_type == card_type and queue_item.player_id != player_id)
			state.append(card_count)

		return state
	
	def to_state_array_fast(self, state, index, player_id: int) -> None:
		# slow_state = self.to_state_array(player_id)
		# start_index = index
		# for i in range(len(slow_state)):
		# 	state[index + i] = slow_state[i]

		item_count = len(self.items)

		if item_count == 2:
			state[index + 0] = self.items[0].value
			state[index + 1] = self.items[1].value
			state[index + 2] = self.limbo_item.item.value if self.limbo_item else 0
		elif item_count == 1:
			state[index + 0] = self.items[0].value
			state[index + 1] = 0
			state[index + 2] = self.limbo_item.item.value if self.limbo_item else 0
		else:
			state[index + 0] = 0
			state[index + 1] = 0
			state[index + 2] = self.limbo_item.item.value if self.limbo_item else 0

		index += 3

		# player_id
		for card_type, _ in CARD_INFO:
			card_count = sum(queue_item.count for queue_item in self.queue if queue_item.card_type == card_type and queue_item.player_id == player_id)
			state[index] = card_count
			index += 1

		# opponent_id
		for card_type, _ in CARD_INFO:
			card_count = sum(queue_item.count for queue_item in self.queue if queue_item.card_type == card_type and queue_item.player_id != player_id)
			state[index] = card_count
			index += 1

		# # comppare to slow state
		# for i in range(len(slow_state)):
		# 	assert state[start_index + i] == slow_state[i]

	def __str__(self):
		item_str = " ".join([item.name for item in self.items])
		queue_str = " ".join([f"{queue_item.card_type.name}x{queue_item.count}" for queue_item in self.queue])
		return f"Items: {item_str} Queue: {queue_str} Limbo: {self.limbo_item}"

def to_one_hot(value: int, count: int) -> List[int]:
	return [1 if value == i else 0 for i in range(count)]

class PlayerDecision:
	class Type(IntEnum):
		DRAW_CARD = 0
		PLACE_CARD_IN_QUEUE = 1
		SHOP_DECISION = 2 # Shop decision is commented for now

	type: Type

	# PLACE_CARD_IN_QUEUE
	card_type: Optional[CardType] = None
	count: Optional[int] = None
	queue_id: Optional[int] = None

	# SHOP_DECISION
	item_id: Optional[int] = None

	def __init__(self, type: int, card_type: Optional[CardType] = None, count: Optional[int] = None, queue_id: Optional[int] = None, item_id: Optional[int] = None):
		self.type = type
		self.card_type = card_type
		self.count = count
		self.item_id = item_id
		self.queue_id = queue_id

	def isValid(self):
		if self.type == PlayerDecision.Type.DRAW_CARD:
			return True
		elif self.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			return self.card_type is not None and self.count is not None
		# elif self.type == PlayerDecision.Type.SHOP_DECISION:
		# 	return self.item_id is not None
		else:
			return False

	def to_state_array(self) -> List[int]:
		if self.type == PlayerDecision.Type.DRAW_CARD:
			encoded_type = 0
		elif self.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			encoded_type = 1 + self.card_type.id
		# elif self.type == PlayerDecision.Type.SHOP_DECISION:
		# 	encoded_type = 1 + len(CARD_INFO)
		else:
			assert False

		return to_one_hot(encoded_type, 1 + len(CARD_INFO))

	def to_state_array_fast(self) -> npt.NDArray[np.float32]:
		if self.type == PlayerDecision.Type.DRAW_CARD:
			encoded_type = 0
		elif self.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			encoded_type = 1 + self.card_type.id
		# elif self.type == PlayerDecision.Type.SHOP_DECISION:
		# 	encoded_type = 1 + len(CARD_INFO)
		else:
			assert False

		result = np.zeros(1 + len(CARD_INFO), dtype=np.float32)
		result[encoded_type] = 1
		return result

	@staticmethod
	def from_state_array(state: float | int) -> Optional["PlayerDecision"]:
		# if type(state) == torch.Tensor or type(state) == list:
			# action_type = torch.argmax(torch.tensor(state))
		# is_numpy_numeral = type(state) == np.float32 or type(state) == np.int32 or type(state) == np.int64 or type(state) == np.float64
		# if type(state) == float or type(state) == int or is_numpy_numeral:
		# 	action_type = int(state)
		# else:
		# 	assert False, "Invalid state type"

		try:
			action_type = int(state)
		except:
			assert False, "Invalid state type"


		if action_type == PlayerDecision.Type.DRAW_CARD:
			return PlayerDecision(PlayerDecision.Type.DRAW_CARD)
		elif 0 < action_type and action_type <= len(CARD_INFO):
			card_type_id = action_type - 1
			card_type = CardType.from_id(card_type_id)

			# card_count = int(state[len(CARD_INFO)+1])
			# TODO: Card count is always 1 for now

			card_count = 1

			# TODO: Queue id is always 0 for now
			queue_id = 0


			return PlayerDecision(PlayerDecision.Type.PLACE_CARD_IN_QUEUE, card_type, card_count, queue_id=queue_id)

		# TODO: Shop decision is commented for now
		# elif action_type == PlayerDecision.Type.SHOP_DECISION:
		# 	item_id = int(state[-1])
		# 	return PlayerDecision(PlayerDecision.Type.SHOP_DECISION, item_id=item_id)

		else:
			assert False, "Invalid action type"

	def __eq__(self, other: Any):
		if self.type != other.type:
			return False
		
		if self.type == PlayerDecision.Type.DRAW_CARD:
			return True
		
		if self.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			return self.card_type == other.card_type and self.count == other.count
		
		# TODO: Shop decision is commented for now
		# if self.type == PlayerDecision.Type.SHOP_DECISION:
		# 	return self.item_id == other.item_id
		
		assert False

	def __str__(self):
		if self.type == PlayerDecision.Type.DRAW_CARD:
			return f"Draw card"
		elif self.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			return f"Place {self.card_type.name}x{self.count} in queue {self.queue_id}"
		
		# TODO: Shop decision is commented for now
		elif self.type == PlayerDecision.Type.SHOP_DECISION:
			return f"Place item {self.item_id} in shop"
		else:
			assert False

class GameStep(IntEnum):
	STATE_START = 0
	STATE_SHOP_0 = 1
	STATE_SHOP_0_DECISION = 2
	STATE_SHOP_1 = 3
	STATE_SHOP_1_DECISION = 4
	STATE_TURN_0 = 5
	STATE_TURN_1 = 7
	STATE_TURN_2 = 9
	STATE_END_TURN = 11
	STATE_END = 12
	STATE_ERROR = 13


DECISION_STATES: Set[GameStep] = {
	GameStep.STATE_SHOP_0_DECISION, GameStep.STATE_SHOP_1_DECISION, 
	GameStep.STATE_TURN_0, GameStep.STATE_TURN_1,
	# GameStep.STATE_TURN_2 #TODO: Implement GameStep.STATE_TURN_2
}

END_GAME_STATES: Set[GameStep] = {
	GameStep.STATE_END, GameStep.STATE_ERROR
}

AI_PLAYER_ID = 0
NPC_PLAYER_ID = 1


@dataclass
class GameState:
	player_states: List[PlayerState] = field(default_factory=list)
	shops: List[ShopState] = field(default_factory=list)
	items_deck: Dict[Item, int] = field(default_factory=dict)
	end_game: bool = False
	state: int = GameStep.STATE_START
	turn: int = 0
	turn_counter: int = 0

	def to_state_array(self, player_id: int) -> List[int]:
		state = np.zeros(1 + 2 * PlayerState.STATE_SIZE + 2 * self.shops[0].STATE_SIZE, dtype=np.float32)
		state[0] = self.state

		index = 0
		for player_state in self.player_states:
			player_state.to_state_array_fast(state, index, player_id == 0)
			index += PlayerState.STATE_SIZE
	
		for shop in self.shops:
			shop.to_state_array_fast(state, index, player_id)
			index += ShopState.STATE_SIZE

		return state
	
	def get_reward(self, player_id: int) -> int:
		if player_id == AI_PLAYER_ID:
			return get_game_score(self)
		else:
			return -get_game_score(self)

	def __str__(self):
		return f""" {self.state}

Player 0: {self.player_states[0]}
Player 1: {self.player_states[1]}

Shop 0: {self.shops[0]}
Shop 1: {self.shops[1]}
"""

class Player:

	def name(self) -> str:
		return False, "Not implemented" # type: ignore

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		assert False, "Not implemented"

	def reset(self) -> None:
		pass

@dataclass
class GameRunner:
	game_state: GameState
	players: List[Player] = field(default_factory=list)

def get_default_item_deck() -> Dict[Item, int]:
	return {
		Item("1", 1): 2,
		Item("2", 2): 6,
		Item("3", 3): 7,
		Item("4", 4): 7,
		Item("5", 5): 8,
		Item("7", 7): 4,
		Item("8", 8): 2,
	}

def draw_from_deck(deck: Dict[Any, int]) -> Any:
	item = random.choices(list(deck.keys()), list(deck.values()))[0]
	deck[item] -= 1

	if deck[item] == 0:
		del deck[item]

	return item

def initialize_player_state() -> PlayerState:
	player_state = PlayerState()

	player_state.deck = get_default_card_deck()

	for _ in range(5):
		draw_card = draw_from_deck(player_state.deck)
		player_state.hand[draw_card] = player_state.hand.get(draw_card, 0) + 1

	return player_state

def initialize_game_state() -> GameState:
	game_state = GameState()

	game_state.player_states = [initialize_player_state(), initialize_player_state()]
	game_state.items_deck = get_default_item_deck()
	game_state.shops = [ShopState(items=[draw_from_deck(game_state.items_deck) for _ in range(2)]) for _ in range(2)]

	return game_state

def run_shop_until(game_state: GameState, shop_id: int, player_id: int) -> None:
	shop = game_state.shops[shop_id]

	score0 = shop.get_player_score(0)
	score1 = shop.get_player_score(1)

	if score0 == score1:
		if shop_id == 0:
			game_state.state = GameStep.STATE_SHOP_1
		else:
			game_state.state = GameStep.STATE_TURN_0
		return
	
	if score0 > score1:
		winner = 0
	else:
		winner = 1

	if player_id != winner:
		if shop_id == 0:
			game_state.state = GameStep.STATE_SHOP_1
		else:
			game_state.state = GameStep.STATE_TURN_0
		return

	if shop.get_item_count() == 2:
		if shop_id == 0:
			game_state.state = GameStep.STATE_SHOP_0_DECISION
		else:
			game_state.state = GameStep.STATE_SHOP_1_DECISION
		return

	elif shop.get_item_count() == 1:
		# Get the item from the limbo
		assert shop.limbo_item

		limbo_item = shop.limbo_item
		player_states = game_state.player_states

		player_states[limbo_item.player_id].points += limbo_item.item.value
		player_states[winner].points += shop.items[0].value

		game_state.shops[shop_id] = ShopState()
		shop = game_state.shops[shop_id]

		# If 2+ items left in deck, draw 2
		# if 1 item left in deck, draw 1
		# if 0 items left in deck, invalidate shop
		items_to_draw = sum(game_state.items_deck.values())

		if items_to_draw == 0:
			game_state.end_game = True
			shop.items = []
			game_state.state = GameStep.STATE_END
			return
		elif items_to_draw >= 2:
			shop.items = [draw_from_deck(game_state.items_deck), draw_from_deck(game_state.items_deck)]
		else:
			# items_to_draw == 1 is not possible, as we number of items in deck is always even
			assert False, "Invalid game state"
	else:
		assert False

def player_can_play(player_state: PlayerState) -> bool:
	return sum(player_state.deck.values()) > 0 or sum(player_state.hand.values()) > 0

def decision_can_be_played(decision: PlayerDecision, player_state: PlayerState) -> bool:
	if decision.type == PlayerDecision.Type.DRAW_CARD:
		return sum(player_state.deck.values()) > 0
	elif decision.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
		return sum(player_state.hand.values()) > 0
	else:
		return False

def game_is_over(game_state: GameState) -> bool:
	return any(not player_can_play(player_state) for player_state in game_state.player_states) or game_state.end_game or game_state.state in END_GAME_STATES

def validate_game_state(game_state: GameState) -> bool:
	player_points = sum([player_state.points for player_state in game_state.player_states])
	total_item_points = sum([item.value * count for item, count in get_default_item_deck().items()])

	if player_points > total_item_points:
		print(f"Player points {player_points} cannot exceed total item points {total_item_points}")
		return False
	
	return True

def get_game_score(game_state: GameState) -> int:
	assert len(game_state.player_states) == 2

	ai_player = game_state.player_states[AI_PLAYER_ID]
	npc_player = game_state.player_states[NPC_PLAYER_ID]

	return ai_player.points - npc_player.points

def print_game_state(game_state: GameState) -> None:
	def hand_state(hand: Dict[CardType, int]) -> str:
		return ' '.join([f'{card_type.name}x{count}' for card_type, count in hand.items()])

	player = game_state.player_states[0]
	print(f"Player 0: P: {player.points:3} D: {sum(player.deck.values()):2} H: {sum(player.hand.values()):2} {hand_state(player.hand):20} {'<-----' if game_state.turn == 0 else '' }")

	player = game_state.player_states[1]
	print(f"Player 1: P: {player.points:3} D: {sum(player.deck.values()):2} H: {sum(player.hand.values()):2} {hand_state(player.hand):20} {'<-----' if game_state.turn == 1 else '' }")

	def queue_state(shop: ShopState) -> str:
		# TODO: Evaluate better
		score = f"{shop.get_player_score(0):2} vs{shop.get_player_score(1):2}"
		cards = ' '.join([f'{queue_item.card_type.name}x{queue_item.count}:{queue_item.player_id}' for queue_item in shop.queue])

		return f"{score} {cards}"

	print(f"Shop 0: {' '.join([item.name for item in game_state.shops[0].items]):8} Q: {queue_state(game_state.shops[0])}")
	print(f"Shop 1: {' '.join([item.name for item in game_state.shops[1].items]):8} Q: {queue_state(game_state.shops[1])}")

def run_player_turn_until(game_state: GameState, player_id: int):
	player_state = game_state.player_states[player_id]

	# def execute_decision(player_decision: PlayerDecision):
	# 	if player_decision.type == PlayerDecision.Type.DRAW_CARD:
	# 		draw_card = draw_from_deck(player_state.deck)
	# 		player_state.hand[draw_card] += 1
	# 	elif player_decision.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
	# 		assert player_decision.count
	# 		assert player_decision.card_type

	# 		player_state.hand[player_decision.card_type] -= player_decision.count
	# 		if player_state.hand[player_decision.card_type] == 0:
	# 			del player_state.hand[player_decision.card_type]
	# 		game_state.shops[player_id].queue.append(QueueItem(player_decision.card_type, player_decision.count, player_id))

	# 	else:
	# 		assert False, "Invalid player decision type"

	if not player_can_play(player_state):
		game_state.end_game = True
		return

	player_decision = player.run_player_decision(game_state, player_id)
	assert player_decision.isValid()
	assert decision_can_be_played(player_decision, player_state)
	execute_decision(player_decision)

	if not player_can_play(player_state):
		game_state.end_game = True
		return

	player_decision = player.run_player_decision(game_state, player_id)
	assert player_decision.isValid()
	assert decision_can_be_played(player_decision, player_state)
	execute_decision(player_decision)

def play_game_until_decision(game_state: GameState) -> None:
	def game_step(game_state: GameState, player_id: int) -> None:
		assert game_state.turn == player_id

		if game_state.state in [GameStep.STATE_START, GameStep.STATE_SHOP_0]:
			run_shop_until(game_state, 0, player_id)
		elif game_state.state == GameStep.STATE_SHOP_1:
			run_shop_until(game_state, 1, player_id)
		elif game_state.state == GameStep.STATE_TURN_0:
			pass
		elif game_state.state == GameStep.STATE_TURN_1:
			pass
		elif game_state.state == GameStep.STATE_TURN_2:
			game_state.state = GameStep.STATE_END_TURN
			if game_is_over(game_state):
				game_state.state = GameStep.STATE_END
				game_state.end_game = True
				return
		elif game_state.state == GameStep.STATE_END:
			pass
		elif game_state.state == GameStep.STATE_ERROR:
			# print("Game state error")
			pass
		elif game_state.state in DECISION_STATES:
			pass
		elif game_state.state == GameStep.STATE_END_TURN:
			pass
		else:
			assert False, f"Invalid game state {game_state.state}"

		if game_state.state == GameStep.STATE_END_TURN:
			if game_state.turn == AI_PLAYER_ID:
				game_state.turn = NPC_PLAYER_ID
			else:
				game_state.turn = AI_PLAYER_ID
			game_state.turn_counter += 1
			game_state.state = GameStep.STATE_SHOP_0

		if game_is_over(game_state):
			game_state.state = GameStep.STATE_END
			game_state.end_game = True
			return

	def can_player_continue(game_state: GameState, player_id: int) -> bool:
		if sum(game_state.player_states[player_id].deck.values()) > 0:
			return True

		if sum(game_state.player_states[player_id].hand.values()) > 0:
			return True
		
		return False

	while True:

		while True and game_state.turn == AI_PLAYER_ID:
			# Run AI player
			game_step(game_state, AI_PLAYER_ID)

			if not can_player_continue(game_state, AI_PLAYER_ID):
				game_state.state = GameStep.STATE_END
				return

			if game_state.state in END_GAME_STATES:
				return

			if game_state.state in DECISION_STATES:
				return

		# Run NPC player
		while True and game_state.turn == NPC_PLAYER_ID:
			game_step(game_state, NPC_PLAYER_ID)

			if not can_player_continue(game_state, NPC_PLAYER_ID):
				game_state.state = GameStep.STATE_END
				return

			if game_state.state in END_GAME_STATES:
				return

			if game_state.state in DECISION_STATES:
				return


def play_game_until_decision_one_player(game_state: GameState, player: Player) -> None:
	while not game_is_over(game_state):
		play_game_until_decision(game_state)

		if game_is_over(game_state):
			return

		if game_state.turn == AI_PLAYER_ID:
			return

		elif game_state.turn == NPC_PLAYER_ID:
			set_decision(game_state, player.run_player_decision(game_state, NPC_PLAYER_ID), NPC_PLAYER_ID)
		else:
			assert False, "Player turn is invalid"

# This is a helper for training the model
def play_game_until_decision_one_player_that_is_not_a_shop_decision(game_state: GameState, npc: Player) -> None:
	while True:
		play_game_until_decision_one_player(game_state, npc)

		if game_is_over(game_state):
			return

		assert game_state.turn == AI_PLAYER_ID, "AI player should play"

		if game_state.state == GameStep.STATE_SHOP_0_DECISION or game_state.state == GameStep.STATE_SHOP_1_DECISION:
			set_decision(game_state, npc.run_player_decision(game_state, AI_PLAYER_ID), AI_PLAYER_ID)
			continue

		if game_state.state == GameStep.STATE_TURN_0 or game_state.state == GameStep.STATE_TURN_1: # TODO: Implement or game_state.state == GameStep.STATE_TURN_2:
			# And we cannot do anything other than draw a card
			# This is so that we make model training easier
			if sum(game_state.player_states[AI_PLAYER_ID].hand.values()) == 0:
				set_decision(game_state, PlayerDecision(PlayerDecision.Type.DRAW_CARD), AI_PLAYER_ID)
				continue
			return

		return

def play_game(game_state: GameState, player0: Player, player_1: Player, verbose: bool = False, skip_shop_decisions: bool = False) -> None:
	def run_shop_decision(game_state: GameState) -> None:
		decision = RandomPlayer().run_player_decision(game_state, game_state.turn)
		set_decision(game_state, decision, game_state.turn)

	while not game_is_over(game_state):
		play_game_until_decision(game_state)

		if verbose:
			print_game_state(game_state)

		if skip_shop_decisions and game_state.state in [GameStep.STATE_SHOP_0_DECISION, GameStep.STATE_SHOP_1_DECISION]:
			run_shop_decision(game_state)
		else:
			if game_state.turn == AI_PLAYER_ID:
				decision = player0.run_player_decision(game_state, AI_PLAYER_ID)
				set_decision(game_state, decision, AI_PLAYER_ID)
			elif game_state.turn == NPC_PLAYER_ID:
				decision = player_1.run_player_decision(game_state, NPC_PLAYER_ID)
				set_decision(game_state, decision, NPC_PLAYER_ID)
			else:
				assert False, "Player turn is invalid"

		if verbose:
			print()
			print(f"Decision: {decision} - Player {game_state.turn}")
			print()

def set_decision(game_state: GameState, decision: Optional[PlayerDecision], player_id: int) -> None:
	if game_state.state == GameStep.STATE_START:
		pass

	elif game_state.state == GameStep.STATE_SHOP_0_DECISION:
		if decision.type != PlayerDecision.Type.SHOP_DECISION:
			game_state.state = GameStep.STATE_ERROR
			return

		shop = game_state.shops[0]

		shop.place_item_in_limbo(decision.item_id, player_id)
		
		game_state.state = GameStep.STATE_SHOP_1
		return

	elif game_state.state == GameStep.STATE_SHOP_1_DECISION:
		if decision.type != PlayerDecision.Type.SHOP_DECISION:
			game_state.state = GameStep.STATE_ERROR
			return

		shop = game_state.shops[1]

		shop.place_item_in_limbo(decision.item_id, player_id)
		
		game_state.state = GameStep.STATE_TURN_0
		return

	elif game_state.state == GameStep.STATE_TURN_0 or game_state.state == GameStep.STATE_TURN_1:
		player_state = game_state.player_states[player_id]

		if decision.type == PlayerDecision.Type.DRAW_CARD:
			if sum(player_state.deck.values()) == 0:
				game_state.state = GameStep.STATE_ERROR
				return

			draw_card = draw_from_deck(player_state.deck)
			player_state.hand[draw_card] += 1

		elif decision.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			if player_state.hand[decision.card_type] < decision.count:
				game_state.state = GameStep.STATE_ERROR
				return

			player_state.hand[decision.card_type] -= decision.count

			if player_state.hand[decision.card_type] == 0:
				del player_state.hand[decision.card_type]

			game_state.shops[decision.queue_id].queue.append(QueueItem(decision.card_type, decision.count, player_id))
		else:
			game_state.state = GameStep.STATE_ERROR
			return

		if game_state.state == GameStep.STATE_TURN_0:
			game_state.state = GameStep.STATE_TURN_1
		elif game_state.state == GameStep.STATE_TURN_1:

		# TODO: Implement GameStep.STATE_TURN_2
		# 	game_state.state = GameStep.STATE_TURN_2
		# elif game_state.state == GameStep.STATE_TURN_2:

			game_state.state = GameStep.STATE_END_TURN

			if game_is_over(game_state):
				game_state.state = GameStep.STATE_END
				game_state.end_game = True
				return

			# # Turn change logic
			# if game_state.turn == AI_PLAYER_ID:
			# 	game_state.turn = NPC_PLAYER_ID
			# else:
			# 	game_state.turn = AI_PLAYER_ID
			# game_state.turn_counter += 1
			# game_state.state = GameStep.STATE_SHOP_0

		else:
			assert False, "Invalid game state"

		return

	elif game_state.state == GameStep.STATE_END:
		return
	elif game_state.state == GameStep.STATE_ERROR:
		return
	else:
		assert False, f"Invalid game state {game_state.state}"

def get_legal_moves(game_state: GameState, player_id: int) -> npt.NDArray[np.float32]:
	actions = np.zeros(1 + len(CARD_INFO), dtype=np.float32)

	if game_state.state == GameStep.STATE_TURN_0 or game_state.state == GameStep.STATE_TURN_1:
		# Can we draw a card?
		actions[0] = sum(game_state.player_states[player_id].deck.values()) > 0

		# Can we place a card in the queue?
		for i, (card_type, _) in enumerate(CARD_INFO):
			actions[i+1] = game_state.player_states[player_id].hand.get(card_type, 0) > 0

	# TODO: Implement GameStep.STATE_TURN_2
	# TODO: Implement shop decision

	return actions

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

import time
def timeit(func: Callable):
	def timed(*args, **kwargs):
		print(f"Timing function {func.__name__}")
		ts = time.time()
		for i in range(1):
			result = func(*args, **kwargs)
		te = time.time()
		print(f"Function {func.__name__} took {(te-ts)/1000} seconds")
		return result
	return timed

if __name__ == "__main__":

	score = 0

	for i in tqdm.tqdm(range(1000)):
		game = initialize_game_state()
		play_game(game, RandomPlayer(), RandomPlayer())

		assert game_is_over(game)

		score += get_game_score(game)

	print(f"Average score: {score / 1000}")


STATE_SIZE = len(initialize_game_state().to_state_array(0))
ACTION_SIZE = len(PlayerDecision(0).to_state_array())