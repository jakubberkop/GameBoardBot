import random
import pickle

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Type

import tqdm

import numpy as np
import numpy.typing as npt

# from pytracy import *
# set_tracing_mode(TracingMode.All)

PRIVATE_STATE = True

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
		elif self.name == "MD":
			return 2
		elif self.name == "PTN":
			return 1
		elif self.name == "SY":
			return 0
		else:
			# "PSN" and "LK" values don't matter as they are counted differently
			assert False, "Invalid card type"


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

	STATE_SIZE = 2 + len(CARD_INFO)
	PRIVATE_SIZE = 2

	def to_state_array_fast(self, array: npt.NDArray[np.float32], index: int, add_private_data: bool = True) -> None:
		array[index] = self.points
		array[index+1] = sum(self.deck.values())

		if not add_private_data:
			return

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
	STATE_SIZE: int = 3 + 2 * len(CARD_INFO) 

	def get_player_score(self, player_id: int) -> int:
		score = 0

		# LK changes counting completely
		if any(queue_item.card_type.name == "LK" for queue_item in self.queue):
			# LK counting rules
			score = sum(queue_item.count for queue_item in self.queue if queue_item.player_id == player_id)
		else:
			# Normal counting rules

			# Sum of all non PSN cards
			score = sum(queue_item.card_type.value * queue_item.count for queue_item in self.queue if queue_item.player_id == player_id and queue_item.card_type.name != "PSN")

			# Count number of PSN.
			psn_count = sum(queue_item.count for queue_item in self.queue if queue_item.card_type.name == "PSN" and queue_item.player_id == player_id)
			# Count score of PSN given by the (N * (N+1)) / 2 formula
			score += (psn_count * (psn_count + 1)) // 2

		if self.limbo_item and self.limbo_item.player_id == player_id:
			score -= self.limbo_item.item.value

		return score

	def get_item_count(self) -> int:
		return len(self.items)
	
	def place_item_in_limbo(self, item_id: int, player_id: int) -> None:
		self.limbo_item = LimboItem(self.items[item_id], player_id)
		self.items.pop(item_id)

	def to_state_array_fast(self, state: npt.NDArray[np.float32], index: int, player_id: int) -> None:
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
			assert False, "Invalid shop state"

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

	def __str__(self):
		item_str = " ".join([item.name for item in self.items])
		queue_str = " ".join([f"{queue_item.card_type.name}x{queue_item.count}" for queue_item in self.queue])
		return f"Items: {item_str} Queue: {queue_str} Limbo: {self.limbo_item}"

class PlayerDecision:
	class Type(IntEnum):
		DRAW_CARD = 0
		PLACE_CARD_IN_QUEUE = 1
		SHOP_DECISION = 2
		SKIP_2_TURN = 3
		PTN_REPLACE_IN_QUEUE = 4
		SY_REPLACE_IN_QUEUE = 5

	type: Type

	# PLACE_CARD_IN_QUEUE
	card_type: Optional[CardType] = None
	count: Optional[int] = None
	queue_id: Optional[int] = None

	# SHOP_DECISION
	item_id: Optional[int] = None

	# PTN_REPLACE_IN_QUEUE
	player_id: Optional[int] = None

	# TODO: Implement SY
	def __init__(self, type: int, card_type: Optional[CardType] = None, count: Optional[int] = None, queue_id: Optional[int] = None, item_id: Optional[int] = None, player_id: Optional[int] = None):
		self.type = PlayerDecision.Type(type)
		self.card_type = card_type
		self.count = count
		self.item_id = item_id
		self.queue_id = queue_id
		self.player_id = player_id

	CARD_OFFSET = 21

	# Encoded Player Decision:
	# 
	# Length          Type
	#  1: Draw card
	# 21: Place card in queue 0
	#   - 7: 1
	#   - 3: 2
	#   - 1: 3
	#   - 1: LK
	#   - 2: MD
	#   - 5: PSN
	#   - 1: PTN
	#   - 1: SY
	# 21: Place card in queue 1
	#   - 7: 1
	#   - 3: 2
	#   - 1: 3
	#   - 1: LK
	#   - 2: MD
	#   - 5: PSN
	#   - 1: PTN
	#   - 17: 1
	#   - 3: 2
	#   - 1: 3
	#   - 1: LK
	#   - 2: MD
	#   - 5: PSN
	#   - 1: PTN
	#   - 1: SY
	#  2: Shop decision
	#   - 0: First item
	#   - 1: Second item
	#  1: End turn (only when in turn 2, MD only turn)
	# 32: PTN replaces a card in each queue for each player
	#   - Q0:
	#    - P0:
	#       - 1
	#       - 2
	#       - 3
	#       - LK
	#       - MD
	#       - PSN
	#       - PTN
	#       - SY
	#     - P1:
	#    - Q1: 
	#     - P0:
	#     - P1:
	# 7: SY replaces itself with a card from another queue
	def _encode_action(self) -> int:
		if self.type == PlayerDecision.Type.DRAW_CARD:
			return 0

		elif self.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			assert self.queue_id is not None
			assert self.count is not None

			index = PlayerDecision.CARD_OFFSET * self.queue_id

			for card_type, card_count in CARD_INFO:
				if card_type == self.card_type:
					break

				index += card_count

			index += self.count - 1
			return index + 1

		elif self.type == PlayerDecision.Type.SHOP_DECISION:
			assert self.item_id is not None
			return 1 + 2 * PlayerDecision.CARD_OFFSET + self.item_id
		elif self.type == PlayerDecision.Type.SKIP_2_TURN:
			return 1 + 2 * PlayerDecision.CARD_OFFSET + 2
		elif self.type == PlayerDecision.Type.PTN_REPLACE_IN_QUEUE:
			assert self.card_type is not None
			assert self.queue_id is not None
			assert self.player_id is not None

			offset = 1 + 2 * PlayerDecision.CARD_OFFSET + 2 + 1
			ptn_decision_offset = (self.queue_id * 2 * len(CARD_INFO)) + (self.player_id * len(CARD_INFO)) + self.card_type.id

			return offset + ptn_decision_offset
		elif self.type == PlayerDecision.Type.SY_REPLACE_IN_QUEUE:
			assert self.card_type is not None
			offset = 1 + 2 * PlayerDecision.CARD_OFFSET + 2 + 1 + len(CARD_INFO) * 2 * 2

			for card_type, card_count in CARD_INFO:
				if card_type == self.card_type:
					break
					
				if card_type.name == "SY":
					continue # Skip SY

				offset += 1

			return offset	
		else:
			assert False, "Invalid player decision type"

	def encode_action(self) -> int:
		action = self._encode_action()
		assert action >= 0 and action < self.state_space_size(), f"Invalid action {action} {self.state_space_size()}"
		return action

	@staticmethod
	def state_space_size() -> int:
		return 1 + 2 * PlayerDecision.CARD_OFFSET + 2 + 1 + len(CARD_INFO) * 2 * 2 + (len(CARD_INFO) - 1)

	@staticmethod
	def from_encoded_action(encoded_action: int) -> "PlayerDecision":
		assert encoded_action >= 0 and encoded_action < PlayerDecision.state_space_size()

		if encoded_action == 0:
			return PlayerDecision(PlayerDecision.Type.DRAW_CARD)
		
		encoded_action -= 1

		if encoded_action < PlayerDecision.CARD_OFFSET * 2:
			if encoded_action < PlayerDecision.CARD_OFFSET:
				queue_id = 0
			else:
				queue_id = 1

			encoded_action %= PlayerDecision.CARD_OFFSET

			for card_type, card_count in CARD_INFO:
				if encoded_action < card_count:
					return PlayerDecision(PlayerDecision.Type.PLACE_CARD_IN_QUEUE, card_type=card_type, count=encoded_action + 1, queue_id=queue_id)
				encoded_action -= card_count

		encoded_action -= PlayerDecision.CARD_OFFSET * 2

		if encoded_action == 0:
			return PlayerDecision(PlayerDecision.Type.SHOP_DECISION, item_id=0)
		encoded_action -= 1

		if encoded_action == 0:
			return PlayerDecision(PlayerDecision.Type.SHOP_DECISION, item_id=1)
		encoded_action -= 1

		if encoded_action == 0:
			return PlayerDecision(PlayerDecision.Type.SKIP_2_TURN)
		encoded_action -= 1

		if encoded_action < len(CARD_INFO) * 2 * 2:
			card_id = encoded_action % len(CARD_INFO)
			encoded_action //= len(CARD_INFO)

			player_id = encoded_action % 2
			queue_id = encoded_action // 2

			assert card_id >= 0 and card_id < len(CARD_INFO)
			assert queue_id == 0 or queue_id == 1
			assert player_id == 0 or player_id == 1

			return PlayerDecision(PlayerDecision.Type.PTN_REPLACE_IN_QUEUE, card_type=CARD_INFO[card_id][0], queue_id=queue_id, player_id=player_id)

		encoded_action -= len(CARD_INFO) * 2 * 2

		assert encoded_action >= 0 and encoded_action < len(CARD_INFO)
		card_type = CARD_INFO[encoded_action][0]

		return PlayerDecision(PlayerDecision.Type.SY_REPLACE_IN_QUEUE, card_type=card_type)

	def __eq__(self, other: Any):
		if self.type != other.type:
			return False
		
		if self.type == PlayerDecision.Type.DRAW_CARD:
			return True
		
		if self.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			return self.card_type == other.card_type and self.count == other.count
		
		if self.type == PlayerDecision.Type.SHOP_DECISION:
			return self.item_id == other.item_id and self.queue_id == other.queue_id
		
		if self.type == PlayerDecision.Type.SKIP_2_TURN:
			return True
		
		if self.type == PlayerDecision.Type.PTN_REPLACE_IN_QUEUE:
			return self.card_type == other.card_type and self.queue_id == other.queue_id and self.player_id == other.player_id

		if self.type == PlayerDecision.Type.SY_REPLACE_IN_QUEUE:
			return self.card_type == other.card_type

		assert False

	def __str__(self):
		if self.type == PlayerDecision.Type.DRAW_CARD:
			return f"Draw card"
		elif self.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			return f"Place {self.card_type.name}x{self.count} in queue {self.queue_id}"
		elif self.type == PlayerDecision.Type.SHOP_DECISION:
			return f"Place item {self.item_id} in shop"
		elif self.type == PlayerDecision.Type.SKIP_2_TURN:
			return f"Skip 2 turn"
		elif self.type == PlayerDecision.Type.PTN_REPLACE_IN_QUEUE:
			assert self.card_type is not None
			assert self.queue_id is not None
			assert self.player_id is not None
			return f"PTN replace {self.card_type.name} in queue {self.queue_id} for player {self.player_id}"
		elif self.type == PlayerDecision.Type.SY_REPLACE_IN_QUEUE:
			assert self.card_type is not None
			return f"SY replace {self.card_type.name} in queue"
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
	GameStep.STATE_TURN_0, GameStep.STATE_TURN_1, GameStep.STATE_TURN_2
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
	_state: GameStep = GameStep.STATE_START
	turn: int = 0
	turn_counter: int = 0

	@property
	def state(self) -> GameStep:
		return self._state
	
	@state.setter
	def state(self, value: GameStep) -> None:
		assert GameStep(value) != GameStep.STATE_ERROR
		# import sys
		# import traceback
		# print(f"State: {GameStep(self._state)} -> {value}", file=sys.stderr)
		# traceback.print_stack()
		self._state = value

	def to_state_array(self, player_id: int) -> npt.NDArray[np.float32]:
		state_size = len(GameStep) + 2 * ShopState.STATE_SIZE + PlayerDecision.state_space_size()

		if PRIVATE_STATE:
			state_size += PlayerState.STATE_SIZE
			state_size += PlayerState.PRIVATE_SIZE * (len(self.player_states) - 1)
		else:
			state_size += PlayerState.STATE_SIZE * len(self.player_states)

		state = np.zeros(state_size, dtype=np.float32)
		
		# One hot encoding of the state
		state[int(self.state)] = 1
		index = len(GameStep)

		# Player states
		for i, player_state in enumerate(self.player_states):
			private_data = i == player_id or not PRIVATE_STATE
			player_state.to_state_array_fast(state, index, private_data)

			if private_data:
				index += PlayerState.PRIVATE_SIZE
			else:
				index += PlayerState.STATE_SIZE

		# Shop states
		for shop in self.shops:
			shop.to_state_array_fast(state, index, player_id)
			index += ShopState.STATE_SIZE

		# Possible actions
		for e in get_legal_moves(self, player_id):
			state[index] = e
			index += 1

		assert index == len(state), f"State should be fully filled {index} != {len(state)}"

		return state

	def __str__(self):
		return f""" {self.state}

Player 0: {self.player_states[0]}
Player 1: {self.player_states[1]}

Shop 0: {self.shops[0]}
Shop 1: {self.shops[1]}
"""

class Player:

	T = Type["Player"]

	def name(self) -> str:
		assert False, "Not implemented"

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
	assert game_state.state in [GameStep.STATE_SHOP_0, GameStep.STATE_SHOP_1]
	
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

			if shop_id == 0:
				game_state.state = GameStep.STATE_SHOP_1
			else:
				game_state.state = GameStep.STATE_TURN_0
		else:
			# items_to_draw == 1 is not possible, as we number of items in deck is always even
			assert False, "Invalid game state"
	else:
		assert False

def player_can_play(player_state: PlayerState) -> bool:
	return sum(player_state.deck.values()) > 0 or sum(player_state.hand.values()) > 0

def game_is_over(game_state: GameState) -> bool:
	return any(not player_can_play(player_state) for player_state in game_state.player_states) or game_state.end_game or game_state.state in END_GAME_STATES

def get_game_score(game_state: GameState) -> int:
	assert len(game_state.player_states) == 2

	ai_player = game_state.player_states[AI_PLAYER_ID]
	npc_player = game_state.player_states[NPC_PLAYER_ID]

	return ai_player.points - npc_player.points

def print_game_state(game_state: GameState) -> None:
	print()

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

def play_game_until_decision(game_state: GameState) -> None:
	def game_step(game_state: GameState, player_id: int) -> None:
		assert game_state.turn == player_id

		if game_state.state == GameStep.STATE_START:
			game_state.state = GameStep.STATE_SHOP_0

		if game_state.state == GameStep.STATE_SHOP_0:
			run_shop_until(game_state, 0, player_id)
		elif game_state.state == GameStep.STATE_SHOP_1:
			run_shop_until(game_state, 1, player_id)
		elif game_state.state == GameStep.STATE_TURN_0:
			pass
		elif game_state.state == GameStep.STATE_TURN_1:
			pass
		elif game_state.state == GameStep.STATE_TURN_2:
			pass
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

		return

@dataclass
class GameRun:
	move_and_states: List[GameState | PlayerDecision] = field(default_factory=list)

	def to_file(self, file_path: str) -> None:
		with open(file_path, "wb") as f:
			pickle.dump(self, f)

	@staticmethod
	def from_file(file_path: str) -> "GameRun":
		with open(file_path, "rb") as f:
			return pickle.load(f)


def play_game(game_state: GameState, player0: Player, player_1: Player, verbose: bool = False):

	game_run = GameRun()
	game_run.move_and_states.append(game_state)

	try:
		while not game_is_over(game_state):
			play_game_until_decision(game_state)
			game_run.move_and_states.append(game_state)

			if verbose:
				print_game_state(game_state)

			if game_is_over(game_state):
				return
			if game_state.turn == AI_PLAYER_ID:
				decision = player0.run_player_decision(game_state, AI_PLAYER_ID)
				game_run.move_and_states.append(decision)
				set_decision(game_state, decision, AI_PLAYER_ID)
			elif game_state.turn == NPC_PLAYER_ID:
				decision = player_1.run_player_decision(game_state, NPC_PLAYER_ID)
				game_run.move_and_states.append(decision)
				set_decision(game_state, decision, NPC_PLAYER_ID)
			else:
				assert False, "Player turn is invalid"

			# TODO: Fix printing
			if verbose:
				print()
				print(f"Decision: {decision} - Player {game_state.turn}")
				print()
	except Exception as e:
		game_run.to_file("last_game_run.pkl")
		raise e

def queue_id_with_sy_card(game_state: GameState, player_id: int) -> Optional[int]:
	for queue_id, shop in enumerate(game_state.shops):
		if any(queue_item.card_type == CardType("SY", 8) and queue_item.player_id == player_id for queue_item in shop.queue):
			return queue_id

	return None

def set_decision(game_state: GameState, decision: Optional[PlayerDecision], player_id: int) -> None:
	assert decision is not None

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

	elif game_state.state in [GameStep.STATE_TURN_0, GameStep.STATE_TURN_1, GameStep.STATE_TURN_2]:
		md_only_state = game_state.state == GameStep.STATE_TURN_2
		player_state = game_state.player_states[player_id]

		if decision.type == PlayerDecision.Type.DRAW_CARD:
			assert not md_only_state, "MD card can't be drawn in this turn"

			if sum(player_state.deck.values()) == 0:
				game_state.state = GameStep.STATE_ERROR
				return

			draw_card = draw_from_deck(player_state.deck)
			player_state.hand[draw_card] += 1

		elif decision.type == PlayerDecision.Type.PLACE_CARD_IN_QUEUE:
			if md_only_state and decision.card_type.name != "MD":
				game_state.state = GameStep.STATE_ERROR
				return

			if player_state.hand[decision.card_type] < decision.count:
				game_state.state = GameStep.STATE_ERROR
				return

			player_state.hand[decision.card_type] -= decision.count

			if player_state.hand[decision.card_type] == 0:
				del player_state.hand[decision.card_type]

			game_state.shops[decision.queue_id].queue.append(QueueItem(decision.card_type, decision.count, player_id))
		
		elif decision.type == PlayerDecision.Type.SKIP_2_TURN:
			# When the player skips the turn, the game state should be set to the next turn
			# as it player has played the MD card
			pass
		
		elif decision.type == PlayerDecision.Type.PTN_REPLACE_IN_QUEUE:
			if player_state.hand.get(CardType("PTN", 6), 0) == 0:
				game_state.state = GameStep.STATE_ERROR
				return

			assert decision.queue_id is not None

			queue = game_state.shops[decision.queue_id].queue

			# Find the card in the queue
			for i, queue_item in enumerate(queue):
				if queue_item.card_type == decision.card_type and queue_item.player_id == decision.player_id:
					queue[i] = QueueItem(CardType("PTN", 6), 1, player_id)

					player_state.hand[CardType("PTN", 6)] -= 1
					if player_state.hand[CardType("PTN", 6)] == 0:
						del player_state.hand[CardType("PTN", 6)]

					break
			else:
				game_state.state = GameStep.STATE_ERROR
				return

		elif decision.type == PlayerDecision.Type.SY_REPLACE_IN_QUEUE:
			queue_id = queue_id_with_sy_card(game_state, player_id)

			assert queue_id is not None

			other_queue_id = 1 - queue_id

			sy_queue = game_state.shops[queue_id].queue
			other_queue = game_state.shops[other_queue_id].queue

			# Find the card in the queue
			for i, queue_item in enumerate(sy_queue):
				if queue_item.card_type == CardType("SY", 8) and queue_item.player_id == player_id:
					sy_queue[i] = QueueItem(decision.card_type, 1, player_id)
					break

			# Remove the card from the other queue
			for i, queue_item in enumerate(other_queue):
				if queue_item.card_type == decision.card_type and queue_item.player_id == player_id:
					del other_queue[i]
					break
		else:
			assert False, "Invalid decision"


		# Changing of the turns after the decision

		if game_state.state == GameStep.STATE_TURN_0:
			game_state.state = GameStep.STATE_TURN_1
		elif game_state.state == GameStep.STATE_TURN_1:
			player_has_md_card = player_state.hand.get(CardType("MD", 4), 0) > 0

			if player_has_md_card:
				game_state.state = GameStep.STATE_TURN_2
			else:
				game_state.state = GameStep.STATE_END_TURN

		elif game_state.state == GameStep.STATE_TURN_2:
			game_state.state = GameStep.STATE_END_TURN

		else:
			assert False, "Invalid game state"

		# DO NOT REMOVE, as this is load bearing code
		# Game end conditions are not verified to be correct
		if game_is_over(game_state):
			game_state.state = GameStep.STATE_END
			game_state.end_game = True
			return

		return

	elif game_state.state == GameStep.STATE_END:
		return
	elif game_state.state == GameStep.STATE_ERROR:
		return
	else:
		assert False, f"Invalid game state {game_state.state}"

def get_legal_moves(game_state: GameState, player_id: int) -> npt.NDArray[np.float32]:
	actions = np.zeros(PlayerDecision.state_space_size(), dtype=np.float32)

	if game_state.state in [GameStep.STATE_TURN_0, GameStep.STATE_TURN_1, GameStep.STATE_TURN_2]:
		# Only MD card can be played in the third turn
		md_only_turn = game_state.state == GameStep.STATE_TURN_2
		player_state = game_state.player_states[player_id]

		# Can we draw a card?
		index = 0

		if not md_only_turn:
			actions[index] = sum(player_state.deck.values()) > 0

		index += 1

		# Can we place a card in the queue #2
		for card_type, card_count in CARD_INFO:
			for j in range(1, card_count + 1):
				# If we are in md_only_turn, we can only place MD cards.
				# In all other cases, we can place any card
				can_place_card = (md_only_turn and card_type.name == "MD") or not md_only_turn
				if can_place_card:
					actions[index] = player_state.hand.get(card_type, 0) >= j

				index += 1

		# Can we place a card in the queue #1 
		for card_type, card_count in CARD_INFO:
			for j in range(1, card_count + 1):
				# If we are in md_only_turn, we can only place MD cards.
				# In all other cases, we can place any card
				can_place_card = (md_only_turn and card_type.name == "MD") or not md_only_turn
				if can_place_card:
					actions[index] = player_state.hand.get(card_type, 0) >= j

				index += 1

		index += 2 # skip shop decisions

		# Can we skip the turn - are we in the second turn <=> have MD card
		actions[index] = md_only_turn
		index += 1
	
		player_has_ptn_card = player_state.hand.get(CardType("PTN", 6), 0) > 0


		def card_in_queue(queue: List[QueueItem], card_type: CardType, player_id: int) -> bool:
			return any(queue_item.card_type == card_type and queue_item.player_id == player_id for queue_item in queue)

		# Can we replace a card in each queue with PTN card
		if player_has_ptn_card and not md_only_turn:
			for queue_id in range(2):
				for ptn_player_id in range(2):
					for card_type, card_count in CARD_INFO:
						actions[index] = card_in_queue(game_state.shops[queue_id].queue, card_type, ptn_player_id)
						index += 1
		else:
			index += len(CARD_INFO) * 2 * 2

		queue_id = queue_id_with_sy_card(game_state, player_id)

		if queue_id is not None and not md_only_turn:
			# Can we replace a card in the queue with SY card
			other_queue_id = 1 - queue_id

			for card_type, card_count in CARD_INFO:
				if card_type.name == "SY":
					continue

				actions[index] = card_in_queue(game_state.shops[other_queue_id].queue, card_type, player_id)
				index += 1
		else:
			index += len(CARD_INFO) - 1


	elif game_state.state == GameStep.STATE_SHOP_0_DECISION or game_state.state == GameStep.STATE_SHOP_1_DECISION:
		index = 1 + 2 * PlayerDecision.CARD_OFFSET

		shop = game_state.shops[0] if game_state.state == GameStep.STATE_SHOP_0_DECISION else game_state.shops[1]
		# We don't have to check whether current player is winning in the shop, as the player can only make a decision if they are winning
		# But the assert is here regardless, to make sure the logic is correct
		assert shop.get_player_score(player_id) > shop.get_player_score(1 - player_id)

		if shop.get_item_count() == 2:
			actions[index + 0] = 1
			actions[index + 1] = 1
		else:
			assert False, "Invalid shop state for decision"

	elif game_state.state in END_GAME_STATES or game_state.state == GameStep.STATE_START:
		pass # No legal moves when game is over or not started
	else:
		assert False, f"Legal moves not implemented for {game_state.state.name}"

	return actions

class RandomPlayer(Player):

	def name(self) -> str:
		return "RandomPlayer"

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		legal_actions = get_legal_moves(game_state, player_id)
		legal_indices = np.argwhere(legal_actions != 0).flatten()

		decision = PlayerDecision.from_encoded_action(random.choice(legal_indices))

		assert decision is not None
		return decision

class AlwaysFirstPlayer(Player):

	def name(self) -> str:
		return "AlwaysFirstPlayer"

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		legal_moves = get_legal_moves(game_state, player_id)
		first_legal = np.argwhere(legal_moves != 0)[-1][0]
		decision = PlayerDecision.from_encoded_action(first_legal)

		assert decision is not None
		return decision

class AlwaysLastPlayer(Player):

	def name(self) -> str:
		return "AlwaysLastPlayer"

	def run_player_decision(self, game_state: GameState, player_id: int) -> PlayerDecision:
		legal_moves = get_legal_moves(game_state, player_id)
		last_legal = np.argwhere(legal_moves != 0)[-1][0]
		decision = PlayerDecision.from_encoded_action(last_legal)

		assert decision is not None
		return decision


def test_decision_encoding_decoding() -> None:
	for i in range(PlayerDecision.state_space_size()):
		decision = PlayerDecision.from_encoded_action(i)
		assert decision.encode_action() == i, f"Decoding and encoding failed for {decision}"
		assert PlayerDecision.from_encoded_action(decision.encode_action()) == decision

def run_last_move():
	game_run = GameRun.from_file("last_game_run.pkl")
	
	last_decision_index = [i for i, e in enumerate(game_run.move_and_states) if isinstance(e, PlayerDecision)][-1]
	last_state_index = [i for i, e in enumerate(game_run.move_and_states) if isinstance(e, GameState)][-1]

	assert abs(last_decision_index - last_state_index) == 1, f"Last decision and last state are not adjacent {last_decision_index} {last_state_index}"

	last_game_state = game_run.move_and_states[last_state_index]
	last_decision = game_run.move_and_states[last_decision_index]

	assert isinstance(last_game_state, GameState)
	assert isinstance(last_decision, PlayerDecision)

	print_game_state(last_game_state)
	print(f"Decision: {last_decision} - Player {last_game_state.turn}")

	print("Legal moves:")
	legal_moves = get_legal_moves(last_game_state, last_game_state.turn)

	for i, e in enumerate(legal_moves):
		if e == 1:
			print(PlayerDecision.from_encoded_action(i))

	set_decision(last_game_state, last_decision, last_game_state.turn)




if __name__ == "__main__":

	# run_last_move()

	# score = 0
	test_decision_encoding_decoding()

	for i in tqdm.tqdm(range(1_000)):
		game = initialize_game_state()
		play_game(game, RandomPlayer(), RandomPlayer(), verbose=False)

	# assert game_is_over(game)

	# score += get_game_score(game)

	# print(f"Average score: {score / 1000}")

	print(PlayerDecision.state_space_size())

# STATE_SIZE = len(initialize_game_state().to_state_array(0))