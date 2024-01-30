from typing import List
import datasets
import numpy as np
import tqdm

from game import AI_PLAYER_ID, GameStep, RandomPlayer, game_is_over, initialize_game_state, play_game_until_decision_one_player_that_is_not_a_shop_decision, set_decision

from dataclasses import dataclass

import pytracy
# pytracy.set_tracing_mode(pytracy.TracingMode.All)

@dataclass
class DataPoint:
	state: np.ndarray
	action: np.ndarray
	reward: int
	done: bool

def generate_one_game() -> List[DataPoint]:
	data: List[DataPoint] = []

	game_state = initialize_game_state()

	random_player = RandomPlayer()

	for _ in range(32):
		play_game_until_decision_one_player_that_is_not_a_shop_decision(game_state, random_player)

		if game_is_over(game_state) or game_state.state == GameStep.STATE_END:
			data_point = DataPoint(state, action.to_state_array_fast(), reward, True)
			data.append(data_point)
			continue

		state = game_state.to_state_array(AI_PLAYER_ID)

		if game_state.state == GameStep.STATE_END_TURN:
			play_game_until_decision_one_player_that_is_not_a_shop_decision(game_state, random_player)

		action = random_player.run_player_decision(game_state, AI_PLAYER_ID)
		set_decision(game_state, action, AI_PLAYER_ID)

		reward = game_state.get_reward(AI_PLAYER_ID)

		data_point = DataPoint(state, action.to_state_array_fast(), reward, False)
		data.append(data_point)
	
	return data



def generate_dataset() -> datasets.Dataset:

	features = {
		"states": datasets.Sequence(datasets.Array2D(shape=(17, 32), dtype="float32")),
		"actions": datasets.Sequence(datasets.Array2D(shape=(6, 32), dtype="float32")),
		"rewards": datasets.Sequence(datasets.Array2D(shape=(1, 32), dtype="int32")),
		"dones": datasets.Sequence(datasets.Array2D(shape=(1, 32), dtype="int32")),
	}

	dict = {
		"states": [],
		"actions": [],
		"rewards": [],
		"dones": [],
	}

	for _ in tqdm.tqdm(range(100_000)):
		game_data = generate_one_game()

		dict["states"].append([x.state for x in game_data])
		dict["actions"].append([x.action for x in game_data])
		dict["rewards"].append([x.reward for x in game_data])
		dict["dones"].append([x.done for x in game_data])


		# >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
        # >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        # >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
        # >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
        # >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        # >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

	ds = datasets.Dataset.from_dict(dict) # , features=features)

	return ds

if __name__ == "__main__":
	ds = generate_dataset()
	ds.save_to_disk("data/dataset3big")



# ds = datasets.load_from_disk("data/dataset1")
	
# # # rename columns
# ds = ds.rename_column("state", "states")
# ds = ds.rename_column("action", "actions")
# ds = ds.rename_column("reward", "rewards")

# ds.save_to_disk("data/dataset1f")

# # ds = datasets.load_from_disk("data/dataset2_F")

# # # Change types to float32
# # ds = ds.cast(datasets.Features({"states": datasets.Value("float32"), "actions": datasets.Value("float32")}))

# # ds.save_to_disk("data/dataset2_T")