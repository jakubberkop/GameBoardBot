from sb3_contrib import MaskablePPO
import evaluate
from game import AlwaysFirstPlayer, AlwaysLastPlayer, RandomPlayer, initialize_game_state, play_game
from game_env import GameEnv

if __name__ == "__main__":
    evaluate.main(10)

    game = initialize_game_state()
    play_game(game, RandomPlayer(), RandomPlayer(), verbose=True)

    oponent_types = [RandomPlayer, AlwaysFirstPlayer, AlwaysLastPlayer]
    env = GameEnv(oponent_types)

    model = MaskablePPO("MlpPolicy", env, n_steps=64)
    model.learn(200)