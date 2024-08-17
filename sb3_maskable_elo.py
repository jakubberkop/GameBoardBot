import argparse
import datetime
import itertools
import os
import random

from dataclasses import dataclass
from typing import List, Optional

from matplotlib.pylab import f
import tqdm

import numpy as np
import elo as e

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import stable_baselines3.common.vec_env as sb3_vec_env

from game import GameRestult, Player, RandomPlayer, AlwaysFirstPlayer, AlwaysLastPlayer, play_single_game
from game_env import GameEnv

from evaluate import PPOPlayer

# import pytracy
# pytracy.set_tracing_mode(pytracy.TracingMode.All)
# pytracy.add_path_to_filter("/")

K = 16 # Based on expert knowledge of a professional board game player
ENV_VEC_SIZE = 100
N_STEP = 64
GAMMA = 0.8
DEVICE = "cuda"
ITERATIONS = 100_000
META_LEARNING_ITERATIONS = 100
LOG_INTERVAL = 20
USE_EXITING_MODELS = True

ELO_GAME_COUNT = 50
GAME_COUNT_PER_LEAGUE = 500
GAME_COUNT_PER_LEVEL = GAME_COUNT_PER_LEAGUE // ELO_GAME_COUNT


MODEL_PATH = "ppo_elo"
TENSORBOARD_PATH = "runs"


@dataclass
class LeagueEntry:
    player: Player
    model: Optional[MaskablePPO]
    rating: e.Rating

class EloLearnig:
    elo: e.Elo
    league: List[LeagueEntry]

    def __init__(self, k: int):
        self.elo = e.Elo(k)
        self.initialize_league()

    def initialize_league(self):
        self.league = []

        for player in [RandomPlayer, AlwaysFirstPlayer, AlwaysLastPlayer]:
            self.league.append(LeagueEntry(player(), None, e.Rating()))

    def load_pretrained_models(self):
        import load_all
        for model, model_name in tqdm.tqdm(load_all.get_all_okay_models(), desc="Loading models"):
            self.league.append(LeagueEntry(PPOPlayer(model, model_name), model, e.Rating()))

def get_env(meta: EloLearnig) -> GameEnv:
    sorted_league = sorted(meta.league, key=lambda x: x.rating, reverse=True)
    # get 10 best players
    player_instances = [l.player for l in sorted_league[:10]]

    if ENV_VEC_SIZE == 1:
        env = GameEnv(player_instances=player_instances)
        env = Monitor(env)
    else:
        env = sb3_vec_env.DummyVecEnv([lambda: GameEnv(player_instances=player_instances) for _ in range(ENV_VEC_SIZE)])
        env = sb3_vec_env.VecMonitor(env)

    return env

def print_league_for_all_models():
    meta = EloLearnig(K)

    meta.load_pretrained_models()

    update_league(meta)
    print_league(meta)

def get_initial_model(env: GameEnv) -> MaskablePPO:
    if USE_EXITING_MODELS:
        return MaskablePPO("MlpPolicy", env, n_steps=N_STEP, gamma=GAMMA, verbose=1, tensorboard_log=TENSORBOARD_PATH, device=DEVICE)

    meta = EloLearnig(K)

    exiting_models = [f for f in os.listdir(MODEL_PATH)]
    assert USE_EXITING_MODELS and len(exiting_models) != 0, "No existing models found"

    for model_name in exiting_models:
        model_name = f"{MODEL_PATH}/{model_name}"
        model = MaskablePPO.load(model_name, env, n_steps=N_STEP, gamma=GAMMA, verbose=1, tensorboard_log=TENSORBOARD_PATH, device=DEVICE)
        meta.league.append(LeagueEntry(PPOPlayer(model, model_name), model, e.Rating()))

    update_league(meta)

    best_player = max(meta.league, key=lambda x: x.rating).player
    assert type(best_player) == PPOPlayer
    return best_player.model


def get_model_for_trainig(meta: EloLearnig) -> MaskablePPO:
    env = get_env(meta)
    ppo_players = [l.player for l in meta.league if l.model is not None]

    # TODO: Sometimes start with a new model
    if len(ppo_players) == 0:
        return get_initial_model(env)

    # Get model with highest rating
    best_player = max(filter(lambda e: e.model, meta.league), key=lambda x: x.rating).player
    assert type(best_player) == PPOPlayer

    return MaskablePPO.load(best_player.model_name, env, n_steps=N_STEP, gamma=GAMMA, verbose=1, tensorboard_log=TENSORBOARD_PATH, device=DEVICE)

def update_league(data: EloLearnig):
    for l in data.league:
        l.rating = e.Rating()

    for _ in tqdm.tqdm(range(ELO_GAME_COUNT)):
        league_game_list = list(itertools.permutations(data.league, 2))

        if len(league_game_list) > GAME_COUNT_PER_LEVEL:
            league_game_list = random.sample(league_game_list, GAME_COUNT_PER_LEVEL)

        for a, b in tqdm.tqdm(league_game_list, leave=False):
            result = play_single_game(a.player, b.player)

            if result == GameRestult.PLAYER_0_WIN:
                a.rating, b.rating = data.elo.rate_1vs1(a.rating, b.rating)
            elif result == GameRestult.PLAYER_1_WIN:
                b.rating, a.rating = data.elo.rate_1vs1(b.rating, a.rating)
            else:
                a.rating, b.rating = data.elo.rate_1vs1(a.rating, b.rating, drawn=True)

        print_league(data)

def print_league(data: EloLearnig):
    for l in sorted(data.league, key=lambda x: x.rating, reverse=True):
        print(f"{l.player.name():40} - {l.rating}")

class EvalOnSucess(BaseCallback):

    def __init__(self, required_success_rate: float):
        self.required_success_rate = required_success_rate
        super().__init__()

    def _on_step(self) -> bool:
        if len(self.model.ep_success_buffer) == self.model.ep_success_buffer:
            success_rate = np.mean(self.model.ep_success_buffer)

            if success_rate >= self.required_success_rate:
                return False

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", type=int, default=ITERATIONS, help="Number of iterations")
    # Add flag to print league
    parser.add_argument("--print_league", action="store_true", help="Print league")

    args = parser.parse_args()

    if args.print_league:
        print_league_for_all_models()
        exit(0)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./model_checkpoints_elo/')
    stop_on_success_rate_callback = EvalOnSucess(0.95)
    callbacks = [checkpoint_callback, stop_on_success_rate_callback]

    metalearning = EloLearnig(K)
    metalearning.load_pretrained_models()
    update_league(metalearning)

    for i in range(META_LEARNING_ITERATIONS):
        print(f"Meta learning iteration {i}/{META_LEARNING_ITERATIONS}")
        model = get_model_for_trainig(metalearning)
        print(model.policy)

        # Learn the model
        model.learn(args.i, tb_log_name="PPO_elo", callback=[checkpoint_callback, stop_on_success_rate_callback], log_interval=LOG_INTERVAL, progress_bar=True)
        model_name = f"{MODEL_PATH}/m_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        model.save(model_name)

        metalearning.league.append(LeagueEntry(PPOPlayer(model, model_name), model, e.Rating()))

        # Update league
        update_league(metalearning)
        print_league(metalearning)
