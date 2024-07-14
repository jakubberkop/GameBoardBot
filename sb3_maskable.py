from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
# from torch.utils.tensorboard import SummaryWriter

from game import RandomPlayer, AlwaysFirstPlayer, AlwaysLastPlayer
from game_env import GameEnv
import stable_baselines3.common.vec_env as sb3_vec_env

# import pytracy
# pytracy.set_tracing_mode(pytracy.TracingMode.All)
# pytracy.add_path_to_filter("/")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("-i", type=int, default=100_000, help="Number of iterations")

    args = parser.parse_args()

    oponent_types = [RandomPlayer, AlwaysFirstPlayer, AlwaysLastPlayer]

    if args.n == 1:
        env = GameEnv(oponent_types)
        env = Monitor(env)
    else:
        env = sb3_vec_env.DummyVecEnv([lambda: GameEnv(oponent_types) for _ in range(args.n)])
        env = sb3_vec_env.VecMonitor(env)
        # env = sb3_vec_env.SubprocVecEnv([lambda: GameEnv() for _ in range(args.n)])

    # model = MaskablePPO.load("ppo_mask_extended_1000000_2024-07-08_20-57-50", env, verbose=1, tensorboard_log="runs")
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path='./model_checkpoints/')

    model = MaskablePPO("MlpPolicy", env, n_steps=64, gamma=0.8, verbose=1, tensorboard_log="runs")

    model.learn(args.i, callback=checkpoint_callback, log_interval=1, progress_bar=True)

    print("Training finished")
    import datetime
    model.save(f"ppo_mask_extended_{args.i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    # del model # remove to demonstrate saving and loading

    # # print("Model loaded")
    # # obs, _ = env.reset()
    # # while True:
    #     # Retrieve current action mask
    #     action_masks = get_action_masks(env)
    #     action, _states = model.predict(obs, action_masks=action_masks)
    #     obs, reward, terminated, truncated, info = env.step(action)

    # evaluate_policy(model, env, n_eval_episodes=20, warn=True)