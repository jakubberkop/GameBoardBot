from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from game_env import GameEnv
import stable_baselines3.common.vec_env as sb3_vec_env

# import pytracy
# pytracy.set_tracing_mode(pytracy.TracingMode.All)
# pytracy.add_path_to_filter("/")

callback = None

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=1, help="Number of parallel environments")

    args = parser.parse_args()

    if args.n == 1:
        env = GameEnv()
        env = Monitor(env)
    else:
        env = sb3_vec_env.DummyVecEnv([lambda: GameEnv() for _ in range(args.n)])
        env = sb3_vec_env.VecMonitor(env)

    model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1 , tensorboard_log="runs")

    iteration_count = 500_000

    model.learn(iteration_count, progress_bar=True, callback=callback)

    model.save(f"ppo_mask_huge_reward_at_the_end{iteration_count}")
    # del model # remove to demonstrate saving and loading

    # model = MaskablePPO.load("ppo_mask")
    # # print("Model loaded")
    # # obs, _ = env.reset()
    # # while True:
    #     # Retrieve current action mask
    #     action_masks = get_action_masks(env)
    #     action, _states = model.predict(obs, action_masks=action_masks)
    #     obs, reward, terminated, truncated, info = env.step(action)

    # evaluate_policy(model, env, n_eval_episodes=20, warn=True)