from typing import List
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from game import get_game_score

import game_env
import datetime
import pathlib
import tqdm

from torch.utils.tensorboard import SummaryWriter

gym.register("GAME", entry_point=game_env.GameEnv, max_episode_steps=10000)
plt.ion()

# if GPU is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print(f"Using device: {DEVICE}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


tensorboard_writer = SummaryWriter()

class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory: deque[object] = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.softmax(self.layer5(x), dim=1)

        return x

NUM_EPISODES = 60000
SHOW_GRAPH_FREQUENCY = 30

# Learning
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
SHOULD_SAVE_RESULTS = True
VERBOSE = False
TRACY = False


if TRACY:
    import pytracy
    pytracy.set_tracing_mode(pytracy.TracingMode.All)


class Trainer:

    def __init__(self):
        self.steps_done: int = 0
        self.env: game_env.GameEnv = gym.make("GAME", render_mode="human" if VERBOSE else None)

        # Get number of actions from gym action space
        n_actions: int = int(self.env.action_space.n)
        # Get the number of state observations
        state, info = self.env.reset()
        n_observations = len(state)

        self.scores: List[float] = []
        self.game_lengths: List[int] = []
        self.actions: List[int] = []
        self.rewards: List[int] = []

        self.policy_net = DQN(n_observations, n_actions).to(DEVICE)
        self.target_net = DQN(n_observations, n_actions).to(DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def select_action(self, state: torch.Tensor, legal_moves: np.ndarray) -> torch.Tensor: #TODO: MAKE IT an int (faster)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        assert legal_moves.sum() > 0, "No legal moves"

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                policy_net_output = self.policy_net(state)

                # Apply the mask to the input tensor
                masked_input = policy_net_output.masked_fill(~torch.tensor(legal_moves, dtype=torch.bool, device=DEVICE), float('-inf'))

                # Perform argmax on the masked input
                action = torch.argmax(masked_input)

                # Print the result

                # torch masked select
                # https://pytorch.org/docs/stable/generated/torch.masked_select.html

                # legal_moves_tensor = torch.tensor(legal_moves, dtype=torch.float32, device=DEVICE)

                # Mask the policy_net_output with the legal_moves
                # masked_policy_net_output = policy_net_output * legal_moves_tensor

                # Get the index of the highest value
                # action = masked_policy_net_output.max(1).indices.view(1, 1).item()

        else:
            # Give legal_moves vector
            action = self.env.action_space.sample(legal_moves)

        assert legal_moves[action] == 1, "Illegal move"

        return torch.tensor([[action]], device=DEVICE, dtype=torch.float32)

    def load(self):
        pass
        # Load newest policy_net weights

        # Get the newest policy_net weights
        # policy_net_weights = sorted(glob.glob("results/*/policy_net.pt"))[-1]
        # print(f"Loading policy_net weights from {policy_net_weights}")
        # policy_net.load_state_dict(torch.load(policy_net_weights))


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        
        not_none_next_states = [t for t in batch.next_state if t is not None]
        not_none_next_count = len(not_none_next_states)
        # print(f"not_none_next_count: {not_none_next_count}")

        if not_none_next_count == 0:
            # print("Skipping optimize_model() because not_none_next_state_count == 0")
            return

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)

        non_final_next_states = torch.cat(not_none_next_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # action_batch_int = action_batch.type(torch.int64)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.type(torch.int64))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" self.target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.L1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot_scores(self, show_result=False):
        plt.figure(1)
        scores_t = torch.tensor(self.scores, dtype=torch.float)
        game_lengths_t = torch.tensor(self.game_lengths, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('STUFF')
        plt.plot(scores_t.numpy(), label="scores")
        plt.plot(game_lengths_t.numpy(), label="Game Lengths")


        # Take 100 episode averages and plot them too
        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label="MEAN scores", )

            # Means of game lengths
            mean_game_lengths = game_lengths_t.unfold(0, 100, 1).mean(1).view(-1)
            mean_game_lengths = torch.cat((torch.zeros(99), mean_game_lengths))
            plt.plot(mean_game_lengths.numpy(), label="MEAN Game Lengths", )
        plt.legend()

        # Plot histogram of actions
        plt.figure(2)
        plt.clf()
        plt.title('Actions')
        plt.xlabel('Action')
        plt.ylabel('Frequency')

        plt.hist(self.actions, bins=range(self.env.action_space.n))

        # Plot the trend of rewards
        plt.figure(3)
        plt.clf()
        plt.title('Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.rewards)

        # plot mean and median of rewards
        if len(self.rewards) >= 100:
            mean_rewards = torch.tensor(self.rewards, dtype=torch.float).unfold(0, 100, 1).mean(1).view(-1)
            mean_rewards = torch.cat((torch.zeros(99), mean_rewards))
            plt.plot(mean_rewards.numpy(), label="MEAN Rewards", )


        plt.pause(0.10)  # pause a bit so that plots are updated


    def main(self):
        for i_episode in tqdm.tqdm(range(NUM_EPISODES)):
            # Initialize the environment and get it's state
            state, info = self.env.reset()

            state: torch.Tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            for _ in count():
                action = self.select_action(state, info["legal_moves"])
                self.actions.append(action.item())

                observation, reward, terminated, truncated, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=DEVICE) #TODO: This can be optimized by using not copy the reward to the GPU every time
                done = terminated or truncated

                self.rewards.append(reward.item())

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Store the transition in self.memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.scores.append(get_game_score(self.env.game_state))
                    self.game_lengths.append(self.env.game_state.turn_counter)

                    tensorboard_writer.add_scalar("score", get_game_score(self.env.game_state), i_episode)
                    tensorboard_writer.add_scalar("game_length", self.env.game_state.turn_counter, i_episode)

                    tensorboard_writer.flush()
                    # if i_episode % SHOW_GRAPH_FREQUENCY == 0:
                    #     self.plot_scores()
                    break

        if SHOULD_SAVE_RESULTS:
            self.save_results()

        self.plot_scores(show_result=True)
        plt.ioff()
        plt.show()


    def save_results(self):
        # Create a new folder for the results
        result_dir = pathlib.Path(f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        result_dir.mkdir(parents=True, exist_ok=True)

        # Learning
        # Save the results
        torch.save(self.policy_net.state_dict(), result_dir / "policy_net.pt")
        self.plot_scores()
        plt.savefig(result_dir / "plot.png")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.main()
    trainer.save_results()
