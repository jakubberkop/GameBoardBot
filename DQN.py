import random
from typing import Any, Dict, Iterable, List, NamedTuple
import numpy as np
# from operator import add
import collections
from game import GameStep.STATE_ERROR, GameState, get_game_score
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import copy
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import pytracy
# pytracy.set_tracing_mode(pytracy.TracingMode.All)

Transition = NamedTuple('Transition', [
    ('state', torch.Tensor),
    ('action', torch.Tensor),
    ('next_state', torch.Tensor),
    ('reward', torch.Tensor),
    ('done', torch.Tensor), 
])

class DQNAgent(torch.nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory: collections.deque[Transition] = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()

    def network(self):
        # Layers
        self.f1 = nn.Linear(21, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 9)

        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        # x = F.softmax(self.f4(x), dim=-1)
        x = F.relu(self.f4(x))
        return x
    
    def get_state(self, game: GameState, player_id: int):
        return np.asarray(game.to_state_array(player_id))

    def set_reward(self, game_state: GameState):
        self.reward = 0

        if game_state.state == GameStep.STATE_ERROR:
            self.reward = -10000
            return self.reward
        
        self.reward = get_game_score(game_state) * 100 + game_state.turn_counter

        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        # self.memory.append(Transition(state, action, next_state, reward, done))
        transition = Transition(
            torch.tensor(state, dtype=torch.float32, device=DEVICE),
            torch.tensor(action, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_state, dtype=torch.float32, device=DEVICE),
            torch.tensor([reward], dtype=torch.float32, device=DEVICE),
            torch.tensor([done], dtype=torch.bool, device=DEVICE),
        )
        self.memory.append(transition)

    def replay_new(self, memory: collections.deque[Transition], batch_size: int):
        # """
        # Replay memory.
        # """
        # if len(memory) > batch_size:
        #     minibatch = random.sample(memory, batch_size)
        # else:
        #     minibatch = memory
        # for state, action, reward, next_state, done in minibatch:
        #     self.train()
        #     torch.set_grad_enabled(True)
        #     target = reward
        #     next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
        #     state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
        #     if not done:
        #         target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
        #     output = self.forward(state_tensor)
        #     target_f = output.clone()
        #     target_f[0][np.argmax(action)] = target
        #     target_f.detach()
        #     self.optimizer.zero_grad()
        #     loss = F.mse_loss(output, target_f)
        #     loss.backward()
        #     self.optimizer.step()



        if len(memory) < batch_size:
            transitions: List[Transition] = memory
        else:
            transitions: List[Transition] = random.sample(memory, batch_size)

        # torch.tensor(, dtype=torch.float32, device=device)
        
        # Change the type of each element in the transition to a tensor
        # and put it in a tuple
        # transitions = [Transition(
        #     torch.tensor(t.state, dtype=torch.float32, device=DEVICE),
        #     torch.tensor(t.action, dtype=torch.float32, device=DEVICE),
        #     torch.tensor(t.next_state, dtype=torch.float32, device=DEVICE),
        #     torch.tensor([t.reward], dtype=torch.float32, device=DEVICE),
        #     torch.tensor([t.done], dtype=torch.bool, device=DEVICE),
        #     )
        #     for t in transitions
        # ]

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        # non_final_mask = ~torch.tensor(batch.done, dtype=torch.bool, device=DEVICE)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not True, batch.done)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s, done in zip(batch.next_state, batch.done) if done is not True])

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        next_state_values = torch.zeros(len(transitions), device=DEVICE)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.forward(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute the actual Q values
        pre_gather_output = self.forward(state_batch)
        output = pre_gather_output.gather(1, action_batch.unsqueeze(1))

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()

        # UserWarning: Using a target size (torch.Size([32, 1])) that is different to the input size (torch.Size([32, 1, 32])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
        loss = criterion(expected_state_action_values.unsqueeze(1), output)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 21)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 21)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)

        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()

        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()