# Quick start

Install dependencies
`pip install -r requirements.txt`

Training project:
`python3 sb3_maskable.py -n 10`

Run tensorboard:
`tensorboard --logdir=runs/`

Evaluate model results: `python3 evaluate.py`

> Notes: evaluate.py hard codes the ppo model path, so it needs to be updated if the model is saved in a different location.


Example output
```
RandomPlayer         vs RandomPlayer        :  52%-42% [ 0]%  5228-4187:[585]                                      
RandomPlayer         vs PPO Player          :   3%- 1% [ 1]%  254-147:[9599] 
PPO Player           vs RandomPlayer        :   1%- 0% [ 1]%  94-38:[9868] 
PPO Player           vs PPO Player          :   0%- 0% [ 1]%  6-5:[9989]
```

# Experiments

## First steps
The game has a discrete action space, where not all actions are valid in every state. This makes it difficult for some algorithms to learn the game. First approach was to have the model learn which actions are valid. I tried to achieve this by providing negative reward, whenever a model provided invalid action. Secondary the game would be restarted on invalid action - this was to make the model learn faster. I also provided a small reward for every valid action, to make the model learn faster.
This approach was not successful, as the model would still provide invalid actions. I think this was because of a sparse reward signals.

## Valid action mask
Then I tried to mask the model prediction with a valid action mask, so that the most propable available action would be chosen. This is a bit tricky as there was no feedback to the model about the invalid actions. Maybe the model will train better if it knew whether the actions were valid or not. 
Worth noting that this implementation was self made, as a 

## Decision Transformer
As oposed to other algorithms used in my experiments Decision Transformer is a supervised learning algorithm, that requires a dataset of games to be trained, which required me to prepare a dataset for it. This is a bit problematic, as the dataset itself needs explore the state space well enough to be useful, and the dataset generation process is quite slow. This is with contrast to reinforcement learning algorithms, that can explore the state space during training.
I didn't manage to make the Decision Transformer work, may be wortwhile revisiting in the future.

## Current focus
Extending the state space with more data for the model.

# Ideas
- Test MaskedPPO with removed randomness
    - Create player that always chooses highest card, and draws randomly if it has no cards
    - Remove randomness from shop decisions - always choose the highest item
- Add queue scores to the state
- Consider different algorithms with mask
    - DQN
    - Decision Transformer
    - Monte Carlo Tree Search - the phd thesis mentions
- Consider that the game is a pvp. Explore the algorithms that are used in pvp games:
    - Multi-Agents Reinforcement Learning (MARL)
    - AlphaZero
    - Q-Learning
    - Actor-Critic
- Provide a vector of possible actions in the state space

# Project structure
Ordered by importance:

- game.py - implements the "pan to nie stał" board game. It simplfiies the game logic a bit, to make it easier to train the AI - for example, it automates few steps that are usually done manually.
- game_env.py - implements the gym environment for the game. Assumes that the game is played agains one opponent, and the opponent is a random player (might be changed in the future).
- evaluate.py - evaluates the model by playing it against other player implementations. Currently implements 
- sb3_maskable.py - trains the model using Proximal Policy Optimization (PPO) algorithm, with mask applied to the action space to only allow valid moves. Currently my main focus.

- decision_transformer.py, dataset.py - experiments with decision transformer. 
- AA.py, game_torch.py - experiments using DQN (Deep Q-Network) algorithm. Wasn't able to make it work. I also used the implementation of 
- game_first.py - first implementation of parts of the game. Might be useful?
- all other files - random experiments, tests, etc.

# Notes

To read:
- https://research.wdss.io/oh-hell/
- https://bennycheung.github.io/game-architecture-card-ai-3
- https://towardsdatascience.com/teaching-a-neural-network-to-play-cards-bb6a42c09e20
- https://arxiv.org/abs/1910.04376
- https://web.stanford.edu/class/aa228/reports/2019/final111.pdf
- https://arxiv.org/abs/2311.17305
- https://www.bip.pw.edu.pl/content/download/59138/554233/file/A.Kawala-Sterniuk%20recenzja.pdf
    - Monte carlo tree search
    - Q-Learning
    - Actor-Critic

- https://www.youtube.com/watch?v=rbZBBTLH32o

Resources:
- https://rlcard.org/

Concecepts:
- MDP - Markov decision process
    - discrete-time stochastic control process
    - goal => good "policy"
- POMDP - Partially observable MDP
- Curse of Dimensionality - phenomena that arise in high-dimensional spaces that do not occur in low-dimensional
- Model-free Algorithm - does not estimate the transition probability distribution (and the reward function) associated with the MDP
- SARSA - State–action–reward–state–action
    - aka. MCQ-L Modified Connectionist Q-Learning
- ε-greedy -  0 < ε < 1 is a parameter controlling the amount of exploration vs. exploitation
state-value function Vπ(s) defined as, expected discounted return starting with state s, i.e. S0 =s , and successively following policy π
- N-policy - With on-policy RL, actions are generated in response to observed environment states using a certain RL policy
- Off-policy RL - maintains two distinct policies: A behavior policy and a target policy
- random utility model (RUM) - where choices are dependent on random state variable
- Multi-Agents Reinforcement Learning (MARL)