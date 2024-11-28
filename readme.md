# Introduction

This project aims to develop an AI capable of playing the board game "Pan tu nie stał". The objective of the game is for players to collect as many items from shops as possible by strategically placing people in queues. The queue with the most points at the end of the round earns the points.
The project implements several key components:

 - Game Mechanics (game.py): Defines the rules and flow of the game.
 - Gym Environment (game_env.py): Creates a custom environment for training the AI using the OpenAI Gym framework.
 - AI Training (sb3_maskable.py): Utilizes the Proximal Policy Optimization (PPO) algorithm, incorporating a mask on the action space to ensure only valid moves are chosen by the AI.

In addition to the standard PPO approach, this project introduces an innovative concept: a league of models, where different AI models compete against each other. The top-performing models are selected to form the next generation of models. This evolutionary approach helps improve the AI’s performance over time. For more details, see [PPO Maskable with ELO rating](#ppo-maskable-with-elo-rating) section.

## Quick start

Install dependencies
`pip install -r requirements.txt`

Training project:
`python3 sb3_maskable.py -n 10`

Run tensorboard:
`tensorboard --logdir=runs/`

Evaluate model results:
`python3 evaluate.py`

Play against the model:
`python3 evaluate.py --human`

> Notes: evaluate.py hard codes the ppo model path, so it needs to be updated if the model is saved in a different location.


Example output
```
RandomPlayer         vs RandomPlayer        :  52%-42% [ 0]%  5228-4187:[585]                                      
RandomPlayer         vs PPO Player          :   3%- 1% [ 1]%  254-147:[9599] 
PPO Player           vs RandomPlayer        :   1%- 0% [ 1]%  94-38:[9868] 
PPO Player           vs PPO Player          :   0%- 0% [ 1]%  6-5:[9989]
```

## PPO Maskable with ELO rating
A crucial element in reinforcement learning (RL) is the model's ability to explore the state space, especially in complex domains like board games, where the problem space is large and diverse. Initially, the PPO model was trained by playing against a random opponent, which it quickly learned to beat. The next step involved training the model to play against itself, which posed a greater challenge but was also quickly mastered. Subsequently, the model was trained to play against another model that had been trained through self-play, further increasing the difficulty.

To automate and enhance this training process, I introduced the concept of a league of models, where different models compete against each other, and the best models are selected for the next generation. This competitive framework is driven by an ELO rating system, commonly used in chess to rank players based on their performance. The ELO system not only helps identify the top-performing models but also provides valuable feedback to each model, indicating its relative strength.

By incorporating the ELO rating mechanism into the PPO training, models were able to achieve significantly higher scores compared to using the PPO algorithm in isolation. The introduction of this competitive structure accelerates learning and results in stronger, more capable models over time.

## Results.
| Model | ELO |
| --- | --- |
| ppo_elo/m_2024-08-19_19-14-00.zip | 1307.0633654577612 |
| ppo_elo/m_2024-08-11_20-57-13.zip | 1256.9805243705578 |
| ppo_elo/m_2024-08-11_20-50-01.zip | 1250.8953925303938 |
| ppo_elo/m_2024-08-11_21-03-04.zip | 1248.07208780831 |
| ppo_elo/m_2024-08-12_22-56-02.zip | 1247.4418892502592 |
| ppo_elo/m_2024-08-11_21-01-24.zip | 1247.4316829509469 |
| ppo_elo/m_2024-08-12_22-50-46.zip | 1245.3225497873134 |
| ppo_elo/m_2024-08-11_21-00-57.zip | 1240.8013225811465 |
| ppo_elo/m_2024-08-12_22-56-31.zip | 1240.6502433059663 |
| ppo_elo/m_2024-08-12_22-54-18.zip | 1236.2963155431853 |
| ppo_elo/m_2024-08-12_22-49-46.zip | 1235.5060524975195 |
| ppo_elo/m_2024-08-11_20-42-30.zip | 1232.7708238333132 |
| ppo_elo/m_2024-08-13_19-03-00.zip | 1232.7003461857066 |
| ppo_elo/m_2024-08-11_20-33-16.zip | 1225.959790183256 |
| ppo_elo/m_2024-08-12_22-52-55.zip | 1224.6350725191814 |
| ppo_elo/m_2024-08-12_22-47-26.zip | 1214.080094021451 |
| ppo_elo/m_2024-08-11_20-55-39.zip | 1210.967503108083 |
| ppo_elo/m_2024-08-13_18-56-33.zip | 1205.2177753776016 |
| ppo_elo/m_2024-08-11_20-54-03.zip | 1203.256282860005 |
| ppo_elo/m_2024-08-13_19-01-50.zip | 1202.9728449304916 |
| ppo_elo/m_2024-08-12_22-50-11.zip | 1202.1465762314026 |
| ppo_elo/m_2024-08-13_18-55-31.zip | 1201.167090222161 |
| ppo_elo/m_2024-08-11_20-58-34.zip | 1200.3761502907435 |
| ppo_elo/m_2024-08-12_22-51-32.zip | 1198.5916414095293 |
| ppo_elo/m_2024-08-12_22-48-32.zip | 1197.1844737631884 |
| ppo_elo/m_2024-08-12_22-54-21.zip | 1193.0910709512352 |
| ppo_elo/m_2024-08-12_22-53-34.zip | 1184.3941963313116 |
| ppo_elo/m_2024-08-11_20-52-02.zip | 1184.3013547192645 |
| ppo_elo/m_2024-08-11_20-49-02.zip | 1183.8547294487512 |
| ppo_elo/m_2024-08-11_20-59-04.zip | 1182.9381551327472 |
| ppo_elo/m_2024-08-12_22-53-54.zip | 1180.4279202127702 |
| ppo_elo/m_2024-08-11_20-35-00.zip | 1178.354977023263 |
| ppo_elo/m_2024-08-12_22-50-49.zip | 1174.439505151145 |
| ppo_elo/m_2024-08-11_20-52-55.zip | 1173.235228631692 |
| ppo_elo/m_2024-08-12_22-47-23.zip | 1172.224459315952 |
| ppo_elo/m_2024-08-11_20-54-43.zip | 1171.4937074135 |
| ppo_elo/m_2024-08-11_20-35-25.zip | 1164.7052552996784 |
| ppo_elo/m_2024-08-11_20-56-19.zip | 1163.587647447371 |
| RandomPlayer | 1160.7971806685368 |
| ppo_elo/m_2024-08-11_20-30-40.zip | 1160.2346597480548 |
| ppo_elo/m_2024-08-11_20-48-11.zip | 1159.2229448455869 |
| ppo_elo/m_2024-08-12_22-53-40.zip | 1157.2191942296704 |
| ppo_elo/m_2024-08-11_20-34-08.zip | 1153.4361568347529 |
| AlwaysFirstPlayer | 1140.052189527113 |
| ppo_elo/m_2024-08-13_19-00-48.zip | 1130.1839154793122 |
| AlwaysLastPlayer | 1123.3176605688175 |


TODO: Compare PPO with PPO ELO rating

## Experiments

### First steps
The game has a discrete action space, where not all actions are valid in every state. This makes it difficult for some algorithms to learn the game. First approach was to have the model learn which actions are valid. I tried to achieve this by providing negative reward, whenever a model provided invalid action. Secondary the game would be restarted on invalid action - this was to make the model learn faster. I also provided a small reward for every valid action, to make the model learn faster.
This approach was not successful, as the model would still provide invalid actions. I think this was because of a sparse reward signals.

### Valid action mask
Then I tried to mask the model prediction with a valid action mask, so that the most propable available action would be chosen. This is a bit tricky as there was no feedback to the model about the invalid actions. Maybe the model will train better if it knew whether the actions were valid or not.

### Decision Transformer
As oposed to other algorithms used in my experiments Decision Transformer is a supervised learning algorithm, that requires a dataset of games to be trained, which required me to prepare a dataset for it. This is a bit problematic, as the dataset itself needs explore the state space well enough to be useful, and the dataset generation process is quite slow. This is with contrast to reinforcement learning algorithms, that can explore the state space during training.
I didn't manage to make the Decision Transformer work, may be wortwhile revisiting in the future.

### Ideas
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
- sb3_maskable_elo.py - implements experimental ELO rating system for the models.
- decision_transformer/* - experiments with decision transformer. 

## Notes

### Concecepts
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

### Resources
- https://rlcard.org/
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
- https://www.researchgate.net/publication/350660009_Self-Play_Reinforcement_Learning_with_Comprehensive_Critic_in_Computer_Games
- http://proceedings.mlr.press/v119/bai20a/bai20a.pdf
- https://arxiv.org/abs/2403.00841
- https://medium.com/applied-data-science/how-to-train-ai-agents-to-play-multiplayer-games-using-self-play-deep-reinforcement-learning-247d0b440717
- https://github.com/IDSIA/sacred/tree/master
- https://vivekratnavel.github.io/omniboard/#/README?id=license
- https://arxiv.org/abs/2303.05197 - Hearthstone
- https://web.stanford.edu/class/aa228/reports/2019/final111.pdf
- https://arxiv.org/abs/2311.17305
- https://www.marl-book.com/download/marl-book.pdf
- https://arxiv.org/pdf/2110.02793
- https://arxiv.org/pdf/1710.03641
- https://www.gm.th-koeln.de/ciopwebpub/Kone15c.d/TR-TDgame_EN.pdf
- https://www.diva-portal.org/smash/get/diva2:1680520/FULLTEXT01.pdf
- https://arxiv.org/pdf/2307.09905
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10154465
- https://arxiv.org/abs/2006.14171