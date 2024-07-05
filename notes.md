To read:
- https://research.wdss.io/oh-hell/
- https://bennycheung.github.io/game-architecture-card-ai-3
- https://towardsdatascience.com/teaching-a-neural-network-to-play-cards-bb6a42c09e20
- https://arxiv.org/abs/1910.04376
- https://web.stanford.edu/class/aa228/reports/2019/final111.pdf
- https://arxiv.org/abs/2311.17305
- file:///home/jakub/Downloads/PhDThesis_Konrad_Godlewski_20221010.pdf
    - Monte carlo tree search
    - Q-Learning
    - Actor-Critic

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
- Off-policy RL maintains two distinct policies: A behavior policy and a target policy

- random utility model (RUM) - where choices are dependent on random state variable

- Multi-Agents Reinforcement Learning (MARL)