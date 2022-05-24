# Ideas & Overview:

- RL: Neural networks approximate value functions to **optimize long-term expected return**
- Reward function engineering is challenging and important, thus we want *simple* **sparse binary reward**
- **HER**: off-policy deep Q-learning for multi-goal RL problems with sparse rewards
    - for some randomly achieved state, find its corresponding goal (if the goal were there, then the trajectory would be how to achieve it)
    - essentially unguided random search from the starting position, so far goals are very hard to reach, similar to Dijkstra

- **HGG**: extend HER by generating intermediate hindsight goals (implicit curriculum) to guide exploration towards goal
    - using euclidean norm as distance between potential hindsight goal and target goal
    Thus cannot circumvent obstacles

# Details:

**RL**: Agent can at every time step $t \in {0, 1, ...}$
    (1) observe current **state** $s_t$ from environment,
    (2) decide on an **action** $a_t$ to take based on **policy** $\pi(s_t)$ and the current state,
    (3) receive a **reward** $r_t = r(s_t, a_t)$ based on the reward funciton,
    (4) environment then changes state according to the **transition function** $s_{t+1} = p(s_t, a_t)$

**Markov Decision Process**: find a good decision policy in a stochastic environment to maximize reward
**Given**:
- **State space** $\mathcal{S}$: set of possible states
- **Initial state distribution** $\mathcal{S_0: S \to \R}$: probabilistic distribution of initial state 
- **Action space** $\mathcal{A}$: set of possible actions
- **Transition function** $p : \mathcal{S \times A \to S}$: maps current state and applied action to its resulting state (could be stochastic: return a distribution instead)
- **Reward function** $r : \mathcal{S \times A \to \R}$: reward that the agent receives for the particular action at the particular state
- **Discount factor** $\gamma \in (0, 1)$: controls how much the agent favors immediate reward as opposed to long-term reward; close to 1 means long term reward is valued highly and close to 0 means short-term reward is much more favored

**Find**:
- **Policy** $\pi : \mathcal{S \to A}$: maps current state to appropriate action in order to maximize expected discounted cumulative return (could be stochastic: return a distribution instead)

**Using**:
- **State-Value function** $V^\pi(s) = \mathbb{E}_{...}[\sum_{t=0}^{\infin} \gamma^t r(s_t, a_t)]$: expected cumulative discounted reward starting from state $s$ and following policy $\pi$
- **Action-Value function** $Q^\pi(s, a) = \mathbb{E}_{...}[\sum_{t=0}^{\infin} \gamma^t r(s_t, a_t)]$: expected cumulative discounted reward starting from state $s$, taking action $a$ and following policy $\pi$
- **Optimal policy/state-value/action-value functions** $\pi^*, V^*, Q^*$
- **Bellman Optimality Equation**: self-consistency equation that splits the problem into *optimal immediate decision* and the *optimal rest of the decisions*
$V^*(s) = \mathbb{E}_{s' \sim p(s, a)}[\mathop{max}_{a \in \mathcal{A}} r(s, a) + \gamma V^*(s')]$
$Q^*(s,a) = \mathbb{E}_{s' \sim p(s, a)}[r(s, a) + \gamma \mathop{max}_{a' \in \mathcal{A}} Q^*(s', a')]$

- **Q-Learning**: If $Q^*$ can be found (or approximated via $Q_\theta$), then the action of the optimal policy is $\pi^*(s) = \mathop{argmax}_a Q_\theta(s, a)$; since $Q$ is a general description of value depending on state-action, **off-policy** learning is possible, meaning that any data from any point during training can be used to update $Q_\theta$ (which enables hindsight replay, etc.)

- **DDPG**: model-free, off-policy, actor-critic RL algorithm for continuous action space
    - **Actor** $\pi(s | \theta^\pi)$: neural network that represents the current policy (deterministic)
    - **Critic** $Q(s,a | \theta^Q)$: neural network that approximates the actors actor-value function $Q^\pi$
    - **Target networks** $\pi'(s | \theta^{\pi'}), Q'(s,a | \theta^{Q'})$ with the target weights are copies of actor's and critic's weights $\theta^\pi, \theta^Q$ that lag behind with a factor $0 < \tau < 1$:
    $\theta^{Q'} \gets \tau\theta^Q + (1-\tau)\theta^{Q'}$
    This stabilizes the update trajectory of the Q-learning, kinda like momentum but different (momentum accelerates and stabilizes the trajectory, polyak averaging hinders overshooting an optimum)

- **UVFA**: extend DDPG as **multi-goal** by adding goal as an input along side with each state
    - **Goal space** $\mathcal{G}$: set of all possible goals
    - **Target goal distribution** $\mathcal{G_T: G \to \R}$: probabilistic distribution of target goals that would be sampled at the beginning of each episode, resulting in the initial state-goal pair $(s_0,g)$
    - Goals can be interpreted as an extension of the state space, thus $s_t$ becoming $s_t \parallel g$ for the policy and value functions

- **HER**: train multi-goal UVFA setups with sparse rewards more efficiently
    - $f_g : \mathcal{S} \to \{0,1\}$ of each and every goal $\mathcal{g \in G}$: predicate that returns 1 if the given state $s$ satisfies the goal $g$
    - $m: \mathcal{S \to G}$: mapping where every state $s$ has at least one goal $g$ that is considered achieved, i.e. $\forall s \in \mathcal{S}: f_{m(s)}(s) = 1$
    - $r_g(s,a)$ which returns -1 if goal is not reached and 0 if goal is reached (constant negative reward)
    - Each episode consists of a **trajectory** $\tau = (s_0, s_1, ..., s_T)$ sequence of reached states, and the transitions $(s_t \parallel g, a_t, r_t, s_{t+1} \parallel g)$ between each state
    - **Replay buffer** $R$: contains not only the past transitions with their original goals, but also some with **hindsight goals** $g' := \mathbb{S}(\tau)$, like $(s_t \parallel g', a_t, r_t, s_{t+1} \parallel g')$, where $\mathbb{S}(\tau)$ a **replay strategy** that chooses an appropriate hindsight goal (such as final state in that trajectory/episode)

- (**EBP**)
- **HGG**: 