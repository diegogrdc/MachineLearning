# Reinforcement Learning

> Reinforcement Learning: ML algorithm method based on rewarding desired behaviors and/or punishing undesired ones. The main idea is to tell a system what to do, and let it figure out how to do it through the reward system. 

On reinforcement learning, we want to find a function that maps from a state $s$ to an action $a$. To do this, we will use *reward functions*. It is a really powerful algorithm, because you tell the program what to do, but you don't tell it how to do it. The program discovers the how on it's own. 

## Applications
- Controlling robots
- Factory optimizations
- Financial (stock) trading
- Playing games

## Return

How do we know if a set of rewards is better or worse than another?. We can use the *return*. This highlights the importance of not just a good reward, but a reward that does not take a really long time to get to, making our algorithm a bit *impatient*.

To take that into consideration, we can use a discount factor $\gamma$. If we get a set of rewards $R$ in order, then our formula would be:

> $R_1 + \gamma R_2 + \gamma^2R_3+...+$ (until terminal state)

We usually use a number close to $1$ for $\gamma$, like $0.99$ or $0.999$.

The returns is the final reward we get after considering the discount factor. This return depends on the set of actions that we take. On negative rewards, the algorithm pushes them as far in the future as possible.

## Policy

In reinforcement learning, our goal is to come up with a policy $\pi$. A policy is  a function $\pi(s) = a$, mapping from states to actions, that tells you what action $a$ to take in a given state $s$.

Our goal is to find a policy $\pi$ to maximize the return.

> Markov Decision Process (MDP): The future only depends only on where you are now, not on how you got here.

## State-Action Value Function

A function typically denoted by $Q(s, a)$. 

This function will give you the return if you start in state $s$, take action $a$ once, and behave optimally after that.

### Picking actions

If we have our function $Q(s, a)$, we know the optimal action is the one with the max possible return $Q(s, a)$ from state $s$ (out of all possible actions $a$).

## Bellman Equation

We can use the bellman equation to help us calculate $Q(s, a)$

Given:
- $Q(s, a)$: Return if we start in $s$, execute $a$ once, and behave optimally after
- $s$: Current state
- $R(s)$: Reward of state $s$
- $a$: Current action
- $s'$: State you get to after taking action $a$
- $a'$: Action you take in state $s'$

Then, we get:

> $Q(s, a) = R(s) + \gamma \max_{a'}Q(s',a')$ 

Or, if we are in a terminal state:

> $Q(s, a) = R(s)$

## Random (Stochastic) Environments

Real environments are not perfect. There is always a random factor that can affect actions or states. To model this, we can associate each action with a probability. We call this a stochastic environment. 

In this environments, we would look to maximize the *expected return*.

>  $Q(s, a) = R(s) + \gamma \ E[max_{a'}Q(s',a')]$ 

## Continuous State Spaces

Most real examples have continuous states (more complex that discrete states). An example is the position of an object $(x, y)$, composed by real numbers, or how fast the $x$ position is changing ($\dot{x}$). 

## Deep Reinforcement Learning

We will train a NN to train/approximate $Q(s, a)$. This network inputs the current state $s$, and computes or approximates $Q(s, a)$. The output layer has a number for each possible action that we can execute, which represents the approximation of the return given that we choose that action.

If we have a trained NN, then whenever we need to make a decision, we can input all possible actions with out current state, and choose the one with the biggest return. 

### Training the NN

We can use Bellman Equations to create a training set with lots of examples.

At first, we don't have any information. So we can do random steps, each one giving us a tuple $(s, a, R(s), s')$. Having lots of these tuples after a simulation, we can create train examples with $x = (s, a)$ and $y = R(s) + \gamma * max_{a'}Q(s',a')$ for each action $a$. 

Notice that at first we don't know $Q(s,a)$, but we will start with random guesses, and improve them with time.

## Algorithm

The following algorithm is called DQN, or Deep Q-Network.

- Initialize NN randomly as a guess of $Q(s, a)$
- Repeat
  - Take actions in simulation. Get tuples $(s, a, R(s), s')$
  - Store 10,000 most recent tuples.
    - We call this the replay buffer
  - Train NN
    - Create training set with stored tuples
      - $x = (s, a)$
      - $y = R(s) + \gamma * max_{a'}Q(s',a')$
  - Train $Q_{new}$ such that $Q_{new}(s, a) \approx y$
  - Set $Q = Q_{new}$ 

## $\epsilon$-Greedy Policy

When simulation, we still don't know what is the best action to take in each state. so how do we choose?
- With probability 0.95, choose the action that maximizes $Q(s, a)$.
  - Greedy steps
- With probability 0.05, choose an action $a$ randomly.
  - Exploration steps

We add this, as initial guesses might push the algorithm to avoid some good actions, and we need to try them to figure out they are good, even if it does not look like it at the start. We call those random steps *exploration steps*. 

> $\epsilon$ is the probability of exploration (in this example 0.05)

A trick is to start with a high $\epsilon$, and gradually decrease it, to encourage exploration when we don't have a lot of information, and encourage greedy steps when we have learned more information. 

## Refinements

### Mini batch

On supervised learning, if we had a really big dataset, instead of using all examples for each gradient step, we use a smaller batch size, making more gradient updates, which might create some noisy gradient path, but speeds up the algorithm.

On reinforcement learning, this would mean, instead of using all stored tuples to create the dataset, create a smaller dataset to have faster iterations.

### Soft updates

On $Q = Q_{new}$, we override the whole NN with the new one. However, we know this steps might sometimes not be in the direction we are looking for, and doing this might make us lose useful information, instead, we can do:

> $Q = 0.99Q + 0.01Q_{new}$

This makes gradual changes to parameters, making *soft updates*, and it makes it more likely for the algorithm to converge. 